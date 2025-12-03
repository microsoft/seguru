#![no_std]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]

use core::f32;

use core::ops::Add;
use crunchy::unroll;
use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk::ScopeUniqueMap;
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Grid2BlockScope, Thread};
use gpu::prelude::*;
use gpu::sync::{Atomic, SharedAtomic};

/*
__global__ void encoder_forward_kernel3(Float4* out,
                               const int* inp, const Float4* wte, const Float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if (idx < N) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        out[b * T * C4 + t * C4 + c4] = add_Float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
    }
}
*/

#[gpu::cuda_kernel]
pub fn encoder_forward_kernel3(
    out: &mut [Float4],
    inp: &[i32],
    wte: &[Float4],
    wpe: &[Float4],
    B: u32,
    T: u32,
    C: u32,
) {
    let mut out = chunk_mut(out, MapContinuousLinear::new(1));
    let C4 = C / 4;
    let bid_x = block_id::<DimX>();
    let bdim_x = block_dim::<DimX>();
    let tid_x = thread_id::<DimX>();
    let idx = bdim_x * bid_x + tid_x;
    let N = B * T * C4;
    if idx < N {
        let bt = idx / C4;
        let b = bt / T;
        let t = bt % T;
        let c4 = idx % C4;
        let ix = inp[(b * T + t) as usize] as u32;

        // bt = b * T + t, bt * C4 + c4 = idx
        // b * T * C4 + t * C4 + c4 = idx
        // out is local so we are at 0
        out[0] = wte[(ix * C4 + c4) as usize].add(wpe[(t * C4 + c4) as usize]);
    }
}

/*
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}
*/
/// really bad naive kernel with atomicAdd
#[gpu::cuda_kernel]
pub fn encoder_backward_kernel(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: i32,
    T: i32,
    C: i32,
) {
    let bdim_x = block_dim::<DimX>() as i32;
    let bid_x = block_id::<DimX>() as i32;
    let tid_x = thread_id::<DimX>() as i32;
    let idx = bdim_x * bid_x + tid_x;
    let N = B * T * C;
    let dwte = gpu::sync::Atomic::new(dwte);
    let dwpe = gpu::sync::Atomic::new(dwpe);

    if idx < N {
        let bt = idx / C;
        let b = bt / T;
        let t = bt % T;
        let c = idx % C;

        let ix = inp[(b * T + t) as usize];

        let dout_btc: f32 = dout[(b * T * C + t * C + c) as usize];
        dwte.index((ix * C + c) as usize).atomic_addf(dout_btc);
        dwpe.index((t * C + c) as usize).atomic_addf(dout_btc);
    }
}

#[cfg(not(feature = "use_iterator"))]
#[gpu::cuda_kernel]
pub fn layernorm_forward_kernel3(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    N: u32,
    C: u32,
) {
    let warp = ThreadWarpTile::<32>;

    // chunk to warp and then to a single thread.
    let warp_group_size = warp.meta_group_size();
    let map = reshape_map!(
        [1] | [(32, 1), warp_group_size * grid_dim::<DimX>()] => layout: [i0, t0, t1]
    );
    let mut chunked_rstd = chunk_mut(rstd, map);

    // chunk to warp and then to a single thread.
    let mut chunked_mean = chunk_mut(mean, map);

    // chunk to warp and then to a all thread in the warp
    // where each thread gets a strided chunk.
    // tid0 + tid1 * 32 * C/32 + i * 32
    let out_map = reshape_map!(
        [C/32] | [32, warp.meta_group_size() * grid_dim::<DimX>()] => layout: [t0, i0, t1]
    );
    let mut chunked_out = chunk_mut(out, out_map);

    let bid_x = block_id::<DimX>();
    let idx = bid_x * warp_group_size + warp.subgroup_id();
    if idx >= N {
        return;
    }

    let lane_id = warp.thread_rank();

    // the row of input that this group of threads is responsible for
    let idx_C = idx * C;
    let x = &inp[idx_C as usize..(idx_C + C) as usize];

    // mean
    let mut local_sum = 0.0f32;
    let mut i = lane_id;
    while i < C {
        local_sum += x[i as usize];
        i += warp.size();
    }
    let sum: f32 = warp.redux(ReduxAdd, local_sum);
    let m = sum / C as f32;
    if lane_id == 0 {
        chunked_mean[0].stcs(m);
    }
    // rstd
    let mut local_sum = 0.0f32;

    let mut i = lane_id;
    while i < C {
        let diff = x[i as usize] - m;
        local_sum += diff * diff;
        i += warp.size();
    }
    let sum = warp.redux(ReduxAdd, local_sum);
    let s = (sum / C as f32 + 1e-5f32).rsqrt();
    if lane_id == 0 {
        chunked_rstd[0].stcs(s);
    }

    // final normalization and scaling by weight/bias
    let mut i: u32 = lane_id;
    let mut idx: u32 = 0;
    while i < C {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        let c = i as usize;
        let n = s * (x[c].ldcs() - m);
        let o = n * weight[c] + bias[c];
        chunked_out[idx].stcs(o);
        i += warp.size();
        idx += 1;
    }
}

/// Around 20% penalty on layernorm_forward_kernel3 due to iterator overhead
#[cfg(feature = "use_iterator")]
#[gpu::cuda_kernel]
pub fn layernorm_forward_kernel3(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    N: u32,
    C: u32,
) {
    let warp = ThreadWarpTile::<32>;

    // chunk to warp and then to a single thread.
    let warp_group_size = warp.meta_group_size();
    let map = reshape_map!(
        [1] | [(32, 1), warp_group_size * grid_dim::<DimX>()] => layout: [i0, t0, t1]
    );
    let mut chunked_rstd = chunk_mut(rstd, map);

    // chunk to warp and then to a single thread.
    let mut chunked_mean = chunk_mut(mean, map);

    // chunk to warp and then to a all thread in the warp
    // where each thread gets a strided chunk.
    // tid0 + tid1 * 32 * C/32 + i * 32
    let out_map = reshape_map!(
        [C/32] | [32, warp.meta_group_size() * grid_dim::<DimX>()] => layout: [t0, i0, t1]
    );
    let mut chunked_out = chunk_mut(out, out_map);

    let bid_x = block_id::<DimX>();
    let idx = bid_x * warp_group_size + warp.subgroup_id();
    if idx >= N {
        return;
    }

    let lane_id = warp.thread_rank();

    // the row of input that this group of threads is responsible for
    let idx_C = idx * C;
    let x = &inp[idx_C as usize..(idx_C + C) as usize];

    // mean
    let mut local_sum = 0.0f32;
    for i in (lane_id..C).step_by(warp.size() as usize) {
        local_sum += x[i as usize];
    }
    let sum: f32 = warp.redux(ReduxAdd, local_sum);
    let m = sum / C as f32;
    if lane_id == 0 {
        chunked_mean[0].stcs(m);
    }

    // rstd
    let mut local_sum = 0.0f32;
    for i in (lane_id..C).step_by(warp.size() as usize) {
        let diff = x[i as usize] - m;
        local_sum += diff * diff;
    }
    let sum = warp.redux(ReduxAdd, local_sum);
    let s = (sum / C as f32 + 1e-5f32).rsqrt();
    if lane_id == 0 {
        chunked_rstd[0].stcs(s);
    }
    // final normalization and scaling by weight/bias
    for (i, c) in (lane_id..C).step_by(warp.size() as usize).enumerate() {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        let n = s * (x[c as usize].ldcs() - m);
        let o = n * weight[c as usize] + bias[c as usize];
        chunked_out[i as _].stcs(o);
    }
}

#[allow(clippy::erasing_op)]
#[gpu::cuda_kernel]
pub fn permute_kernel(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    inp: &[f32],
    B: u32,
    N: u32,
    NH: u32,
    d: u32,
) {
    let mut q = chunk_mut(q, MapContinuousLinear::new(1));
    let mut k = chunk_mut(k, MapContinuousLinear::new(1));
    let mut v = chunk_mut(v, MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();

    if idx < B * NH * N * d {
        let b = idx / (NH * N * d);
        let mut rest = idx % (NH * N * d);
        let nh_ = rest / (N * d);
        rest %= N * d;
        let n = rest / d;
        let d_ = rest % d;
        let inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;

        // q, k, v are local
        q[0] = inp[(inp_idx) as usize].ldcs();
        k[0] = inp[(inp_idx + NH * d) as usize].ldcs();
        v[0] = inp[(inp_idx + 2 * (NH * d)) as usize].ldcs();
    }
}

/*
__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}
*/

#[gpu::cuda_kernel]
pub fn permute_kernel_backward(
    dinp: &mut [f32],
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    B: u32,
    N: u32,
    NH: u32,
    D: u32,
) {
    // shape (B, NH, N, D) to (B, N, 3, NH, D)
    // Q[b][nh_][n][d_][0] = inp[b][n][0][nh_][d_]
    // K[b][nh_][n][d_][0] = inp[b][n][1][nh_][d_]
    // V[b][nh_][n][d_][0] = inp[b][n][2][nh_][d_]
    // swap id2 <-> id3
    // insert id0 between id2 and id3
    let map = gpu::reshape_map!([3] | [D, N, NH, B]  => layout: [t0, t2, i0, t1, t3]);
    let mut dinp = chunk_mut(dinp, map);
    let idx = gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>();
    if idx < (B * NH * N * D) {
        let idx = idx as usize;
        dinp[0] = dq[idx];
        dinp[1] = dk[idx];
        dinp[2] = dv[idx];
    }
}

/*
__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = __ldcs(&inp[idx]);
    }
}
*/

#[gpu::cuda_kernel]
pub fn unpermute_kernel(inp: &[f32], out: &mut [f32], B: u32, N: u32, NH: u32, D: u32) {
    // shape (B, NH, N, D) to (B, N, NH, D)
    // inp shape: (B, NH, N, D) out shape (B, N, NH, D)
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    // swap tid1 and tid2
    //let map = gpu::reshape_map!([B as usize, NH as usize, N as usize, D as usize] | [1] => layout: [0, 2, 1, 3, 4]);
    let map = gpu::reshape_map!([1] | [D, N, NH, B]  => layout: [i0, t0, t2, t1, t3]);

    let mut out = chunk_mut(out, map);
    let idx = gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>();
    if idx < B * NH * N * D {
        out[0] = inp[idx as usize].ldcs();
    }
}

/*
__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}
*/

#[gpu::cuda_kernel]
pub fn unpermute_kernel_backward(dinp: &mut [f32], dout: &[f32], B: u32, N: u32, NH: u32, D: u32) {
    // shape (B, N, NH, D) to (B, NH, N, D)
    // inp shape: (B, NH, N, D) out shape (B, N, NH, D)
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    // swap tid1 and tid2
    let map = gpu::MapContinuousLinear::new(1);
    let mut dinp = chunk_mut(dinp, map);
    let idx = gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>();
    if idx < (NH * N * B * D) {
        let b = idx / (NH * N * D);
        let mut rest = idx % (NH * N * D);
        let nh_ = rest / (N * D);
        rest %= N * D;
        let n = rest / D;
        let d_ = rest % D;
        let other_idx = (b * (NH * N * D)) + (n * (NH * D)) + (nh_ * D) + d_;
        dinp[0] = dout[other_idx as usize].ldcs();
    }
}

/*
__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const Float4* x_vec = reinterpret_cast<const Float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        Float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}
*/

#[gpu::cuda_kernel]
#[allow(clippy::needless_range_loop)]
pub fn softmax_forward_kernel5(
    out: &mut [f32],
    inv_temperature: f32,
    inp: &[Float4],
    N: u32,
    T: u32,
) {
    assert!(T % 4 == 0);
    let warp = ThreadWarpTile::<32>;
    let grid2warp = build_chunk_scope(Grid, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    // idx * T + i0 * warp_size + thread_rank.
    // ((gdim - 1 - bid) * meta_group_size + subgroup_id) * T + i0 * warp_size + thread_rank
    let mut out = out
        .chunk_to_scope(
            grid2warp,
            reshape_map!(
                [T] | [warp.meta_group_size(), grid_dim::<DimX>()] => layout: [i0, t0, -t1]
            ),
        )
        .chunk_to_scope(
            warp2thread,
            reshape_map!(
                [T/32] | [32] => layout: [t0, i0]
            ),
        );
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    let global_warp_id_reversed =
        (grid_dim::<DimX>() - block_id::<DimX>() - 1) * warp.meta_group_size() + warp.subgroup_id(); // backward order
    if global_warp_id_reversed >= N * T {
        return;
    }
    let own_pos = global_warp_id_reversed % T;
    let pos_by_4 = own_pos / 4;
    let x_vec = &inp[(global_warp_id_reversed * T / 4) as usize..];
    let x: &[f32] = x_vec.flatten();
    let mut maxval = f32::MIN;
    let mut sumval = 0.0f32;
    for i in (warp.thread_rank()..pos_by_4).step_by(warp.size() as usize) {
        let v4: Float4 = x_vec[i as usize];
        let old_maxval = maxval;
        for i in 0..4 {
            maxval = maxval.max(v4[i]);
        }
        sumval *= (inv_temperature * (old_maxval - maxval)).exp();
        for i in 0..4 {
            sumval += (inv_temperature * (v4[i] - maxval)).exp();
        }
    }
    let pos = 4 * pos_by_4 + warp.thread_rank();
    if pos <= own_pos {
        let old_maxval = maxval;
        maxval = maxval.max(x[pos as usize]);
        sumval *= (inv_temperature * (old_maxval - maxval)).exp();
        sumval += (inv_temperature * (x[pos as usize] - maxval)).exp();
    }

    let global_maxval = warp.redux(ReduxMax, maxval);
    sumval *= (inv_temperature * (maxval - global_maxval)).exp();
    let sum = warp.redux(ReduxAdd, sumval);
    let norm = 1.0f32 / sum;
    // divide the whole row by the sum
    // If global_warp_id_reversed = 0, only the first thread writes to out[idx]
    // If global_warp_id_reversed = 1, only the first two threads write to out[idx]
    // ...
    // If global_warp_id_reversed = T-1, if T > 32, all threads write to out[idx]

    #[cfg(feature = "use_iterator")]
    for (idx, i) in (warp.thread_rank()..=own_pos)
        .step_by(warp.size() as usize)
        .enumerate()
    {
        // recalculation is faster than doing the round-trip through memory.
        let ev = (inv_temperature * (x[i as usize].ldcs() - global_maxval)).exp();
        //__stcs(out + idx * T + i, ev * norm);
        out[idx as _].stcs(ev * norm);
    }
    #[cfg(not(feature = "use_iterator"))]{
    let mut idx_out = 0;
    let mut i = warp.thread_rank();
    while i <= own_pos {
        // recalculation is faster than doing the round-trip through memory.
        let ev = (inv_temperature * (x[i as usize].ldcs() - global_maxval)).exp();
        //__stcs(out + idx * T + i, ev * norm);
        out[idx_out].stcs(ev * norm);
        i += warp.size();
        idx_out += 1;
    }
}
}
/*
__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}
*/

#[gpu::cuda_kernel]
pub fn residual_forward_kernel(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: u32) {
    let mut out = chunk_mut(out, gpu::MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    if idx < N {
        out[0] = inp1[idx as usize].ldcs() + inp2[idx as usize].ldcs();
    }
}

/*
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
*/

#[gpu::cuda_kernel]
pub fn gelu_forward_kernel(out: &mut [f32], inp: &[f32], N: u32) {
    let GELU_SCALING_FACTOR: f32 = (2.0f32 / core::f32::consts::PI).sqrt();

    let mut out = chunk_mut(out, gpu::MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    if idx < N {
        let xi = inp[idx as usize].ldcs();
        let cube = 0.044715 * xi * xi * xi;
        out[0] = 0.5 * xi * (1.0 + (GELU_SCALING_FACTOR * (xi + cube)).tanh());
    }
}

/*
__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}
*/

#[gpu::cuda_kernel]
pub fn gelu_backward_kernel_llm(dinp: &mut [f32], inp: &[f32], dout: &[f32], N: u32) {
    let gelu_scaling_factor: f32 = (2.0f32 / core::f32::consts::PI).sqrt();
    let mut dinp = chunk_mut(dinp, MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    if idx < N {
        let idx = idx as usize;
        let x = inp[idx].ldcs();
        let cube = 0.044715 * x * x * x;
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[0].stcs(local_grad * dout[idx].ldcs());
    }
}

// When dinp == dout
#[gpu::cuda_kernel]
pub fn gelu_backward_kernel(dinp: &mut [f32], inp: &[f32], N: u32) {
    let gelu_scaling_factor: f32 = (2.0f32 / core::f32::consts::PI).sqrt();

    let mut dinp_local = chunk_mut(dinp, MapContinuousLinear::new(1));
    let idx = dinp_local.local2global(0);
    if idx < N {
        let x = inp[idx as usize].ldcs();
        let cube = 0.044715 * x * x * x;
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);
        let dout = dinp_local[0].ldcs();
        dinp_local[0].stcs(local_grad * dout);
    }
}

/*
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}
*/
#[gpu::device]
#[inline(always)]
fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    weight.fma(end, (-weight).fma(start, start))
}

/*
// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
__global__ void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    extern __shared__ float smem[]; // of size block_size (128)
    const int warp_id = threadIdx.x / warpSize; // warp index in the block, 0,1,2,3
    const int lane_id = threadIdx.x % warpSize; // thread index in the warp, 0,1,2,...,31
    const int tl = blockIdx.x * warpSize; // pointer to the start column for this block
    const int vstep = blockDim.x / warpSize; // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}
*/

// TODO: Add tests
// This constructs matmul_backward with cublasSgemm
#[gpu::cuda_kernel(dynamic_shared)]
pub fn matmul_backward_bias_kernel4(dbias: &mut [f32], dout: &[f32], B: u32, T: u32, OC: u32) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    let block_size = block_dim::<DimX>();
    let warp = ThreadWarpTile::<32>;
    // block_id * warp_size + lane_id
    let mut dbias_chunk = chunk_mut(
        dbias,
        reshape_map!(
            [1] | [warp.size(), (warp.meta_group_size(), 1), grid_dim::<DimX>()] => layout: [i0, t0, t1, t2]
        ),
    );
    let smem = smem_alloc.alloc::<f32>(block_size as usize);
    let mut smem_chunk = smem.chunk_mut(MapContinuousLinear::new(1));

    let warp_id = thread_id::<DimX>() / warp.size(); // warp index in the block, 0,1,2,3
    let lane_id = thread_id::<DimX>() % warp.size(); // thread index in the warp, 0,1,2,...,31
    let tl = block_id::<DimX>() * warp.size(); // pointer to the start column for this block
    let vstep = block_size / warp.size(); // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    let dout_col = &dout[(tl + lane_id) as usize..]; // B*T*OC - bid * 32.

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    let mut dout_sum = 0.0f32;
    for row in (warp_id..(B * T)).step_by(vstep as usize) {
        dout_sum += dout_col[(row * OC) as usize].ldcs();
    }
    //lane_id + warp_id * warpSize
    smem_chunk[0] = dout_sum;
    sync_threads();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f32;
    if warp_id == 0 {
        for j in 0..vstep {
            dout_sum += *smem[(lane_id + (j * warp.size())) as usize];
        }
        dbias_chunk[0] += dout_sum;
    }
}

/*
__global__ void layernorm_backward_kernel2(float* dinp, float* dweight, float* dbias,
                                           const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                           int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
    }
}
*/

#[cfg(not(feature = "use_iterator"))]
#[gpu::cuda_kernel(dynamic_shared)]
pub fn layernorm_backward_kernel2(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: u32,
    T: u32,
    C: u32,
) {
    // dinp, dout, inp shape: (B, T, C)
    // weight, dbias shape: (C,)
    // mean, rstd shape: (B, T)
    let warp = ThreadWarpTile::<32>;
    let grid2warp = build_chunk_scope(Grid, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let bid_x = block_id::<DimX>();
    let bdim_x = block_dim::<DimX>();
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();

    let global_warp_id = bid_x * (warp.meta_group_size()) + (warp.subgroup_id());

    let N = B * T;
    if global_warp_id >= N {
        return;
    } // thread guards

    let warp_offset = (global_warp_id / T) * T + global_warp_id % T;
    // gdim = 32 * B * T;
    // dinp_warp_chunk = dinp + warp_id/T * T * C + (warp_id % T) * C
    // dinp_thread_chunk = dinp_warp_chunk + lane_id + i * warp_size
    /*let map = gpu::reshape_map!( [C/32] | [32, B, T] => layout: [t0, i0, t1, t2]);
    let mut dinp_chunk = chunk_mut(dinp, map);
    */
    let mut dinp_chunk = dinp
        .chunk_to_scope(
            grid2warp,
            gpu::reshape_map!( [C] | [B, T] => layout: [i0, t0, t1]),
        )
        .chunk_to_scope(
            warp2thread,
            gpu::reshape_map!( [C/32] | [32] => layout: [t0, i0]),
        );

    let dout_bt = &dout[(warp_offset * C) as usize..];
    let inp_bt = &inp[(warp_offset * C) as usize..];
    let mean_bt = mean[warp_offset as usize];
    let rstd_bt = rstd[warp_offset as usize];
    // the first half of shared memory is bias, second is weight
    let dbias_shared = smem_alloc.alloc::<f32>(C as usize);
    let mut dbias_shared_chunk = dbias_shared.chunk_mut(gpu::MapLinear::new(1));
    let dweight_shared = smem_alloc.alloc::<f32>(C as usize);
    let mut dweight_shared_chunk = dweight_shared.chunk_mut(gpu::MapLinear::new(1));
    // init shared memory to zero
    // replace iterator with while loop
    let mut id = tid;
    let mut i = 0;
    while id < C {
        dbias_shared_chunk[i as _] = 0.0;
        dweight_shared_chunk[i as _] = 0.0;
        id += bdim_x;
        i += 1;
    }
    sync_threads();

    // first: two reduce operations
    let mut dnorm_mean_local = 0.0f32;
    let mut dnorm_norm_mean_local = 0.0f32;

    let mut i = lane_id;
    while i < C {
        let norm_bti = (inp_bt[i as usize] - mean_bt) * rstd_bt;
        let dnorm_i = weight[i as usize] * dout_bt[i as usize];
        dnorm_mean_local += dnorm_i;
        dnorm_norm_mean_local += dnorm_i * norm_bti;
        i += warp.size();
    }

    let dnorm_mean = warp.redux(ReduxAdd, dnorm_mean_local);
    let dnorm_norm_mean = warp.redux(ReduxAdd, dnorm_norm_mean_local);
    let dnorm_mean = dnorm_mean / C as f32;
    let dnorm_norm_mean = dnorm_norm_mean / C as f32;

    // now iterate again and accumulate all the gradients
    let dbias_shared_atom = SharedAtomic::new(dbias_shared);
    let dweight_shared_atom = SharedAtomic::new(dweight_shared);

    let mut i = lane_id;
    let mut k = 0;
    while i < C {
        let norm_bti = (inp_bt[i as usize] - mean_bt) * rstd_bt;
        let dnorm_i = weight[i as usize] * dout_bt[i as usize];
        // gradient contribution to bias
        dbias_shared_atom
            .index(i as usize)
            .atomic_addf(dout_bt[i as usize]);
        // gradient contribution to weight
        let w = norm_bti * dout_bt[i as usize];
        dweight_shared_atom.index(i as usize).atomic_addf(w);
        // gradient contribution to input
        let mut dval = 0.0f32;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_chunk[k as _] += dval;
        i += warp.size();
        k += 1;
    }
    sync_threads(); // required before reading in shared memory

    // write to global memory
    let dbias = Atomic::new(dbias);
    let dweight = Atomic::new(dweight);

    let mut i = tid;
    while i < C {
        dbias
            .index(i as usize)
            .atomic_addf(*dbias_shared[i as usize]);
        dweight
            .index(i as usize)
            .atomic_addf(*dweight_shared[i as usize]);
        i += bdim_x;
    }
}

#[cfg(feature = "use_iterator")]
#[gpu::cuda_kernel(dynamic_shared)]
pub fn layernorm_backward_kernel2(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: u32,
    T: u32,
    C: u32,
) {
    // dinp, dout, inp shape: (B, T, C)
    // weight, dbias shape: (C,)
    // mean, rstd shape: (B, T)
    let warp = ThreadWarpTile::<32>;
    let grid2warp = build_chunk_scope(Grid, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let bid_x = block_id::<DimX>();
    let bdim_x = block_dim::<DimX>();
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();

    let global_warp_id = bid_x * (warp.meta_group_size()) + (warp.subgroup_id());

    let N = B * T;
    if global_warp_id >= N {
        return;
    } // thread guards

    let warp_offset = (global_warp_id / T) * T + global_warp_id % T;
    // gdim = 32 * B * T;
    // dinp_warp_chunk = dinp + warp_id/T * T * C + (warp_id % T) * C
    // dinp_thread_chunk = dinp_warp_chunk + lane_id + i * warp_size
    /*let map = gpu::reshape_map!( [C/32] | [32, B, T] => layout: [t0, i0, t1, t2]);
    let mut dinp_chunk = chunk_mut(dinp, map);
    */
    let mut dinp_chunk = dinp
        .chunk_to_scope(
            grid2warp,
            gpu::reshape_map!( [C] | [B, T] => layout: [i0, t0, t1]),
        )
        .chunk_to_scope(
            warp2thread,
            gpu::reshape_map!( [C/32] | [32] => layout: [t0, i0]),
        );

    let dout_bt = &dout[(warp_offset * C) as usize..];
    let inp_bt = &inp[(warp_offset * C) as usize..];
    let mean_bt = mean[warp_offset as usize];
    let rstd_bt = rstd[warp_offset as usize];
    // the first half of shared memory is bias, second is weight
    let dbias_shared = smem_alloc.alloc::<f32>(C as usize);
    let mut dbias_shared_chunk = dbias_shared.chunk_mut(gpu::MapLinear::new(1));
    let dweight_shared = smem_alloc.alloc::<f32>(C as usize);
    let mut dweight_shared_chunk = dweight_shared.chunk_mut(gpu::MapLinear::new(1));
    // init shared memory to zero
    for (i, _) in (tid..C).step_by(block_dim::<DimX>() as usize).enumerate() {
        dbias_shared_chunk[i as _] = 0.0;
        dweight_shared_chunk[i as _] = 0.0;
    }
    sync_threads();

    // first: two reduce operations
    let mut dnorm_mean_local = 0.0f32;
    let mut dnorm_norm_mean_local = 0.0f32;
    for i in (lane_id..C).step_by(warp.size() as usize) {
        let norm_bti = (inp_bt[i as usize] - mean_bt) * rstd_bt;
        let dnorm_i = weight[i as usize] * dout_bt[i as usize];
        dnorm_mean_local += dnorm_i;
        dnorm_norm_mean_local += dnorm_i * norm_bti;
    }

    let dnorm_mean = warp.redux(ReduxAdd, dnorm_mean_local);
    let dnorm_norm_mean = warp.redux(ReduxAdd, dnorm_norm_mean_local);
    let dnorm_mean = dnorm_mean / C as f32;
    let dnorm_norm_mean = dnorm_norm_mean / C as f32;

    // now iterate again and accumulate all the gradients
    let dbias_shared_atom = SharedAtomic::new(dbias_shared);
    let dweight_shared_atom = SharedAtomic::new(dweight_shared);
    for (k, i) in (lane_id..C).step_by(warp.size() as usize).enumerate() {
        let norm_bti = (inp_bt[i as usize] - mean_bt) * rstd_bt;
        let dnorm_i = weight[i as usize] * dout_bt[i as usize];
        // gradient contribution to bias
        dbias_shared_atom
            .index(i as usize)
            .atomic_addf(dout_bt[i as usize]);
        // gradient contribution to weight
        let w = norm_bti * dout_bt[i as usize];
        dweight_shared_atom.index(i as usize).atomic_addf(w);
        // gradient contribution to input
        let mut dval = 0.0f32;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_chunk[k as _] += dval;
    }
    sync_threads(); // required before reading in shared memory

    // write to global memory
    let dbias = Atomic::new(dbias);
    let dweight = Atomic::new(dweight);
    for i in (tid..C).step_by(bdim_x as usize) {
        dbias
            .index(i as usize)
            .atomic_addf(*dbias_shared[i as usize]);
        dweight
            .index(i as usize)
            .atomic_addf(*dweight_shared[i as usize]);
    }
}

/*
__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            // i0: 0..T_per_block
            // (block - blockid - 1) * T_per_block * T + (T_per_block - 1 - i1) * T  + t3 * Block=
            // block * T_per_block * T - T
            // = T * T - T
            // (T - 1 - T_per_block*blockIdx.x - i1) * T + i0 * BlockSize
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}
*/

#[gpu::cuda_kernel]
pub fn softmax_autoregressive_backward_kernel(
    dpreatt: &mut [f32],
    datt: &[f32],
    att: &[f32],
    _B: u32,
    T: u32,
    _C: u32,
    scale: f32,
) {
    // dpreatt, datt, att shape: (B * NH, T, T)
    assert!(Config::BDIM_Z == 1);
    assert!(Config::BDIM_X * Config::BDIM_Y == 256);
    const BLOCK_SIZE: u32 = 256;
    const T_PER_BLOCK: u32 = 4;
    // T / 4, B * NH
    let block = Block;
    let warp = ThreadWarpTile::<32>;
    // Requires:
    // - GDIM_X = T/4
    // - GDIM_Y: unrestricted
    // - BLOCK_SIZE = 256
    // - T_PER_BLOCK = 4
    // - T % 4 == 0 (BLOCK_SIZE/T)
    // - T can be smaller or larger than BLOCK_SIZE.
    // arr[][T/4][min(BLOCK_SIZE, T)][4][T/BLOCK_SIZE]
    let arr_i0_size = T.div_ceil(BLOCK_SIZE);
    let arr_t0_size = if BLOCK_SIZE < T { BLOCK_SIZE } else { T };
    let arr_t1_size = T.div_ceil(T_PER_BLOCK);
    let dpreatt_map = gpu::reshape_map!(
        [arr_i0_size, T_PER_BLOCK] |
        [
            (BLOCK_SIZE, arr_t0_size),
            (grid_dim::<DimX>(), arr_t1_size), //reversed
            grid_dim::<DimY>(),
        ] =>
        layout: [t0, i0, -i1, -t1, t2]
    );
    let mut dpreatt_bth = chunk_mut(dpreatt, dpreatt_map);

    let mut block_acc = gpu::GpuShared::<[f32; 32]>::zero();
    let block2warp = build_chunk_scope(block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let idx = block_id::<DimY>();

    // go through blocks in reverse order, so the slowest block starts first
    let t0 = (T - 1) as i32 - (T_PER_BLOCK * block_id::<DimX>()) as i32;

    // Assign 32 elements to each warp
    // and then assign 1 element to each thread in the warp
    // Since block_acc.len() = 32, only the first warp writes to block_acc
    let mut block_acc_init_chunk = block_acc
        .chunk_to_scope(block2warp, gpu::MapContinuousLinear::new(32))
        .chunk_to_scope(warp2thread, gpu::MapContinuousLinear::new(1));
    if warp.subgroup_id() == 0 {
        block_acc_init_chunk[0] = 0.0;
    }

    let lane_id = warp.thread_rank();
    for to in 0..T_PER_BLOCK {
        let t = t0 - to as i32;
        if t < 0 {
            return;
        }
        let t = t as u32;
        let bth = idx * T * T + t * T;
        let att_bth = &att[bth as usize..];
        let datt_bth = &datt[bth as usize..];
        let mut local_sum = 0.0f32;
        #[cfg(feature = "use_iterator")]
        {
            for t2 in (block.thread_rank()..=t).step_by(BLOCK_SIZE as usize) {
                local_sum += att_bth[t2 as usize] * datt_bth[t2 as usize];
            }
        }

        #[cfg(not(feature = "use_iterator"))]
        {
            let mut t2 = block.thread_rank();
            while t2 <= t {
                local_sum += att_bth[t2 as usize] * datt_bth[t2 as usize];
                t2 += BLOCK_SIZE;
            }
        }
        // warp-level reduction
        let acc = warp.redux(ReduxAdd, local_sum);

        // This is required to make sure block_acc is ready before the next step
        // and required to make sure the write happens after the read in previous step
        // Fix bug in LLM.c
        sync_threads();

        // block-level accumulation
        // Thus only lane_0 can write to block_acc_chunk
        let mut block_acc_chunk = block_acc
            .chunk_to_scope(block2warp, gpu::MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, gpu::MapContinuousLinear::new(1));
        if lane_id == 0 {
            block_acc_chunk[0] = acc;
        }
        sync_threads();

        // block-level reduction
        local_sum = warp.redux(ReduxAdd, block_acc[lane_id as usize]);

        #[cfg(feature = "use_iterator")]
        {
            for t2 in (block.thread_rank()..=t).step_by(BLOCK_SIZE as usize) {
                local_sum += att_bth[t2 as usize] * datt_bth[t2 as usize];
            }
        }

        #[cfg(feature = "use_iterator")]
        {
            for (i, t3) in (0_u32..).zip((block.thread_rank()..=t).step_by(BLOCK_SIZE as usize)) {
                let acc = att_bth[t3 as usize].ldcs() * (datt_bth[t3 as usize].ldcs() - local_sum);
                let val = scale * acc;
                dpreatt_bth[(i, to)].stcs(val);
            }
        }

        #[cfg(not(feature = "use_iterator"))]
        {
            let mut t3 = block.thread_rank();
            let mut i = 0;
            while t3 <= t {
                let acc = att_bth[t3 as usize].ldcs() * (datt_bth[t3 as usize].ldcs() - local_sum);
                let val = scale * acc;
                dpreatt_bth[(i, to)].stcs(val);
                t3 += BLOCK_SIZE;
                i += 1;
            }
        }
    }
}

/*
__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}
*/
/// TODO: add tests
#[gpu::cuda_kernel]
pub fn adamw_kernel2(
    params_memory: &mut [f32],
    grads_memory: &[f32],
    m_memory: &mut [f32],
    v_memory: &mut [f32],
    num_parameters: i32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    eps: f32,
    weight_decay: f32,
) {
    let mut params_memory = chunk_mut(params_memory, gpu::MapContinuousLinear::new(1));
    let mut m_memory = chunk_mut(m_memory, gpu::MapContinuousLinear::new(1));
    let mut v_memory = chunk_mut(v_memory, gpu::MapContinuousLinear::new(1));
    let idx = (block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>()) as i32;
    if idx >= num_parameters {
        return;
    }
    let grad = grads_memory[idx as usize];
    let m = m_memory[0];
    let v = v_memory[0];
    // update the first moment (momentum)
    let m = lerp(grad, m, beta1);
    m_memory[0] = m;
    // update the second moment (RMSprop)
    let v = lerp(grad * grad, v, beta2);
    v_memory[0] = v;
    let m_hat = m / beta1_correction; // m_hat
    let v_hat = v / beta2_correction; // v_hat
    let old_param = params_memory[0];
    params_memory[0] =
        old_param - learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * old_param);
}

/*
__device__ SoftmaxParams prepare_softmax_blockwide_noFloat4(cg::thread_block_tile<32>& warp,
                                                   int idx, const float* inp, int V, int P) {
    // same but not Float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf((old_maxval - thread_maxval));
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}
*/

struct SoftmaxParams {
    scale: f32,
    offset: f32,
}

// inp: Shape(V,)
#[inline(always)]
#[gpu::device]
fn prepare_softmax_blockwide_noFloat4<M: ScopeUniqueMap<Grid2BlockScope, IndexType = u32>>(
    inp: &GlobalGroupChunk<'_, f32, Grid2BlockScope, M>,
    V: u32,
) -> SoftmaxParams {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    let mut shared_maxval = gpu::GpuShared::<[f32; 32]>::zero();
    let mut shared_sumval = gpu::GpuShared::<[f32; 32]>::zero();

    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let mut shared_maxval_chunk = shared_maxval
        .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
        .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));

    let mut shared_sumval_chunk = shared_sumval
        .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
        .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));

    // bidx * P + i * bdimx
    let mut thread_maxval = f32::NEG_INFINITY;
    let mut thread_sumval = 0.0f32;

    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the
    let tid = thread_id::<gpu::DimX>();
    let block_size = block_dim::<DimX>();
    for i in (0..=V + tid - block_size).rev().step_by(block_size as _) {
        let v = inp[i as _];
        let old_maxval = thread_maxval;
        thread_maxval = thread_maxval.max(v);
        thread_sumval *= (old_maxval - thread_maxval).exp();
        thread_sumval += (v - thread_maxval).exp();
    }
    let num_warps = warp.meta_group_size();
    let lane_id = warp.thread_rank();
    // reduce maxval within each warp
    let warp_maxval = warp.redux(ReduxMax, thread_maxval);
    // thread 0 in each warp writes to shared memory
    if lane_id == 0 {
        shared_maxval_chunk[0] = warp_maxval;
    }
    sync_threads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as
    let warp_maxval = if lane_id < num_warps {
        // ensure shared_maxval[lane_id] is initialized
        shared_maxval[lane_id as usize]
    } else {
        f32::MIN
    };
    // now reduce the maxval among the warp threads
    let block_maxval = warp.redux(ReduxMax, warp_maxval);
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= (thread_maxval - block_maxval).exp();
    // (warp-level) reduce sumval, thread 0 in each warp saves result in
    let warp_sumval = warp.redux(ReduxAdd, thread_sumval);
    if lane_id == 0 {
        shared_sumval_chunk[0] = warp_sumval;
    }

    sync_threads();
    // same strategy, now reduce sumval across warps
    let warp_sumval = if lane_id < num_warps {
        // ensure shared_sumval[lane_id] is initialized
        shared_sumval[lane_id as usize]
    } else {
        0.0
    };
    let block_sumval = warp.redux(ReduxAdd, warp_sumval);
    // return the softmax parameters
    SoftmaxParams {
        scale: 1.0 / block_sumval,
        offset: block_maxval,
    }
}

/*// same as 2 but not using Float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
__global__ void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_noFloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}
*/

#[gpu::cuda_kernel]
pub fn fused_classifier_kernel3(
    logits: &mut [f32],
    losses: &mut [f32],
    probs: &mut [f32],
    dlosses: &[f32],
    targets: &[i32],
    B: u32,
    T: u32,
    V: u32, // vocab_size
    P: u32, // padded_vocab_size P >= V
) {
    let block = Block;
    let grid2block = build_chunk_scope(Grid, block);
    let block2thread = build_chunk_scope(block, Thread);

    // chunk logits to (B*T, P)
    let logits_block = logits.chunk_to_scope(grid2block, MapContinuousLinear::new(P as _)); // blockid * P + tid

    let mut losses_chunk = losses
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    let idx = gpu::block_id::<gpu::DimX>();
    if idx >= B * T {
        return;
    }
    let ix = targets[idx as usize];
    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    // assert!(P >= V);
    let sp = prepare_softmax_blockwide_noFloat4(&logits_block, V);
    // calculate the probability needed for the loss and update (single-threaded)
    if gpu::thread_id::<gpu::DimX>() == 0 {
        /*
        let prob = ((logits_block[ix as _] - sp.offset).exp()) * sp.scale;
        losses_chunk[0] = -prob.ln();
        */
        // The above is the llm.c implementation, but it is not stable enough in LLVM.
        // When enabling -ffast-math, the assembly uses exp2 and log2 which are less accurate.
        // If logits_block[ix as _] - sp.offset ~= -80 - -100, exp2f will underflow to 0,
        // and log2f(0) = -inf, resulting in -inf loss.
        // Below is equivalent, but more numerically stable
        losses_chunk[0] = -(logits_block[ix as _] - sp.offset) - sp.scale.ln()
    }
    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    let dloss = if !dlosses.is_empty() {
        dlosses[idx as usize]
    } else {
        1.0 / (B * T) as f32
    };

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    // sync safety: logits_block is only used in this thread block.
    // prepare_softmax_blockwide_noFloat4 has sync_threads internally.
    let mut logits_chunk = logits_block.chunk_to_scope(
        block2thread,
        reshape_map!( [V.div_ceil(block_dim::<DimX>())] | [block_dim::<DimX>()] => layout: [t0, i0]),
    );
    let prob_is_empty = probs.is_empty();
    let mut probs_chunk = probs
        .chunk_to_scope(grid2block, MapContinuousLinear::new(V as _))
        .chunk_to_scope(
            block2thread,
            reshape_map!( [V.div_ceil(block_dim::<DimX>())] | [block_dim::<DimX>()] => layout: [t0, i0]),
        );

    // use while loop
    #[cfg(not(feature = "use_iterator"))]
    {
        let mut k = 0;
        let mut i = thread_id::<gpu::DimX>();
        while i < V {
            // this is the 2nd read of logits after the one in prepare_softmax2
            // this data will never be needed again, so we reduce cache persistence
            let v = logits_chunk[k as _].ldcs();
            let prob = ((v - sp.offset).exp()) * sp.scale;
            if !prob_is_empty {
                probs_chunk[k as _] = prob;
            }
            let indicator = if i as i32 == ix { 1.0 } else { 0.0 };
            logits_chunk[k as _] = (prob - indicator) * dloss;
            k += 1;
            i += block_dim::<DimX>();
        }
    }
    #[cfg(feature = "use_iterator")]
    {
        for (k, i) in (gpu::thread_id::<gpu::DimX>()..V)
            .step_by(gpu::block_dim::<gpu::DimX>() as _)
            .enumerate()
        {
            // this is the 2nd read of logits after the one in prepare_softmax2
            // this data will never be needed again, so we reduce cache persistence
            let v = logits_chunk[k as _].ldcs();
            let prob = ((v - sp.offset).exp()) * sp.scale;
            if !prob_is_empty {
                probs_chunk[k as _] = prob;
            }
            let indicator = if i as i32 == ix { 1.0 } else { 0.0 };
            logits_chunk[k as _] = (prob - indicator) * dloss;
        }
    }
}

/*
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                   const float* inp, const float* weight, const float* bias,
                                                                   int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                Float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    int si_start = 4*(16 * threadIdx.y + threadIdx.x);

    for (int so = 0; so < C; so += 32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4) {
            Float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                Float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            Float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}
*/

#[inline(always)]
#[gpu::device]
fn dot4(a: Float4, b: Float4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[gpu::cuda_kernel]
#[gpu::attr(nvvm_launch_bound(256, 1, 1, 2))]
#[allow(clippy::needless_range_loop)]
pub fn matmul_forward_kernel4(
    out: &mut [Float4],
    inp: &[Float4],
    weight: &[Float4],
    bias: &[Float4],
    C: u32,
    OC: u32,
) {
    // precondition:
    // 1. gdim_y = OC / (8*bdim_y) => gdim_y <= OC, bdim_y * 8 <= OC, dim_y * 8 <= OC,
    // 2. gdim_x = B * T / (8* bdim_x)
    assert!(Config::BDIM_X <= 16);
    assert!(Config::BDIM_X >= 2);
    assert!(Config::BDIM_Y <= 16);
    assert!(Config::BDIM_Y >= 2);
    assert!(OC % 128 == 0);
    assert!(C % 32 == 0);
    assert!(OC == 8 * dim::<DimY>());
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.

    let bid_x = block_id::<DimX>();
    let bid_y = block_id::<DimY>();
    let bdim_y = block_dim::<DimY>();
    let tid_x = thread_id::<DimX>();
    let tid_y = thread_id::<DimY>();

    let oc = 8 * (bid_y * bdim_y + tid_y);

    //  128 * blockIdx.y + 128 * blockIdx.x * OC + i * OC + (8*threadIdx.y) + (8*threadIdx.x) * OC + j)
    // do not swap 2 and 3 in permutation, otherwise we will have out of bound access.
    let map = gpu::reshape_map!(
        [2, 8] | [16, grid_dim::<DimX>(), 16, (grid_dim::<DimY>(), (OC / 128))]  =>
        layout: [i0, t2, t3, i1, t0, t1]
    );

    // flatten::<4>() to handle 4 f32 at a time
    let mut out_thread = chunk_mut(out, map);

    // buffers to cache chunks of the input matrices
    let mut lhs_s = GpuShared::<[Float4; 8 * 128]>::zero();
    let mut rhs_s = GpuShared::<[Float4; 8 * 128]>::zero();

    // adjust our immutable for the current block
    let inp_offset = 128 * bid_x * C;
    let weight_offset = 128 * bid_y * C;

    let inp = &inp[(inp_offset / 4) as usize..];
    let weight = &weight[(weight_offset / 4) as usize..];

    let mut vals = [[Float4::default(); 2]; 8];
    if !bias.is_empty() {
        for i in 0..8 {
            for j in 0..2 {
                vals[i][j as usize] = bias[(oc / 4 + j) as usize];
            }
        }
    }

    let si_start = 4 * (16 * tid_y + tid_x);

    // (k * 32 * 32 + tid_y * 2 * 32 + xby8 * 32 + xmod8 * 4 + i)
    let map = reshape_map!(
        [1, 4] | [8, (block_dim::<DimX>().div_ceil(8), 2), (block_dim::<DimY>(), 16)] =>
        layout: [i0, t0, t1, t2, i1]
    );
    for so in (0..C).step_by(32) {
        gpu::sync::sync_threads();
        // Each thread handle [f32; 32].

        let mut lhs_s_chunk = lhs_s.chunk_mut(map);
        let mut rhs_s_chunk = rhs_s.chunk_mut(map);
        let xmod8 = tid_x % 8;
        let xby8 = tid_x / 8;
        let xo = 4 * xmod8;
        // 2 * threadIdx.y + xby8; y < 128; y += 32
        // for (k, y) in ((2 * thread_id::<DimY>() + xby8)..128)
        //    .step_by(32)
        //    .enumerate()
        for k in 0..4 {
            let y = (2 * tid_y + xby8) + k * 32;
            // st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            let inp_index = y * C + so + xo;
            let lhs_vec: Float4 = inp[(inp_index / 4) as usize];
            lhs_s_chunk[(0, k as _)] = lhs_vec;
            // st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
            let rhs_vec: Float4 = weight[(inp_index / 4) as usize];
            rhs_s_chunk[(0, k as _)] = rhs_vec;
        }
        gpu::sync::sync_threads();
        for i in 0..8 {
            let si = si_start + i * 4;
            let mut rhs = [Float4::default(); 8];
            for u in 0..8 {
                let rhs_idx = (u + 8 * tid_y) * 8 + (si % 32) / 4;
                rhs[u as usize] = rhs_s[rhs_idx as usize];
            }
            unroll! {for ii in 0..8 {
                let lhs_index = (ii * 8) + ((8 * tid_x) * 8 + (si % 32) / 4) as usize;
                let lhs = lhs_s[lhs_index];
                for j in 0..2 {
                    for k in 0..4 {
                        // Use v4 load
                        vals[ii][j][k] += dot4(lhs, rhs[j * 4 + k]);
                    }
                }
            }}
        }
    }
    for i in 0..8 {
        out_thread[(0, i)] = vals[i as usize][0];
        out_thread[(1, i)] = vals[i as usize][1];
    }
}
