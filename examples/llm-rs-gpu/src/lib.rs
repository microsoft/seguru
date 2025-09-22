#![no_std]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Grid, Thread};
use gpu::{block_id, chunk_mut, float4, CacheStreamLoadStore, GPUDeviceFloatIntrinsics, MapLinear};

#[gpu_macros::device]
#[inline(always)]
pub fn add_float4(a: &float4, b: &float4) -> float4 {
    float4 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
        w: a.w + b.w,
    }
}

#[gpu_macros::cuda_kernel]
pub fn encoder_forward_kernel3(
    out: &mut [float4],
    inp: &[i32],
    wte: &[float4],
    wpe: &[float4],
    B: i32,
    T: i32,
    C: i32,
) {
    let mut out = chunk_mut(out, MapLinear::new(1));
    let C4 = C / 4;
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;
    let N = B * T * C4;
    if idx < N {
        let bt = idx / C4;
        let b = bt / T;
        let t = bt % T;
        let c4 = idx % C4;
        let ix = inp[(b * T + t) as usize];

        // bt = b * T + t, bt * C4 + c4 = idx
        // b * T * C4 + t * C4 + c4 = idx
        // out is local so we are at 0
        out[0] = add_float4(&wte[(ix * C4 + c4) as usize], &wpe[(t * C4 + c4) as usize]);
    }
}

/// really bad naive kernel with atomicAdd
#[gpu_macros::cuda_kernel]
pub fn encoder_backward_kernel(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: i32,
    T: i32,
    C: i32,
) {
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;
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

#[gpu_macros::cuda_kernel]
pub fn layernorm_forward_kernel3(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    N: i32,
    C: i32,
) {
    let C = C as usize;
    let N = N as u32;
    let warp = ThreadWarpTile::<32>;
    let grid2warp = build_chunk_scope(Grid, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    // chunk to warp and then to a single thread.
    let mut chunked_rstd = rstd
        .chunk_to_scope(grid2warp, MapLinear::new(1))
        .chunk_to_scope(warp2thread, MapLinear::new(1));

    // chunk to warp and then to a single thread.
    let mut chunked_mean = mean
        .chunk_to_scope(grid2warp, MapLinear::new(1))
        .chunk_to_scope(warp2thread, MapLinear::new(1));

    // chunk to warp and then to a all thread in the warp
    // where each thread gets a strided chunk.
    let mut chunked_out = out
        .chunk_to_scope(grid2warp, MapLinear::new(C as _))
        .chunk_to_scope(warp2thread, MapLinear::new(1));

    let idx = block_id::<gpu::DimX>() * warp.meta_group_size() + warp.subgroup_id();
    let lane_id = warp.thread_rank();
    if idx >= N as usize {
        return;
    }
    // the row of input that this group of threads is responsible for
    let idx_C = idx * C as usize;
    let x = &inp[idx_C..idx_C + C as usize];

    // mean
    let mut local_sum = 0.0f32;
    for i in (lane_id..C).step_by(warp.size()) {
        local_sum += x[i as usize];
    }
    let sum: f32 = warp.redux(ReduxAdd, local_sum);
    let m = sum / C as f32;
    if lane_id == 0 {
        chunked_mean[0].stcs(m);
    }

    // rstd
    let mut local_sum = 0.0f32;
    for i in (lane_id..C).step_by(warp.size()) {
        let diff = x[i as usize] - m;
        local_sum += diff * diff;
    }
    let sum = warp.redux(ReduxAdd, local_sum);
    let s = (sum / C as f32 + 1e-5f32).rsqrt();
    if lane_id == 0 {
        chunked_rstd[0].stcs(s);
    }
    // final normalization and scaling by weight/bias
    for (i, c) in (lane_id..C).step_by(warp.size()).enumerate() {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        let c = c as usize;
        let n = s * (x[c].ldcs() - m);
        let o = n * weight[c] + bias[c];
        chunked_out[i].stcs(o);
    }
}

#[allow(clippy::erasing_op)]
#[gpu_macros::cuda_kernel]
pub fn permute_kernel(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    inp: &[f32],
    B: i32,
    N: i32,
    NH: i32,
    d: i32,
) {
    let mut q = chunk_mut(q, MapLinear::new(1));
    let mut k = chunk_mut(k, MapLinear::new(1));
    let mut v = chunk_mut(v, MapLinear::new(1));
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;

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

#[gpu_macros::cuda_kernel]
pub fn permute_kernel_backward(
    dinp: &mut [f32],
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    B: i32,
    N: i32,
    NH: i32,
    D: i32,
) {
    // shape (B, NH, N, D) to (B, N, 3, NH, D)
    // Q[b][nh_][n][d_][0] = inp[b][n][0][nh_][d_]
    // K[b][nh_][n][d_][0] = inp[b][n][1][nh_][d_]
    // V[b][nh_][n][d_][0] = inp[b][n][2][nh_][d_]
    // swap id2 <-> id3
    // insert id0 between id2 and id3
    let map = gpu::reshape_map!([3] | [D as usize, N as usize, NH as usize, B as usize]  => layout: [t0, t2, i0, t1, t3]);
    let mut dinp = chunk_mut(dinp, map);
    let idx = gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>();
    if idx < (B * NH * N * D) as usize {
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

#[gpu_macros::cuda_kernel]
pub fn unpermute_kernel(inp: &[f32], out: &mut [f32], B: i32, N: i32, NH: i32, D: i32) {
    // shape (B, NH, N, D) to (B, N, NH, D)
    // inp shape: (B, NH, N, D) out shape (B, N, NH, D)
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    // swap tid1 and tid2
    //let map = gpu::reshape_map!([B as usize, NH as usize, N as usize, D as usize] | [1] => layout: [0, 2, 1, 3, 4]);
    let map = gpu::reshape_map!([1] | [D as usize, N as usize, NH as usize, B as usize]  => layout: [i0, t0, t2, t1, t3]);

    let mut out = chunk_mut(out, map);
    let idx = gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>();
    if idx < (B * NH * N * D) as usize {
        out[0] = inp[idx].ldcs();
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

#[gpu_macros::cuda_kernel]
pub fn unpermute_kernel_backward(dinp: &mut [f32], dout: &[f32], B: i32, N: i32, NH: i32, D: i32) {
    // shape (B, N, NH, D) to (B, NH, N, D)
    // inp shape: (B, NH, N, D) out shape (B, N, NH, D)
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    // swap tid1 and tid2
    let map = gpu::MapLinear::new(1);
    let mut dinp = chunk_mut(dinp, map);
    let idx = (gpu::block_dim::<gpu::DimX>() * gpu::block_id::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as i32;
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
