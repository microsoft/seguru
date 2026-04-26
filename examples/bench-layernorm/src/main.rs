//! LayerNorm forward — two SeGuRu ports of PyTorch's kernel, compared to:
//!   - CUDA C++: PyTorch's vectorized_layer_norm_kernel (hand-tuned)
//!   - Naive SeGuRu: mechanical 1:1 translation of PyTorch's simple
//!     `LayerNormForwardCUDAKernel` (one block per row, thread-strided loop,
//!     per-block reduction via shared memory).
//!   - Idiomatic SeGuRu: warp-cooperative row reduction, `reshape_map!`,
//!     subslice, `ldcs` — written from scratch (no reuse of llm-rs kernels).
//!
//! Shape: M rows of length N (N const-generic so reshape_map can partition it).

#![allow(non_snake_case)]

use gpu::CacheStreamLoadStore;
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};
use std::time::Instant;

const N: u32 = 1024;
const BDIM_NAIVE: u32 = 256; // threads/block for naive kernel (N % BDIM_NAIVE == 0)
const WARPS_PER_BLOCK: u32 = 8;
const BDIM_IDIOM: u32 = 32 * WARPS_PER_BLOCK;

// ======================================================================
// Naive port: one block per row, block-wide reduction via shared memory.
// Mirrors PyTorch's simple `RowwiseMomentsCUDAKernel` + `LayerNormForwardCUDAKernel`.
// ======================================================================
#[gpu::cuda_kernel]
pub fn layernorm_naive(x: &[f32], gamma: &[f32], beta: &[f32], y: &mut [f32]) {
    let row = block_id::<DimX>();
    let tid = thread_id::<DimX>();

    let mut sdata = gpu::GpuShared::<[f32; BDIM_NAIVE as usize]>::zero();
    let mut sdata2 = gpu::GpuShared::<[f32; BDIM_NAIVE as usize]>::zero();

    let row_off = row * N;

    // Pass 1: partial sum for mean
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < N {
        local_sum += x[(row_off + i) as usize];
        i += BDIM_NAIVE;
    }
    {
        let mut sc = sdata.chunk_mut(reshape_map!([1] | [BDIM_NAIVE] => layout: [i0, t0]));
        sc[0] = local_sum;
    }
    sync_threads();

    let mut stride = BDIM_NAIVE / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = sdata.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            sc[0] = sc[0] + sc[1];
        }
        sync_threads();
        stride /= 2;
    }
    // Explicit sync before reading sdata[0] (the reduction wrote it under tid<stride).
    let mean = sdata[0] / (N as f32);

    // Pass 2: partial sum for variance (uses sdata2 so sdata read above isn't flagged).
    let mut local_sq = 0.0f32;
    let mut i = tid;
    while i < N {
        let d = x[(row_off + i) as usize] - mean;
        local_sq += d * d;
        i += BDIM_NAIVE;
    }
    {
        let mut sc = sdata2.chunk_mut(reshape_map!([1] | [BDIM_NAIVE] => layout: [i0, t0]));
        sc[0] = local_sq;
    }
    sync_threads();

    let mut stride = BDIM_NAIVE / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = sdata2.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            sc[0] = sc[0] + sc[1];
        }
        sync_threads();
        stride /= 2;
    }
    let rstd = (sdata2[0] / (N as f32) + 1e-5f32).rsqrt();

    // Pass 3: normalize + affine.
    // Output chunk: per-thread owns N/BDIM_NAIVE elements strided by BDIM_NAIVE;
    // blocks stride by N. layout: [t0, i0, t1] → pos = t0 + i0*BDIM + t1*N.
    let mut y_chunk = chunk_mut(
        y,
        reshape_map!(
            [N / BDIM_NAIVE] | [BDIM_NAIVE, grid_dim::<DimX>()] => layout: [t0, i0, t1]
        ),
    );
    let mut i = tid;
    let mut slot: u32 = 0;
    while i < N {
        let xv = x[(row_off + i) as usize];
        y_chunk[slot] = (xv - mean) * rstd * gamma[i as usize] + beta[i as usize];
        i += BDIM_NAIVE;
        slot += 1;
    }
}

// ======================================================================
// Idiomatic port: one warp per row. Uses SeGuRu's ThreadWarpTile + redux,
// subslice iteration, ldcs streaming loads, reshape_map for output chunks.
// ======================================================================
#[gpu::cuda_kernel]
pub fn layernorm_idiomatic(x: &[f32], gamma: &[f32], beta: &[f32], y: &mut [f32]) {
    let warp = ThreadWarpTile::<32>;
    let warps_per_block = warp.meta_group_size();

    let row = block_id::<DimX>() * warps_per_block + warp.subgroup_id();
    let lane = warp.thread_rank();

    // Subslice the row — eliminates per-element bounds checks on global reads.
    let row_off = row * N;
    let x_row = &x[row_off as usize..(row_off + N) as usize];

    // Pass 1: warp reduction for mean.
    let mut s: f32 = 0.0;
    let mut i = lane;
    while i < N {
        s += x_row[i as usize].ldcs();
        i += warp.size();
    }
    let sum: f32 = warp.redux(ReduxAdd, s);
    let mean = sum / (N as f32);

    // Pass 2: warp reduction for variance.
    let mut sq: f32 = 0.0;
    let mut i = lane;
    while i < N {
        let d = x_row[i as usize].ldcs() - mean;
        sq += d * d;
        i += warp.size();
    }
    let var: f32 = warp.redux(ReduxAdd, sq);
    let rstd = (var / (N as f32) + 1e-5f32).rsqrt();

    // Pass 3: normalize + affine. Output partitioned per-thread via reshape_map:
    // [N/32] slots per thread; thread axis t0=lane(32), t1=(warp_in_block×block_in_grid).
    // Memory layout: [t0, i0, t1] → pos = t0 + i0*32 + t1*N. Matches strided writes.
    let mut y_chunk = chunk_mut(
        y,
        reshape_map!(
            [N / 32] | [32, warps_per_block * grid_dim::<DimX>()] => layout: [t0, i0, t1]
        ),
    );
    let mut slot: u32 = 0;
    let mut i = lane;
    while i < N {
        let v = x_row[i as usize].ldcs();
        let normed = (v - mean) * rstd;
        let out = normed * gamma[i as usize].ldcs() + beta[i as usize].ldcs();
        y_chunk[slot].stcs(out);
        i += warp.size();
        slot += 1;
    }
}

// ======================================================================
// Vectorized port: warp-per-row, Float4 loads, FUSED single-pass stats
// (local sum + sumsq, then two warp.redux calls). Mirrors PyTorch's
// vectorized_layer_norm_kernel design at the single-warp level.
// ======================================================================
#[gpu::cuda_kernel]
pub fn layernorm_vectorized(
    x_vec: &[Float4],
    gamma_vec: &[Float4],
    beta_vec: &[Float4],
    y_vec: &mut [Float4],
) {
    let warp = ThreadWarpTile::<32>;
    let warps_per_block = warp.meta_group_size();

    let row = block_id::<DimX>() * warps_per_block + warp.subgroup_id();
    let lane = warp.thread_rank();

    const N4: u32 = N / 4;
    let row_off4 = row * N4;
    let x_row = &x_vec[row_off4 as usize..(row_off4 + N4) as usize];

    // Fused pass: local sum + sum-of-squares with Float4 loads.
    let mut s: f32 = 0.0;
    let mut sq: f32 = 0.0;
    let mut i = lane;
    while i < N4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 {
            let vk = v[k];
            s += vk;
            sq += vk * vk;
        }
        i += warp.size();
    }
    let sum: f32 = warp.redux(ReduxAdd, s);
    let sumsq: f32 = warp.redux(ReduxAdd, sq);
    let inv_n = 1.0f32 / (N as f32);
    let mean = sum * inv_n;
    let var = sumsq * inv_n - mean * mean;
    let rstd = (var + 1e-5f32).rsqrt();

    // Output: Float4 partitioning. Each warp writes N/4 Float4's for its row.
    // Per-thread slots: N4/32 = 8; strides: lane(32), then row (warps_per_block*gdim).
    let mut y_chunk = chunk_mut(
        y_vec,
        reshape_map!(
            [N4 / 32] | [32, warps_per_block * grid_dim::<DimX>()] => layout: [t0, i0, t1]
        ),
    );
    let mut slot: u32 = 0;
    let mut i = lane;
    while i < N4 {
        let v: Float4 = x_row[i as usize];
        let g: Float4 = gamma_vec[i as usize];
        let b: Float4 = beta_vec[i as usize];
        let mut out: Float4 = Float4::new([0.0; 4]);
        for k in 0..4 {
            out[k] = (v[k] - mean) * rstd * g[k] + b[k];
        }
        y_chunk[slot] = out;
        i += warp.size();
        slot += 1;
    }
}

// ======================================================================
// Host driver & benchmark
// ======================================================================
fn cpu_reference(x: &[f32], gamma: &[f32], beta: &[f32], y: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        let row = &x[i * n..(i + 1) * n];
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        for j in 0..n {
            y[i * n + j] = (row[j] - mean) * rstd * gamma[j] + beta[j];
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let m: usize = 8192;
    let n: usize = N as usize;
    let iters = 100;

    let mut rng_state: u32 = 0x12345678;
    let mut rnd = || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
    };

    let h_x: Vec<f32> = (0..m * n).map(|_| rnd()).collect();
    let h_gamma: Vec<f32> = (0..n).map(|_| rnd() + 1.0).collect();
    let h_beta: Vec<f32> = (0..n).map(|_| rnd() * 0.1).collect();

    let mut h_ref = vec![0.0f32; m * n];
    cpu_reference(&h_x, &h_gamma, &h_beta, &mut h_ref, m, n);

    gpu_host::cuda_ctx(0, |ctx, md| {
        let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
        let d_g = ctx.new_tensor_view(h_gamma.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(h_beta.as_slice()).unwrap();

        // ---------- Naive ----------
        {
            let mut h_y = vec![0.0f32; m * n];
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let gdim: u32 = m as u32;

            let config = gpu_host::gpu_config!(gdim, 1, 1, BDIM_NAIVE, 1, 1, 0);
            layernorm_naive::launch(config, ctx, md, &d_x, &d_g, &d_b, &mut d_y).unwrap();
            ctx.sync().unwrap();
            d_y.copy_to_host(&mut h_y).unwrap();
            let err = max_abs_diff(&h_y, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gdim, 1, 1, BDIM_NAIVE, 1, 1, 0);
                layernorm_naive::launch(cfg, ctx, md, &d_x, &d_g, &d_b, &mut d_y).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!(
                "layernorm naive      SeGuRu: {:8.2} us/iter  (M={} N={}, max_err={:.3e})",
                us, m, n, err
            );
        }

        // ---------- Idiomatic ----------
        {
            let mut h_y = vec![0.0f32; m * n];
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let gdim: u32 = (m as u32).div_ceil(WARPS_PER_BLOCK);

            let config = gpu_host::gpu_config!(gdim, 1, 1, BDIM_IDIOM, 1, 1, 0);
            layernorm_idiomatic::launch(config, ctx, md, &d_x, &d_g, &d_b, &mut d_y).unwrap();
            ctx.sync().unwrap();
            d_y.copy_to_host(&mut h_y).unwrap();
            let err = max_abs_diff(&h_y, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gdim, 1, 1, BDIM_IDIOM, 1, 1, 0);
                layernorm_idiomatic::launch(cfg, ctx, md, &d_x, &d_g, &d_b, &mut d_y).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!(
                "layernorm idiomatic  SeGuRu: {:8.2} us/iter  (M={} N={}, max_err={:.3e})",
                us, m, n, err
            );
        }

        // ---------- Vectorized (Float4, fused stats) ----------
        {
            // Build Float4 host buffers from the f32 inputs (rows are N-aligned).
            let to_f4 = |v: &[f32]| -> Vec<Float4> {
                v.chunks_exact(4)
                    .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let h_x4 = to_f4(&h_x);
            let h_g4 = to_f4(&h_gamma);
            let h_b4 = to_f4(&h_beta);
            let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); h_x4.len()];

            let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
            let d_g4 = ctx.new_tensor_view(h_g4.as_slice()).unwrap();
            let d_b4 = ctx.new_tensor_view(h_b4.as_slice()).unwrap();
            let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();
            let gdim: u32 = (m as u32).div_ceil(WARPS_PER_BLOCK);

            let cfg = gpu_host::gpu_config!(gdim, 1, 1, BDIM_IDIOM, 1, 1, 0);
            layernorm_vectorized::launch(cfg, ctx, md, &d_x4, &d_g4, &d_b4, &mut d_y4).unwrap();
            ctx.sync().unwrap();
            d_y4.copy_to_host(h_y4.as_mut_slice()).unwrap();
            // Flatten Float4 back to f32 for verification.
            let h_y_flat: &[f32] = h_y4.as_slice().flatten();
            let err = max_abs_diff(h_y_flat, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gdim, 1, 1, BDIM_IDIOM, 1, 1, 0);
                layernorm_vectorized::launch(cfg, ctx, md, &d_x4, &d_g4, &d_b4, &mut d_y4).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!(
                "layernorm vectorized SeGuRu: {:8.2} us/iter  (M={} N={}, max_err={:.3e})",
                us, m, n, err
            );
        }
    });
}
