//! Translation of `cuda/sigmoid.cu` into SeGuRu.
//!
//! Mirrors the CUDA kernel shape:
//!   - `float4` vectorized body over `n4 = n / 4` elements
//!   - scalar tail over the remaining `n - n4*4` elements
//!   - block size = 256
//!
//! The CUDA source uses a grid-stride loop with a capped grid of 4096 blocks.
//! SeGuRu `chunk_mut` writes do not support grid-stride (skill-doc: "Grid-stride
//! loops DO NOT work"), so we launch exactly enough threads for one element
//! per thread while keeping the same `float4` vectorization and block size.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn sigmoid_vec_kernel(x4: &[Float4], y4: &mut [Float4], n4: u32) {
    let mut y_chunk = chunk_mut(y4, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n4 {
        let v: Float4 = x4[idx as usize];
        let mut r = Float4::new([0.0; 4]);
        r[0] = 1.0 / (1.0 + GPUDeviceFloatIntrinsics::exp(-v[0]));
        r[1] = 1.0 / (1.0 + GPUDeviceFloatIntrinsics::exp(-v[1]));
        r[2] = 1.0 / (1.0 + GPUDeviceFloatIntrinsics::exp(-v[2]));
        r[3] = 1.0 / (1.0 + GPUDeviceFloatIntrinsics::exp(-v[3]));
        y_chunk[0] = r;
    }
}

#[gpu::cuda_kernel]
pub fn sigmoid_tail_kernel(x: &[f32], y: &mut [f32], tail_count: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < tail_count {
        let v = x[idx as usize];
        y_chunk[0] = 1.0 / (1.0 + GPUDeviceFloatIntrinsics::exp(-v));
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path,
    out_dir: &Path,
    iters: usize,
    shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2, "sigmoid: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    let n4 = n / 4;
    let tail = n - n4 * 4;
    let h_x4: Vec<Float4> = h_x[..n4 * 4]
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); n4];
    let h_x_tail: Vec<f32> = h_x[n4 * 4..].to_vec();
    let mut h_y_tail: Vec<f32> = vec![0f32; tail];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();
    let d_x_tail = ctx.new_tensor_view(h_x_tail.as_slice()).unwrap();
    let mut d_y_tail = ctx.new_tensor_view(h_y_tail.as_mut_slice()).unwrap();

    let n4_u = n4 as u32;
    let tail_u = tail as u32;

    let bs: u32 = 256;
    let gs_vec: u32 = if n4_u > 0 { n4_u.div_ceil(bs) } else { 0 };
    let gs_tail: u32 = if tail_u > 0 { tail_u.div_ceil(bs) } else { 0 };

    let launch = |d_x4: &_, d_y4: &mut _, d_x_tail: &_, d_y_tail: &mut _| {
        if gs_vec > 0 {
            let cfg = gpu_host::gpu_config!(gs_vec, 1, 1, bs, 1, 1, 0);
            sigmoid_vec_kernel::launch(cfg, ctx, md, d_x4, d_y4, n4_u).unwrap();
        }
        if gs_tail > 0 {
            let cfg = gpu_host::gpu_config!(gs_tail, 1, 1, bs, 1, 1, 0);
            sigmoid_tail_kernel::launch(cfg, ctx, md, d_x_tail, d_y_tail, tail_u).unwrap();
        }
    };

    // Warm up once before timing.
    launch(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        launch(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        launch(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y4.copy_to_host(h_y4.as_mut_slice()).unwrap();
    if tail_u > 0 {
        d_y_tail.copy_to_host(h_y_tail.as_mut_slice()).unwrap();
    }
    drop(d_y4);
    drop(d_x4);
    drop(d_y_tail);
    drop(d_x_tail);

    let h_y4_flat: &[f32] = h_y4.as_slice().flatten();
    h_y[..h_y4_flat.len()].copy_from_slice(h_y4_flat);
    if tail > 0 {
        h_y[n4 * 4..].copy_from_slice(&h_y_tail);
    }

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
