//! LeakyReLU — translated from `cuda/leaky_relu.cu`.
//!
//! Source CUDA kernel:
//!   - `float4` vectorized grid-stride loop over `n4 = n / 4` elements
//!   - elementwise `v > 0 ? v : v * slope` on each of the 4 lanes
//!   - scalar tail loop over the final `n - n4*4` elements
//!   - block = 256, grid capped at `sm_count * 32`
//!
//! SeGuRu port:
//!   - Mirror the `float4` vectorization with `Float4`.
//!   - Per the skill doc, grid-stride write loops are incompatible with
//!     `chunk_mut`, so we launch one thread per `Float4` lane (1 element per
//!     thread) — the documented SeGuRu equivalent of the CUDA grid-stride
//!     pattern.
//!   - Scalar tail handled by a second kernel that writes the trailing
//!     `n - n4*4` f32 slots.  For the benchmark shape `[2048, 65536]` the
//!     tail is empty, but we keep the path to mirror the .cu one-for-one.
//!   - Block size 256, same slope semantics, same strict `>` comparison.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn leaky_relu_vec_kernel(x4: &[Float4], y4: &mut [Float4], slope: f32, n4: u32) {
    let mut y_chunk = chunk_mut(y4, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n4 {
        let v: Float4 = x4[idx as usize];
        let mut out = Float4::new([0.0; 4]);
        for k in 0..4 {
            let vk = v[k];
            out[k] = if vk > 0.0 { vk } else { vk * slope };
        }
        y_chunk[0] = out;
    }
}

#[gpu::cuda_kernel]
pub fn leaky_relu_tail_kernel(x: &[f32], y: &mut [f32], slope: f32, n_tail: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n_tail {
        let v = x[idx as usize];
        y_chunk[0] = if v > 0.0 { v } else { v * slope };
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
    assert_eq!(shape.len(), 2, "leaky_relu: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    // Build Float4 host buffers for the vectorized body.
    let n4 = n / 4;
    let tail = n - n4 * 4;

    let mut h_x4: Vec<Float4> = Vec::with_capacity(n4);
    for c in h_x[..n4 * 4].chunks_exact(4) {
        h_x4.push(Float4::new([c[0], c[1], c[2], c[3]]));
    }
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); n4];

    // Tail scratch buffers (empty when n % 4 == 0).
    let h_x_tail: Vec<f32> = h_x[n4 * 4..].to_vec();
    let mut h_y_tail: Vec<f32> = vec![0f32; tail];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let d_x_tail = ctx.new_tensor_view(h_x_tail.as_slice()).unwrap();
    let mut d_y_tail = ctx.new_tensor_view(h_y_tail.as_mut_slice()).unwrap();

    let slope: f32 = 0.01;
    let bs: u32 = 256;

    let n4_u = n4 as u32;
    let gs_vec: u32 = if n4_u == 0 { 1 } else { n4_u.div_ceil(bs) };

    let tail_u = tail as u32;
    let gs_tail: u32 = if tail_u == 0 { 1 } else { tail_u.div_ceil(bs) };

    let launch_once = |d_x4: &_, d_y4: &mut _, d_xt: &_, d_yt: &mut _| {
        if n4_u > 0 {
            let cfg = gpu_host::gpu_config!(gs_vec, 1, 1, bs, 1, 1, 0);
            leaky_relu_vec_kernel::launch(cfg, ctx, md, d_x4, d_y4, slope, n4_u).unwrap();
        }
        if tail_u > 0 {
            let cfg = gpu_host::gpu_config!(gs_tail, 1, 1, bs, 1, 1, 0);
            leaky_relu_tail_kernel::launch(cfg, ctx, md, d_xt, d_yt, slope, tail_u).unwrap();
        }
    };

    // Warm up once before timing.
    launch_once(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        launch_once(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        launch_once(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
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

    // Flatten Float4 body + scalar tail back into h_y.
    {
        let flat: &[f32] = h_y4.as_slice().flatten();
        h_y[..flat.len()].copy_from_slice(flat);
    }
    if tail > 0 {
        h_y[n4 * 4..].copy_from_slice(&h_y_tail);
    }

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
