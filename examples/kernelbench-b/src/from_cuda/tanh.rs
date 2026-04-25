//! Translation of `cuda/tanh.cu` into SeGuRu.
//!
//! Mirrors the CUDA kernel shape:
//!   - `float4` vectorized body over `n4 = n / 4` elements
//!   - scalar tail over the remaining `n - n4*4` elements
//!   - block size = 256

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

#[gpu::cuda_kernel]
pub fn tanh_vec_kernel(x4: &[Float4], y4: &mut [Float4], n4: u32) {
    let mut y_chunk = chunk_mut(y4, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n4 {
        let v: Float4 = x4[idx as usize];
        let mut r = Float4::new([0.0; 4]);
        r[0] = GPUDeviceFloatIntrinsics::tanh(v[0]);
        r[1] = GPUDeviceFloatIntrinsics::tanh(v[1]);
        r[2] = GPUDeviceFloatIntrinsics::tanh(v[2]);
        r[3] = GPUDeviceFloatIntrinsics::tanh(v[3]);
        y_chunk[0] = r;
    }
}

#[gpu::cuda_kernel]
pub fn tanh_tail_kernel(x: &[f32], y: &mut [f32], n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        y_chunk[0] = GPUDeviceFloatIntrinsics::tanh(x[idx as usize]);
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
    assert_eq!(shape.len(), 2, "tanh: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];
    let n4 = n / 4;
    let tail = n - n4 * 4;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); n4];
    let h_x_tail: Vec<f32> = h_x[n4 * 4..].to_vec();
    let mut h_y_tail = vec![0f32; tail];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();
    let d_x_tail = ctx.new_tensor_view(h_x_tail.as_slice()).unwrap();
    let mut d_y_tail = ctx.new_tensor_view(h_y_tail.as_mut_slice()).unwrap();

    let bs: u32 = 256;
    let n4u = n4 as u32;
    let tail_u = tail as u32;
    let gs_vec: u32 = if n4u > 0 { n4u.div_ceil(bs) } else { 1 };
    let gs_tail: u32 = if tail_u > 0 { tail_u.div_ceil(bs) } else { 1 };

    let launch = |d_x4: &_, d_y4: &mut _, d_xt: &_, d_yt: &mut _| {
        if n4u > 0 {
            let cfg = gpu_host::gpu_config!(gs_vec, 1, 1, bs, 1, 1, 0);
            tanh_vec_kernel::launch(cfg, ctx, md, d_x4, d_y4, n4u).unwrap();
        }
        if tail_u > 0 {
            let cfg = gpu_host::gpu_config!(gs_tail, 1, 1, bs, 1, 1, 0);
            tanh_tail_kernel::launch(cfg, ctx, md, d_xt, d_yt, tail_u).unwrap();
        }
    };

    launch(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        launch(&d_x4, &mut d_y4, &d_x_tail, &mut d_y_tail);
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

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
