//! Translation of `cuda/tanh.cu` into SeGuRu.
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
use gpu_host::{TensorView, TensorViewMut};

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
pub fn tanh_tail_kernel(x: &[f32], y: &mut [f32], tail_start: u32, n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = tail_start + idx;
    if i < n {
        y_chunk[0] = GPUDeviceFloatIntrinsics::tanh(x[i as usize]);
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path,
    out_dir: &Path,
    iters: usize,
    shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2, "tanh: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    // Reinterpret f32 tensor views as Float4 views for the vectorized path.
    let d_x4 = unsafe { &*(&d_x as *const _ as *const TensorView<'_, [Float4]>) };
    let d_y4 = unsafe { &mut *(&mut d_y as *mut _ as *mut TensorViewMut<'_, [Float4]>) };

    let nn   = n as u32;
    let n4   = nn / 4;
    let tail_start = n4 * 4;
    let tail_count = nn - tail_start;

    let bs: u32 = 256;
    let gs_vec:  u32 = if n4 > 0 { n4.div_ceil(bs) } else { 0 };
    let gs_tail: u32 = if tail_count > 0 { tail_count.div_ceil(bs) } else { 0 };

    let launch = |ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
                  md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
                  d_x4: &TensorView<'_, [Float4]>,
                  d_y4: &mut TensorViewMut<'_, [Float4]>,
                  d_x:  &TensorView<'_, [f32]>,
                  d_y:  &mut TensorViewMut<'_, [f32]>| {
        if gs_vec > 0 {
            let cfg = gpu_host::gpu_config!(gs_vec, 1, 1, bs, 1, 1, 0);
            tanh_vec_kernel::launch(cfg, ctx, md, d_x4, d_y4, n4).unwrap();
        }
        if gs_tail > 0 {
            let cfg = gpu_host::gpu_config!(gs_tail, 1, 1, bs, 1, 1, 0);
            tanh_tail_kernel::launch(cfg, ctx, md, d_x, d_y, tail_start, nn).unwrap();
        }
    };

    // Warm up once before timing.
    launch(ctx, md, d_x4, d_y4, &d_x, &mut d_y);
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        launch(ctx, md, d_x4, d_y4, &d_x, &mut d_y);
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        launch(ctx, md, d_x4, d_y4, &d_x, &mut d_y);
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
