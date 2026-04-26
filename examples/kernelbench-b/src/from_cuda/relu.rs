//! ReLU elementwise kernel — translated from `cuda/relu.cu`.
//!
//! CUDA strategy being mirrored:
//!   - Reinterpret `x`/`y` as `float4*` (vector-of-4 loads/stores).
//!   - block_size = 256, grid-stride loop over `n4 = n / 4` Float4 elements.
//!   - Scalar tail for `n % 4 != 0`.
//!
//! SeGuRu adaptation:
//!   - Grid-stride loops don't compose with `chunk_mut` (skill-doc golden
//!     rule). We launch enough threads for one Float4 per thread instead,
//!     which preserves the vectorized memory-access pattern.
//!   - A second scalar kernel handles the tail when `n % 4 != 0`.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn relu_kernel_vec4(x: &[Float4], y: &mut [Float4], n4: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n4 {
        let v: Float4 = x[idx as usize];
        let mut out: Float4 = Float4::new([0.0; 4]);
        for k in 0..4 {
            let vk = v[k];
            out[k] = if vk > 0.0 { vk } else { 0.0 };
        }
        y_chunk[0] = out;
    }
}

#[gpu::cuda_kernel]
pub fn relu_kernel_tail(x: &[f32], y: &mut [f32], tail_start: u32, n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let gi = tail_start + idx;
    if gi < n {
        let v = x[gi as usize];
        y_chunk[0] = if v > 0.0 { v } else { 0.0 };
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
    assert_eq!(shape.len(), 2, "relu: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];
    let n4 = n / 4;
    let tail = n - n4 * 4;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    // Repack host buffers as Float4. Tail (if any) stays in the scalar
    // portion for the second-kernel pass.
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); n4];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    // Scalar tail uses the original f32 slices.
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bs: u32 = 256;
    let nn4 = n4 as u32;
    let gs4: u32 = if nn4 == 0 { 1 } else { nn4.div_ceil(bs) };
    let nn = n as u32;
    let tail_start: u32 = (n4 * 4) as u32;
    let gs_tail: u32 = if tail == 0 {
        0
    } else {
        (tail as u32).div_ceil(bs)
    };

    let launch =
        |ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>, d_x4: &_, d_y4: &mut _, d_x: &_, d_y: &mut _| {
            if nn4 > 0 {
                let cfg = gpu_host::gpu_config!(gs4, 1, 1, bs, 1, 1, 0);
                relu_kernel_vec4::launch(cfg, ctx, md, d_x4, d_y4, nn4).unwrap();
            }
            if gs_tail > 0 {
                let cfg = gpu_host::gpu_config!(gs_tail, 1, 1, bs, 1, 1, 0);
                relu_kernel_tail::launch(cfg, ctx, md, d_x, d_y, tail_start, nn).unwrap();
            }
        };

    // Warm up once before timing.
    launch(ctx, &d_x4, &mut d_y4, &d_x, &mut d_y);
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        launch(ctx, &d_x4, &mut d_y4, &d_x, &mut d_y);
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        launch(ctx, &d_x4, &mut d_y4, &d_x, &mut d_y);
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Copy both outputs back and merge (Float4 holds the main body, scalar
    // tail kernel wrote directly into `h_y`).
    d_y4.copy_to_host(h_y4.as_mut_slice()).unwrap();
    if tail > 0 {
        d_y.copy_to_host(&mut h_y).unwrap();
    }
    drop(d_y);
    drop(d_x);
    drop(d_y4);
    drop(d_x4);

    // Overwrite the vectorized portion of h_y from h_y4 (tail already written
    // via d_y if present).
    {
        let flat: &[f32] = h_y4.as_slice().flatten();
        h_y[..flat.len()].copy_from_slice(flat);
    }

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
