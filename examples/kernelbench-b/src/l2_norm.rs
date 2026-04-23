//! L2 normalization along dim=1 of a 2-D tensor [B, D].
//!
//! PyTorch reference:
//!     s = sqrt(sum(x**2, dim=1, keepdim=True) + 1e-12)  # shape (B, 1)
//!     y = x / s                                          # shape (B, D)
//!
//! Strategy: two kernels, both elementwise w.r.t. their output.
//!   1. `l2_norm_reduce` — one thread per row; loops over D contiguous
//!      elements, accumulates sum of squares, writes
//!      inv_scale = rsqrt(sum_sq + eps).
//!   2. `l2_norm_apply`  — one thread per output element; multiplies x by
//!      the pre-computed inv_scale for its row.
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: `u32` kernel params and indices.
//!   - Golden Rule #2: subslice for row access — one bounds check per row.
//!   - Golden Rule #6: `chunk_mut` + `MapContinuousLinear::new(1)`.
//!   - Golden Rule #7 (copy_to_host before write_bin).

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn l2_norm_reduce(
    x: &[f32],
    inv_scale: &mut [f32],
    B: u32,
    D: u32,
    eps: f32,
) {
    let mut out = chunk_mut(inv_scale, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < B {
        let row = &x[(idx * D) as usize..((idx + 1) * D) as usize];
        let mut sumsq = 0.0f32;
        let mut i: u32 = 0;
        while i < D {
            let v = row[i as usize];
            sumsq += v * v;
            i += 1;
        }
        out[0] = (sumsq + eps).rsqrt();
    }
}

#[gpu::cuda_kernel]
pub fn l2_norm_apply(
    x: &[f32],
    inv_scale: &[f32],
    y: &mut [f32],
    D: u32,
    N: u32,
) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < N {
        let row = idx / D;
        out[0] = x[idx as usize] * inv_scale[row as usize];
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
    assert_eq!(shape.len(), 2, "l2_norm: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    let n_total = b * d;

    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y   = vec![0f32; n_total];
    let mut h_inv = vec![0f32; b];

    let d_x       = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y   = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let mut d_inv = ctx.new_tensor_view(h_inv.as_mut_slice()).unwrap();

    let bb: u32 = b as u32;
    let dd: u32 = d as u32;
    let nn: u32 = n_total as u32;
    let eps: f32 = 1e-12;

    let bs: u32 = 256;
    let gs_red: u32 = bb.div_ceil(bs);
    let gs_app: u32 = nn.div_ceil(bs);

    // Warm up once before timing.
    {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        l2_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, dd, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        l2_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, dd, nn).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        l2_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, dd, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        l2_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, dd, nn).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        l2_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, dd, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        l2_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, dd, nn).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy_to_host before write_bin.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_inv);
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
