//! `torch.sum(x, dim=-1)`: sum reduction over the last dimension.
//!
//! Input:  `[B, D]` (contiguous fp32)
//! Output: `[B]`    (one f32 scalar per row)
//!
//! Strategy: one thread per row. The thread subslices its row from the flat
//! input, accumulates a scalar sum, and writes one output value. Mirrors
//! the single-thread-per-pixel pattern from `rms_norm.rs` (reduce pass).
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: `u32` kernel params and indices.
//!   - "Memory Write Patterns": `chunk_mut` + `MapContinuousLinear::new(1)`,
//!     LOCAL slot index `out[0]`.
//!   - "Inner Loop Optimization": subslice `&x[row*D..(row+1)*D]` for one
//!     bounds check over the whole row.
//!   - Golden Rule #7: `copy_to_host` before `write_bin`.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn sum_dim_kernel(
    x: &[f32],
    y: &mut [f32],
    B: u32,
    D: u32,
) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let row = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < B {
        // Subslice eliminates per-element bounds checks inside the accumulation loop.
        let x_row = &x[(row * D) as usize..((row + 1) * D) as usize];
        let mut sum = 0.0f32;
        for v in x_row {
            sum += v;
        }
        out[0] = sum;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir:  &Path,
    out_dir: &Path,
    iters:   usize,
    shape:   &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2, "sum_dim: shape must be [B, D]");
    let (b, d) = (shape[0], shape[1]);
    let n_total = b * d;

    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; b];

    let d_x     = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let dd = d as u32;
    let bs: u32 = 256;
    let gs: u32 = bb.div_ceil(bs);

    // Warmup (untimed).
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, bb, dd).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup timing.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, bb, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, bb, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before writing to disk.
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
