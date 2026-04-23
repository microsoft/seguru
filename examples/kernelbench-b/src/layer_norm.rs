//! Layer normalization (no affine transform) for a 2-D tensor [B, D].
//!
//! PyTorch reference:
//!   y = F.layer_norm(x, (x.shape[-1],), eps=1e-5)
//!
//! Formula (per row):
//!   mean = sum(row) / D
//!   var  = sum((row - mean)^2) / D
//!   y    = (row - mean) / sqrt(var + eps)
//!
//! Strategy: one warp per row. A single fused pass accumulates `sum` and
//! `sum_sq` simultaneously, then two `warp.redux` calls reduce both across
//! lanes. A second pass writes the normalized output.
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Warp-strided subslice pattern for row reads (subslice ONCE, index into it).
//!   - `ThreadWarpTile::redux` for cross-lane reduction.
//!   - `reshape_map!` + `chunk_mut` for strided per-thread output writes.
//!   - `u32` kernel parameters and indices (Golden Rule #1).
//!   - Golden Rule #7: `copy_to_host` before `write_bin`.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::prelude::*;

/// Number of warps per block (block_dim = 32 * WARPS_PER_BLOCK).
const WARPS_PER_BLOCK: u32 = 8;

/// One warp per row. Fused mean+var in one pass, then normalized write.
#[gpu::cuda_kernel]
pub fn layer_norm_kernel(
    x: &[f32],
    y: &mut [f32],
    B: u32,
    D: u32,
    eps: f32,
) {
    let warp = ThreadWarpTile::<32>;
    let warps_per_block = warp.meta_group_size();
    let row = block_id::<DimX>() * warps_per_block + warp.subgroup_id();

    if row >= B {
        return;
    }

    let lane = warp.thread_rank();
    let row_off = row * D;

    // Subslice the row once — one bounds check for all accesses.
    let x_row = &x[row_off as usize..(row_off + D) as usize];

    // Fused pass: accumulate sum and sum_sq in a single traversal.
    let mut local_sum = 0.0f32;
    let mut local_sumsq = 0.0f32;
    let mut i = lane;
    while i < D {
        let v = x_row[i as usize];
        local_sum += v;
        local_sumsq += v * v;
        i += warp.size();
    }

    // Warp-wide reduce: two redux calls, one per accumulator.
    let sum: f32 = warp.redux(ReduxAdd, local_sum);
    let sumsq: f32 = warp.redux(ReduxAdd, local_sumsq);

    let inv_d = 1.0f32 / (D as f32);
    let mean = sum * inv_d;
    let var = sumsq * inv_d - mean * mean;
    let rstd = (var + eps).rsqrt();

    // Pass 2: write normalized output.
    // Each warp owns one row; each lane writes D/32 elements strided by 32.
    // layout [t0, i0, t1]: pos = t0 + i0*32 + t1*D
    //   t0 ∈ [0, 32)                              — lane within warp
    //   i0 ∈ [0, D/32)                            — slot index per lane
    //   t1 ∈ [0, warps_per_block * grid_dim)      — warp (== row) index
    let out_map = reshape_map!(
        [D / 32] | [32, warps_per_block * grid_dim::<DimX>()] => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = lane;
    while i < D {
        y_chunk[slot] = (x_row[i as usize] - mean) * rstd;
        i += warp.size();
        slot += 1;
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
    assert_eq!(shape.len(), 2, "layer_norm: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(
        b % WARPS_PER_BLOCK as usize == 0,
        "B must be divisible by WARPS_PER_BLOCK ({WARPS_PER_BLOCK})"
    );
    assert!(
        d % 32 == 0,
        "D must be divisible by 32 (warp size) for reshape_map"
    );

    let n_total = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let dd = d as u32;
    let eps: f32 = 1e-5;

    // block_dim = 32 * WARPS_PER_BLOCK; grid_dim = B / WARPS_PER_BLOCK
    let bs: u32 = 32 * WARPS_PER_BLOCK;
    let gs: u32 = bb / WARPS_PER_BLOCK;

    // Warmup
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, bb, dd, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, bb, dd, eps).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before reading h_y.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
