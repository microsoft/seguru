//! Row-wise softmax over the last dimension of a 2-D tensor [B, D].
//!
//! PyTorch reference:
//!     m = max(row)
//!     s = sum(exp(row - m))
//!     y = exp(row - m) / s
//!
//! Strategy: two kernels.
//!   1. `softmax_stats_kernel` — one block per row; threads cooperatively compute the
//!      row max and row sum-of-exp using the numerically-stable online algorithm,
//!      warp-shuffle reductions + shared memory cross-warp reduction.  Thread 0 of
//!      each block writes `row_max[row]` and `row_sum[row]`.
//!   2. `softmax_apply_kernel` — one thread per element; reads the precomputed stats
//!      and writes the normalised output.
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rules #1, #3, #6: u32 params, MapContinuousLinear, chunk_mut local [0].
//!   - Golden Rule #7: copy_to_host before write_bin.
//!   - Warp reductions: ThreadWarpTile::redux with ReduxMax and ReduxAdd.
//!   - GpuShared + chunk_to_scope for cross-warp reduction via shared memory.
//!   - chunk_to_scope(grid2block, ...) for writing one value per block.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Thread};
use gpu::prelude::*;

/// Pass 1: block-per-row cooperative stats.
///
/// Each of the `D` row elements is visited by the thread with `thread_id % BLOCK == element % BLOCK`.
/// Online max+sum accumulation avoids a second pass over the row.
/// BLOCK = 256 → 8 warps → smem needs 8 slots (we reserve 32 for safety up to BLOCK=1024).
#[gpu::cuda_kernel]
pub fn softmax_stats_kernel(x: &[f32], row_max: &mut [f32], row_sum: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // Shared memory slots: one per warp (up to 32 warps = BLOCK 1024).
    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // One output element per block — chained Grid→Block→Thread scope.
    let mut max_out = row_max
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    let mut sum_out = row_sum
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    // Subslice the row once for O(1) bounds checks (skill doc: subslice pattern).
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Online max+sum (single pass over the row) ---
    let mut local_max = f32::NEG_INFINITY;
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        let old_max = local_max;
        local_max = local_max.max(v);
        local_sum *= GPUDeviceFloatIntrinsics::exp(old_max - local_max);
        local_sum += GPUDeviceFloatIntrinsics::exp(v - local_max);
        i += block_dim::<DimX>();
    }

    // --- Warp reduce max (XOR-butterfly → all lanes get warp max) ---
    let warp_max = warp.redux(ReduxMax, local_max);
    // Lane 0 writes warp max to shared memory.
    {
        let mut smem_max_chunk = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_max_chunk[0] = warp_max;
        }
    }
    sync_threads();

    // Cross-warp max reduce: every warp loads the `num_warps` warp-maxes from smem
    // (lanes 0..num_warps), pads the rest with -inf, and runs another XOR-butterfly.
    // Because all warps load the *same* data, all lanes in all warps end up with
    // the global block max — no extra broadcast needed.
    let smem_val = if lane_id < num_warps { smem_max[lane_id as usize] } else { f32::NEG_INFINITY };
    let block_max = warp.redux(ReduxMax, smem_val);

    // Rescale local_sum from local_max to block_max.
    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);

    // --- Warp reduce sum ---
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut smem_sum_chunk = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_sum_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    // Cross-warp sum reduce (same broadcast trick as max).
    let smem_val = if lane_id < num_warps { smem_sum[lane_id as usize] } else { 0.0f32 };
    let block_sum = warp.redux(ReduxAdd, smem_val);

    // Thread 0 writes the row statistics (skill doc: chunk_to_scope, one per block).
    if tid == 0 {
        max_out[0] = block_max;
        sum_out[0] = block_sum;
    }
}

/// Pass 2: elementwise normalisation.  One thread per output element.
#[gpu::cuda_kernel]
pub fn softmax_apply_kernel(
    x: &[f32],
    row_max: &[f32],
    row_sum: &[f32],
    y: &mut [f32],
    D: u32,
    N: u32, // B * D
) {
    let mut y_out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < N {
        let row = idx / D;
        let m = row_max[row as usize];
        let s = row_sum[row as usize];
        y_out[0] = GPUDeviceFloatIntrinsics::exp(x[idx as usize] - m) / s;
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
    assert_eq!(shape.len(), 2, "softmax: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    let n_total = b * d;

    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];
    let mut h_max = vec![0f32; b];
    let mut h_sum = vec![0f32; b];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let mut d_max = ctx.new_tensor_view(h_max.as_mut_slice()).unwrap();
    let mut d_sum = ctx.new_tensor_view(h_sum.as_mut_slice()).unwrap();

    let bb = b as u32;
    let dd = d as u32;
    let nn = n_total as u32;

    // Stats kernel: one block per row.
    let bs_stats: u32 = 256;
    let gs_stats: u32 = bb; // one block per row

    // Apply kernel: one thread per element.
    let bs_apply: u32 = 256;
    let gs_apply: u32 = nn.div_ceil(bs_apply);

    // Warm up.
    {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, bs_stats, 1, 1, 0);
        softmax_stats_kernel::launch(cfg_s, ctx, md, &d_x, &mut d_max, &mut d_sum, dd).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, bs_apply, 1, 1, 0);
        softmax_apply_kernel::launch(cfg_a, ctx, md, &d_x, &*d_max, &*d_sum, &mut d_y, dd, nn)
            .unwrap();
    }
    ctx.sync().unwrap();

    // Warmup timing.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, bs_stats, 1, 1, 0);
        softmax_stats_kernel::launch(cfg_s, ctx, md, &d_x, &mut d_max, &mut d_sum, dd).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, bs_apply, 1, 1, 0);
        softmax_apply_kernel::launch(cfg_a, ctx, md, &d_x, &*d_max, &*d_sum, &mut d_y, dd, nn)
            .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, bs_stats, 1, 1, 0);
        softmax_stats_kernel::launch(cfg_s, ctx, md, &d_x, &mut d_max, &mut d_sum, dd).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, bs_apply, 1, 1, 0);
        softmax_apply_kernel::launch(cfg_a, ctx, md, &d_x, &*d_max, &*d_sum, &mut d_y, dd, nn)
            .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before reading / persisting.
    d_y.copy_to_host(&mut h_y).unwrap();

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
