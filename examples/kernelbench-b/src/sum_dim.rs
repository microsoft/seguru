//! `torch.sum(x, dim=-1)`: sum reduction over the last dimension.
//!
//! Input:  `[B, D]` (contiguous fp32)
//! Output: `[B]`    (one f32 scalar per row)
//!
//! Strategy: one block per row, cooperative reduction (Golden Rule #7 —
//! row-reduction rule). With `B=128, D=16384`, a 1-thread-per-row kernel
//! only spawns 128 threads and leaves an A100 ~1% busy. Instead:
//!
//!   1. Launch `gs = B`, `bs = 256` (8 warps per block).
//!   2. Each thread grid-strides over its row, accumulating a partial sum
//!      (stride = `block_dim`).
//!   3. Warp-reduce via `warp.redux(ReduxAdd, partial)`.
//!   4. Lane 0 of each warp writes its warp-sum to shared memory.
//!   5. Warp 0 loads the `num_warps` slots (padding the rest with 0.0) and
//!      runs another warp-reduce for the final block sum.
//!   6. Thread 0 writes the scalar result to `y[row]`.
//!
//! Mirrors the structural pattern used in `softmax.rs` / `layer_norm.rs` /
//! `rms_norm.rs`, and the raw-CUDA reference in `cuda/sum_dim.cu`.
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: `u32` kernel params and indices.
//!   - Golden Rule #7 (new): 1-block-per-row cooperative row reduction.
//!   - Golden Rule #7 (host-side): `copy_to_host` before `write_bin`.
//!   - "Warp reductions": `ThreadWarpTile::redux` with `ReduxAdd`.
//!   - "Tree reduction in shared memory": `GpuShared` + warp-0 second reduce.
//!   - `chunk_to_scope(grid2block, ...)` for writing one value per block.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Thread};
use gpu::prelude::*;

/// One block per row. `BLOCK = 256` → 8 warps → smem reserves 32 slots
/// (safe up to `BLOCK = 1024`).
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[f32], y: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One shared-memory slot per warp (up to 32 warps).
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // One output element per block — chained Grid→Block→Thread scope.
    let mut sum_out = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    // Subslice the row once for O(1) bounds checks.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Thread-local grid-strided partial sum ---
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D {
        local_sum += x_row[i as usize];
        i += block_dim::<DimX>();
    }

    // --- Warp-reduce ---
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut smem_chunk = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    // --- Cross-warp reduce (all warps load the same smem and reduce; every
    //     lane ends up with the block sum — no broadcast needed). ---
    let smem_val = if lane_id < num_warps { smem_sum[lane_id as usize] } else { 0.0f32 };
    let block_sum = warp.redux(ReduxAdd, smem_val);

    if tid == 0 {
        sum_out[0] = block_sum;
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
    let gs: u32 = bb; // one block per row

    // Warmup (untimed).
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup timing.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7 (host-side): copy device → host before writing to disk.
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
