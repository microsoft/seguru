//! Row-wise sum reduction over the last dim of a 2-D tensor [B, D] → [B].
//!
//! PyTorch reference:
//!     y = torch.sum(x, dim=-1)     // shape: [B]
//!
//! Strategy (cuda-to-seguru-porting-skill.md "Row-Reduction Strategy"):
//!   - B=4096 rows, D=16384 row width → "1 block per row + float4 loads".
//!   - grid = B blocks, block = 256 threads (8 warps).
//!   - Each thread strides across its row as Float4 (D4 = D/4 = 4096) and
//!     accumulates a scalar partial = v.x + v.y + v.z + v.w.
//!   - Block-wide sum: warp.redux(ReduxAdd) → one slot per warp in smem →
//!     second warp.redux (with 0.0 padding for lanes ≥ num_warps).
//!   - Thread 0 writes the scalar result y[row].
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: u32 for all GPU-side indices / sizes.
//!   - Golden Rule #2: subslice the row once to amortize bounds checks.
//!   - Golden Rule #7 (host): copy_to_host before write_bin.
//!   - Row-Reduction Strategy: 1 block per row.
//!   - "Always vectorize when D % 4 == 0 (Float4 loads)" — kernel takes
//!     `x: &[Float4]`, accumulates `v.x + v.y + v.z + v.w` per lane.
//!   - Warp reductions: ThreadWarpTile::<32> + ReduxAdd.
//!   - Cross-warp reduction via `GpuShared::<[f32; 32]>` + chunk_to_scope.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Grid, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;

/// One block per row; `BLOCK=256` threads (8 warps) per block. `D4 = D/4`.
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[Float4], y: &mut [f32], D4: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();

    // Scratch: one slot per warp (≤32 warps for BLOCK≤1024).
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // Subslice this block's Float4 row once.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // Strided Float4 accumulation.
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        local_sum += v[0] + v[1] + v[2] + v[3];
        i += block_dim::<DimX>();
    }

    // Warp-level reduce, then cross-warp via smem, then second warp-reduce.
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut s = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_sum;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, sv);

    // One scalar output per block. Chain Grid→Block→Thread scope so every
    // thread sees the block's single output slot (avoids the Thread-scope
    // default that would create grid*block slots, OOB for y of length B).
    let mut y_chunk = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    if tid == 0 {
        y_chunk[0] = block_sum;
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
    assert_eq!(shape.len(), 2, "sum_dim: shape=[B, D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(
        d % 4 == 0,
        "sum_dim: D must be divisible by 4 for Float4 loads"
    );

    let n_in = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_in);
    let mut h_y = vec![0f32; b];

    // Host-side: repack f32 into Float4. (cudaMalloc guarantees 16B alignment.)
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let d4 = (d / 4) as u32;
    const BLOCK: u32 = 256;
    let bs: u32 = BLOCK;
    let gs: u32 = b as u32;

    // Untimed warmup.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();

    // Timed warmup.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before persisting.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_x4);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
