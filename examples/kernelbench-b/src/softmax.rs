//! Row-wise softmax over the last dimension of a 2-D tensor [B, D].
//!
//! PyTorch reference:
//!   y = torch.softmax(x, dim=-1)
//!   = exp(x - row_max) / sum(exp(x - row_max))
//!
//! Strategy (per skill doc "Softmax Recipe"):
//!   Single-kernel fused online softmax (Milakov-Gimelshein), 1 block per row.
//!   The skill doc explicitly warns that the 2-kernel stats-then-apply split
//!   is ~1.3× slower; fused saves a launch and the round trip through
//!   `row_max[]` / `row_sum[]` global buffers.
//!
//!   - blockDim = 256 (8 warps), gridDim = B.
//!   - Pass 1: each thread strides across the row, running the online
//!     (old_max → new_max rescale) recurrence to fuse max+sum in one sweep.
//!   - Block-wide max reduce: warp.redux(ReduxMax) → smem slot per warp
//!     → second warp.redux where every lane loads the `num_warps` entries
//!     (padded with -inf) so every thread gets `block_max` without a
//!     broadcast.
//!   - Rescale each thread's partial sum from local_max to block_max,
//!     then repeat the warp→smem→warp shape with ReduxAdd for block_sum.
//!   - Pass 2: re-read x_row (cheap, hot in L1) and write
//!     `exp(x - block_max) * inv_sum` via a reshape_map chunk_mut so the
//!     store-side bounds check is proved away.
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: u32 for all GPU-side indices / sizes.
//!   - Golden Rule #2 / subslice pattern: `x_row = &x[row*D..(row+1)*D]`
//!     once, index into it in both passes.
//!   - Golden Rule #3: MapContinuousLinear for 1D chunks.
//!   - Golden Rule #6: chunk_mut uses LOCAL indices (`y_chunk[slot]`).
//!   - Row-reduction strategy: 1 block per row (B=4096, D=8192 fits).
//!   - Softmax Recipe: online max+sum, NEG_INFINITY identity, hoisted inv_sum.
//!   - reshape_map! [t0, i0, t1] output layout eliminates store bounds check
//!     (requires D % BLOCK == 0 — asserted host-side).
//!   - Warp reductions: ThreadWarpTile::redux with ReduxMax / ReduxAdd.
//!   - Cross-warp reduction via `GpuShared::<[f32;32]>` + chunk_to_scope.
//!   - Golden Rule #7 (host): copy_to_host before write_bin.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Thread};
use gpu::prelude::*;

/// One block per row; 8 warps per block (BLOCK = 256).
#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[f32], y: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // Reserve 32 slots (max warps for BLOCK=1024). Unused slots stay at identity.
    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // Subslice the row once — O(1) bounds-check amortization over both passes.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Pass 1: online max+sum in a single strided traversal ---
    let mut local_max = f32::NEG_INFINITY;
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        let old_max = local_max;
        local_max = local_max.max(v);
        // Rescale running sum whenever the max grows, then add new exp term.
        local_sum *= GPUDeviceFloatIntrinsics::exp(old_max - local_max);
        local_sum += GPUDeviceFloatIntrinsics::exp(v - local_max);
        i += block_dim::<DimX>();
    }

    // --- Block-wide max reduce: warp redux → smem → warp redux ---
    let warp_max = warp.redux(ReduxMax, local_max);
    {
        let mut s = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_max;
        }
    }
    sync_threads();
    // Every warp loads the first `num_warps` slots, pads rest with -inf,
    // and runs another XOR-butterfly. All lanes end up with block_max.
    let sv = if lane_id < num_warps {
        smem_max[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let block_max = warp.redux(ReduxMax, sv);

    // Rescale each thread's partial sum from local_max → block_max.
    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);

    // --- Block-wide sum reduce (same shape; identity = 0.0) ---
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

    // Hoist the reciprocal out of the inner write loop (skill: div is slow).
    let inv_sum = 1.0f32 / block_sum;

    // --- Pass 2: write normalized output ---
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D
    //   t0 ∈ [0, BLOCK)       — thread within block
    //   i0 ∈ [0, D/BLOCK)     — slot index per thread
    //   t1 ∈ [0, grid_dim)    — block (== row) index
    // Requires D % BLOCK == 0 (asserted host-side).
    let out_map = reshape_map!(
        [D / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D {
        y_chunk[slot] =
            GPUDeviceFloatIntrinsics::exp(x_row[i as usize] - block_max) * inv_sum;
        i += block_dim::<DimX>();
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
    assert_eq!(shape.len(), 2, "softmax: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);

    // BLOCK=256. reshape_map! output requires D divisible by BLOCK.
    const BLOCK: u32 = 256;
    assert!(
        d % BLOCK as usize == 0,
        "D must be divisible by BLOCK ({BLOCK}) for reshape_map output layout"
    );

    let n_total = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let dd = d as u32;

    // 1 block per row.
    let bs: u32 = BLOCK;
    let gs: u32 = bb;

    // Untimed warmup launch.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();

    // Timed warmup.
    let warmup_iters: usize = 5;
    ctx.sync().unwrap();
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    ctx.sync().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before reading / persisting.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
