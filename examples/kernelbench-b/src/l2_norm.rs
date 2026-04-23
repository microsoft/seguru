//! L2 normalization along dim=1 of a 2-D tensor [B, D].
//!
//! PyTorch reference (see `python/compare2.py`):
//!     s = sqrt(sum(x*x, dim=-1, keepdim=True) + 1e-12)   # shape (B, 1)
//!     y = x / s                                           # shape (B, D)
//!
//! Strategy (mirrors raw CUDA reference `cuda/l2_norm.cu`):
//!   **one block per row, cooperative reduction**.
//!
//!   - `gs = B`, `bs = BLOCK = 256`. 256 threads per row × 4096 rows = ~1M
//!     resident threads, well above the A100 saturation threshold. The previous
//!     1-thread-per-row implementation only launched `B` active threads and was
//!     ~8× slower than raw CUDA.
//!   - Pass 1: each thread accumulates `sum_sq` over a strided slice of its row.
//!     Warp-reduce via `warp.redux(ReduxAdd, …)`, then cross-warp-reduce via
//!     shared memory (same XOR-butterfly broadcast trick as `softmax.rs`).
//!   - Broadcast `rstd = rsqrt(sum_sq + eps)` to every thread (all lanes of all
//!     warps end up with the full block sum after the second warp redux, so no
//!     extra shared-memory broadcast is needed).
//!   - Pass 2: every thread writes `x[i] * rstd` to its strided slice of `y`
//!     using a `reshape_map!` output chunk (Golden Rule #6 — no per-store
//!     bounds check).
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Row-Reduction Strategy: 1 block per row for small-B / large-D.
//!   - Golden Rule #7 (block/warp-per-row instead of 1 thread/row).
//!   - `ThreadWarpTile::redux` for cross-lane reduction.
//!   - `GpuShared` + `chunk_to_scope` for cross-warp shared-memory reduction.
//!   - `reshape_map!` + `chunk_mut` for strided per-thread output writes.
//!   - Subslice-per-row for O(1)-bounds-checked row reads.
//!   - Golden Rule #1: `u32` kernel params and indices.
//!   - Golden Rule #7 (host-side): `copy_to_host` before `write_bin`.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Thread};
use gpu::prelude::*;

/// Threads per block. Must be a multiple of 32 and divide `D`.
const BLOCK: u32 = 256;

/// Fused l2-norm: one block per row. Pass 1 computes sum-of-squares
/// cooperatively via warp redux + shared-memory cross-warp reduce; pass 2
/// writes `y[i] = x[i] * rsqrt(sum_sq + eps)` for the whole row.
#[gpu::cuda_kernel]
pub fn l2_norm_kernel(x: &[f32], y: &mut [f32], D: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One slot per warp for cross-warp reduction (up to BLOCK=1024 → 32 warps).
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();

    // Subslice the row once — O(1) bounds checks for all element reads.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Pass 1: strided sum of squares ---
    let mut local = 0.0f32;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        local += v * v;
        i += block_dim::<DimX>();
    }

    // Warp-level reduce.
    let warp_sum = warp.redux(ReduxAdd, local);

    // Lane 0 of each warp publishes its partial to shared memory.
    {
        let mut smem_chunk = smem_sumsq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    // Cross-warp reduce: every warp loads all warp-partials from smem
    // (lanes 0..num_warps), pads the rest with 0, and does another XOR-butterfly.
    // All lanes of all warps then hold the full block sum — no extra broadcast.
    let smem_val = if lane_id < num_warps {
        smem_sumsq[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sumsq = warp.redux(ReduxAdd, smem_val);

    let rstd = (block_sumsq + eps).rsqrt();

    // --- Pass 2: strided write of `x * rstd` ---
    //
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D
    //   t0 ∈ [0, BLOCK)          — thread within block
    //   i0 ∈ [0, D / BLOCK)      — slot index per thread
    //   t1 ∈ [0, grid_dim)       — block (== row) index
    let out_map = reshape_map!(
        [D / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D {
        y_chunk[slot] = x_row[i as usize] * rstd;
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
    assert_eq!(shape.len(), 2, "l2_norm: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(
        d % BLOCK as usize == 0,
        "D ({d}) must be divisible by BLOCK ({BLOCK}) for reshape_map output"
    );
    let n_total = b * d;

    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb: u32 = b as u32;
    let dd: u32 = d as u32;
    let eps: f32 = 1e-12;

    let bs: u32 = BLOCK;
    let gs: u32 = bb; // one block per row

    // Warm up once before timing.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd, eps).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd, eps).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before reading / persisting.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
