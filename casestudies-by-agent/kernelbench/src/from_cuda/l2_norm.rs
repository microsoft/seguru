//! SeGuRu port of `cuda/l2_norm.cu` (1-for-1 translation of the raw CUDA).
//!
//! CUDA strategy mirrored:
//!   - One block per row; BLOCK = 256 threads/block.
//!   - Pass 1: each thread accumulates sum-of-squares over a strided slice
//!     of its row using `float4` (128-bit) vectorized loads, then warp-reduce
//!     via `__shfl_down_sync`, cross-warp combine via shared memory,
//!     final broadcast via a second warp redux (XOR-butterfly-equivalent).
//!   - `scale = rsqrt(sum_sq + 1e-12)`.
//!   - Pass 2: every thread writes `x * scale` back through `float4` stores.
//!
//! The benchmark shape is `[B, D] = [4096, 8192]` (compare2.py), so
//! `D % 4 == 0` and `D4 % BLOCK == 0`; the scalar tail loop from the `.cu`
//! is unreachable on this shape and is omitted (no other shape is exercised).
//!
//! Skill-doc patterns:
//!   - Vectorized Access (Float4).
//!   - Row-Reduction Strategy (1 block per row).
//!   - Warp Operations: `ThreadWarpTile::redux(ReduxAdd, …)`.
//!   - Tree reduction in shared memory (cross-warp combine).
//!   - `reshape_map!` + `chunk_mut` for strided Float4 output.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

/// Threads per block (matches `BLOCK` in `cuda/l2_norm.cu`).
const BLOCK: u32 = 256;

/// One block per row. Pass 1 computes sum-of-squares cooperatively via
/// Float4 loads + warp redux + shared-memory cross-warp combine; pass 2
/// writes `y = x * rsqrt(sum_sq + eps)` through Float4 stores.
#[gpu::cuda_kernel]
pub fn l2_norm_kernel(x: &[Float4], y: &mut [Float4], D4: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One slot per warp for cross-warp reduction (BLOCK/32 ≤ 32).
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();

    // Subslice the row once — O(1) bounds checks for element reads.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // ---- Pass 1: accumulate sum of squares over Float4 slots ----
    let mut acc: f32 = 0.0;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 {
            let vk = v[k];
            acc += vk * vk;
        }
        i += block_dim::<DimX>();
    }

    // Intra-warp reduction.
    let w_acc = warp.redux(ReduxAdd, acc);

    // Lane 0 of each warp publishes its partial to shared memory.
    {
        let mut smem_chunk = smem_sumsq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_chunk[0] = w_acc;
        }
    }
    sync_threads();

    // Cross-warp reduce: every warp loads all warp-partials from smem
    // (lanes 0..num_warps), pads with 0, and does another warp redux.
    // After this redux, all 32 lanes of every warp hold the block sum —
    // matches the `.cu`'s cross-warp shuffle + broadcast (no extra smem write).
    let smem_val = if lane_id < num_warps {
        smem_sumsq[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sumsq = warp.redux(ReduxAdd, smem_val);

    let scale = (block_sumsq + eps).rsqrt();

    // ---- Pass 2: strided Float4 writes of x * scale ----
    //
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D4
    //   t0 ∈ [0, BLOCK)          — thread within block
    //   i0 ∈ [0, D4 / BLOCK)     — slot index per thread
    //   t1 ∈ [0, grid_dim)       — block (== row) index
    let out_map = reshape_map!(
        [D4 / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot: u32 = 0;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let mut r = Float4::new([0.0; 4]);
        for k in 0..4 {
            r[k] = v[k] * scale;
        }
        y_chunk[slot] = r;
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
    assert_eq!(shape.len(), 2, "l2_norm_fc: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(
        d % 4 == 0,
        "D ({d}) must be divisible by 4 for Float4 loads"
    );
    let d4 = d / 4;
    assert!(
        d4 % BLOCK as usize == 0,
        "D/4 ({d4}) must be divisible by BLOCK ({BLOCK}) for reshape_map output"
    );
    let n_total = b * d;

    let h_x = super::super::read_bin(&in_dir.join("x.bin"), n_total);
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); h_x4.len()];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let bb: u32 = b as u32;
    let dd4: u32 = d4 as u32;
    let eps: f32 = 1e-12;

    let bs: u32 = BLOCK;
    let gs: u32 = bb; // one block per row

    // Warm up once before timing.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd4, eps).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd4, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd4, eps).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before reading / persisting.
    d_y4.copy_to_host(h_y4.as_mut_slice()).unwrap();
    drop(d_x4);

    let h_y_flat: &[f32] = h_y4.as_slice().flatten();
    super::super::write_bin(&out_dir.join("y.bin"), h_y_flat);

    (kernel_us, warmup_us)
}
