//! SeGuRu port of `cuda/layer_norm.cu` (1-for-1 translation of the raw CUDA).
//!
//! CUDA strategy mirrored:
//!   - One block per row (`gs = B`); `BLOCK = 256` threads/block (= 8 warps).
//!   - Pass 1: each thread accumulates `sum` and `sum_sq` over a strided
//!     slice of its row using `float4` (128-bit) vectorized loads, then
//!     warp-reduces via `__shfl_down_sync` (→ `warp.redux(ReduxAdd, …)`).
//!     Lane 0 of each warp writes its `sum`/`sum_sq` partial to shared
//!     memory (`s_sum[WARPS]`, `s_sumsq[WARPS]`).
//!   - After `__syncthreads`, thread 0 scalar-sums the `WARPS` partials,
//!     computes `mean = total/D` and `rstd = rsqrt(var + eps)`, and
//!     broadcasts both via shared memory.
//!   - Pass 2: every thread writes normalized output through `float4`
//!     stores using a `reshape_map!` output chunk.
//!
//! The benchmark shape is `[B, D] = [4096, 8192]` (compare2.py), so
//! `D % 4 == 0` and `D4 % BLOCK == 0`; the scalar tail loop from the `.cu`
//! is unreachable on this shape and is omitted (no other shape is exercised).
//!
//! Skill-doc patterns:
//!   - Vectorized Access (Float4).
//!   - Row-Reduction Strategy: 1 block per row.
//!   - Warp Operations: `ThreadWarpTile::redux(ReduxAdd, …)`.
//!   - Cross-warp combine via `GpuShared`.
//!   - `reshape_map!` + `chunk_mut` for strided Float4 output.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Thread};
use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

/// Threads per block (matches `BLOCK = 256` in `cuda/layer_norm.cu`).
const BLOCK: u32 = 256;

/// Fused layer-norm (no affine): one block per row, Float4 load/store,
/// warp-shuffle + cross-warp shared-memory reduction, thread-0 finalize.
#[gpu::cuda_kernel]
pub fn layer_norm_kernel(x: &[Float4], y: &mut [Float4], D: u32, D4: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let tid = thread_id::<DimX>();
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // Cross-warp scratch: one slot per warp for `sum` and `sum_sq`.
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();
    // 1-element broadcast buffers for `mean` / `rstd` (mirrors `s_mean`, `s_rstd`).
    let mut smem_mean = GpuShared::<[f32; 1]>::zero();
    let mut smem_rstd = GpuShared::<[f32; 1]>::zero();

    // Subslice the row once (Float4-stride).
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // -------------------- Pass 1: Float4 sum + sum_sq --------------------
    let mut local_sum = 0.0f32;
    let mut local_sumsq = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 {
            let vk = v[k];
            local_sum += vk;
            local_sumsq += vk * vk;
        }
        i += block_dim::<DimX>();
    }

    // Warp-level reduce (shuffle).
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    let warp_sumsq = warp.redux(ReduxAdd, local_sumsq);

    // Lane 0 of each warp publishes its partial to shared memory.
    {
        let mut sum_c = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        let mut sq_c = smem_sumsq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane == 0 {
            sum_c[0] = warp_sum;
            sq_c[0] = warp_sumsq;
        }
    }
    sync_threads();

    // Thread 0 finalizes: scalar sum across warps, then mean/rstd.
    // Thread 0 == warp 0, lane 0 — writes slot 0 of each broadcast buffer.
    {
        let mut mean_c = smem_mean
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        let mut rstd_c = smem_rstd
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if tid == 0 {
            let mut total_sum = 0.0f32;
            let mut total_sumsq = 0.0f32;
            let mut w = 0u32;
            while w < num_warps {
                total_sum += smem_sum[w as usize];
                total_sumsq += smem_sumsq[w as usize];
                w += 1;
            }
            let inv_d = 1.0f32 / (D as f32);
            let mean = total_sum * inv_d;
            let var = total_sumsq * inv_d - mean * mean;
            let rstd = (var + eps).rsqrt();
            mean_c[0] = mean;
            rstd_c[0] = rstd;
        }
    }
    sync_threads();

    let mean = smem_mean[0];
    let rstd = smem_rstd[0];

    // -------------------- Pass 2: strided Float4 store -------------------
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D4
    //   t0 ∈ [0, BLOCK)         — thread within block
    //   i0 ∈ [0, D4 / BLOCK)    — slot index per thread
    //   t1 ∈ [0, grid_dim)      — block (== row) index
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
        let mut out = Float4::new([0.0; 4]);
        for k in 0..4 {
            out[k] = (v[k] - mean) * rstd;
        }
        y_chunk[slot] = out;
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
    assert_eq!(shape.len(), 2, "layer_norm_fc: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(d % 4 == 0, "D ({d}) must be divisible by 4 for Float4 loads");
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
    let dd: u32 = d as u32;
    let dd4: u32 = d4 as u32;
    let eps: f32 = 1e-5;

    let bs: u32 = BLOCK;
    let gs: u32 = bb; // one block per row

    // Warm up once before timing.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd, dd4, eps).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup timing.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd, dd4, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, dd, dd4, eps).unwrap();
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
