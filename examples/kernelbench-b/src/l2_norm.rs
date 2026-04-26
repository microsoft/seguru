//! Row-wise L2 normalization over the last dimension of a 2-D tensor [B, D].
//!
//! PyTorch reference:
//!   y = x / sqrt(sum(x*x, dim=-1, keepdim=True) + 1e-12)
//!
//! Strategy (cuda-to-seguru-porting-skill.md):
//!   - "Row-Reduction Strategy" → 1 block per row, BLOCK=256 (8 warps), grid=B.
//!     B=4096 rows × D=8192 cols: 4096 blocks saturate the SMs; one block per
//!     row keeps the reduction intra-block (no split-row combine).
//!   - "Always vectorize when D % 4 == 0 (Float4 loads)" → kernel takes
//!     `x: &[Float4]` and writes `y: &mut [Float4]` with D4 = D/4 = 2048.
//!     Cuts load/store instruction count 4× vs scalar.
//!   - "Softmax Recipe" shape (single-kernel fused reduce + apply, one scalar
//!     per block held in registers): Pass 1 Float4-strided sum-of-squares,
//!     block-wide reduce via warp.redux + 32-slot smem + second warp.redux,
//!     hoist `scale = rsqrt(sumsq + eps)` outside the write loop. Pass 2
//!     re-reads x_row as Float4 (hot in L1/L2) and writes Float4 output.
//!   - "one scalar per block" pitfall does NOT apply here: our output is
//!     full [B, D] (per-element), not [B]. We use `reshape_map!` for the
//!     strided Float4 output (matches the layernorm vectorized template).
//!
//! Skill-doc patterns used:
//!   - Golden Rule #1: u32 for all GPU-side indices / sizes.
//!   - Golden Rule #2: subslice the row once (`x_row`) to amortize bounds checks.
//!   - Golden Rule #3 / #6: reshape_map + chunk_mut with LOCAL indices.
//!   - Warp reductions: `ThreadWarpTile::<32>::redux(ReduxAdd, …)`.
//!   - Cross-warp reduction via `GpuShared::<[f32; 32]>` + chunk_to_scope.
//!   - Golden Rule #7: copy_to_host before write_bin.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;

/// One block per row; BLOCK=256 threads (8 warps). `D4 = D/4`.
#[gpu::cuda_kernel]
pub fn l2_norm_kernel(x: &[Float4], y: &mut [Float4], D4: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // Reserve 32 slots (max warps for BLOCK≤1024). Tail padded with 0.0.
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();

    // Subslice this block's Float4 row once.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // --- Pass 1: Float4-strided sum of squares ---
    let mut local_sumsq = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        local_sumsq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        i += block_dim::<DimX>();
    }

    // --- Block-wide sum reduce: warp → smem → warp ---
    let warp_sumsq = warp.redux(ReduxAdd, local_sumsq);
    {
        let mut s = smem_sumsq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_sumsq;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_sumsq[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sumsq = warp.redux(ReduxAdd, sv);

    // Hoist the reciprocal-sqrt: one multiply per output element.
    let scale = (block_sumsq + eps).rsqrt();

    // --- Pass 2: Float4 strided output via reshape_map ---
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D4
    //   t0 ∈ [0, BLOCK)       — thread within block
    //   i0 ∈ [0, D4/BLOCK)    — slot index per thread
    //   t1 ∈ [0, grid_dim)    — block (== row) index
    // Requires D4 % BLOCK == 0 (asserted host-side).
    let out_map = reshape_map!(
        [D4 / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let mut out = Float4::new([0.0; 4]);
        out[0] = v[0] * scale;
        out[1] = v[1] * scale;
        out[2] = v[2] * scale;
        out[3] = v[3] * scale;
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
    assert_eq!(shape.len(), 2, "l2_norm: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);

    // BLOCK=256. Float4 loads require D % 4 == 0. reshape_map! output
    // layout additionally requires D4 % BLOCK == 0, i.e. D % (4*BLOCK) == 0.
    const BLOCK: u32 = 256;
    assert!(
        d % 4 == 0,
        "l2_norm: D must be divisible by 4 for Float4 loads"
    );
    let d4 = d / 4;
    assert!(
        d4 % BLOCK as usize == 0,
        "l2_norm: D/4 must be divisible by BLOCK ({BLOCK}) for reshape_map output layout"
    );

    let n_total = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    // Host-side: repack f32 → Float4 (see sum_dim.rs template).
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); h_x4.len()];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let d4_u32 = d4 as u32;
    let eps: f32 = 1e-12;

    // 1 block per row.
    let bs: u32 = BLOCK;
    let gs: u32 = b as u32;

    // Untimed warmup launch.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4_u32, eps).unwrap();
    }
    ctx.sync().unwrap();

    // Timed warmup.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4_u32, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l2_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4_u32, eps).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before persisting.
    d_y4.copy_to_host(&mut h_y4).unwrap();
    drop(d_x4);

    // Unpack Float4 → contiguous f32 (same layout; [B, D] row-major).
    for (i, v) in h_y4.iter().enumerate() {
        let base = i * 4;
        h_y[base] = v[0];
        h_y[base + 1] = v[1];
        h_y[base + 2] = v[2];
        h_y[base + 3] = v[3];
    }

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
