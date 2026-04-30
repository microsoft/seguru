//! Row-wise LayerNorm over the last dimension of a 2-D tensor [B, D],
//! with no affine parameters (gamma/beta). Float4-vectorized I/O.
//!
//! PyTorch reference:
//!   y = F.layer_norm(x, (D,), eps=1e-5)
//!     = (x - mean(x, dim=-1, keepdim=True))
//!         * rsqrt(var(x, dim=-1, keepdim=True, unbiased=False) + eps)
//!
//! Strategy (per skill doc "Row-Reduction Strategy" + LayerNorm case study):
//!   - Shape [4096, 8192], D % 4 == 0 → use Float4 loads AND stores.
//!   - One block per row; BLOCK = 256 threads (8 warps). D4 = D/4 = 2048.
//!   - Pass 1: fused sum + sumsq over Float4 elements in a single sweep:
//!         s  += v.x + v.y + v.z + v.w
//!         sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w
//!     Block-wide reduce via warp.redux → smem slot/warp → sync → warp.redux
//!     to broadcast block_sum / block_sumsq to every lane. Compute mean and
//!     rstd once, in fp32 registers.
//!   - Pass 2: re-read x_row (hot in L1/L2) as Float4, write normalized
//!     Float4 output via reshape_map! ([t0, i0, t1], D4 units) so the store
//!     bounds check is proved away. Requires D4 % BLOCK == 0 (8192/4 = 2048,
//!     2048 % 256 == 0 ✓).
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: u32 for GPU-side indices / sizes.
//!   - Golden Rule #2: subslice `x_row` once to amortize bounds checks.
//!   - "Always vectorize when D % 4 == 0 (Float4 loads)" recipe
//!     (both loads AND stores vectorized; closes ~1.23× gap to raw CUDA).
//!   - Warp reductions: `ThreadWarpTile::<32>` + `ReduxAdd`.
//!   - Cross-warp reduction via `GpuShared::<[f32; 32]>` + `chunk_to_scope`.
//!   - reshape_map! [t0, i0, t1] output layout eliminates store bounds check.
//!   - Accumulate/normalize in fp32; hoist `mean`/`rstd` out of inner loop.
//!   - Golden Rule #7 (host): copy_to_host before write_bin.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;

/// One block per row; 8 warps per block (BLOCK = 256). Float4-vectorized.
/// Fused single-kernel LayerNorm with no affine.
#[gpu::cuda_kernel]
pub fn layer_norm_kernel(x: &[Float4], y: &mut [Float4], D4: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One slot per warp (max 32 warps for BLOCK=1024). Unused slots stay 0.0
    // (identity for ReduxAdd) so the second warp redux is safe.
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();
    let mut smem_sq = GpuShared::<[f32; 32]>::zero();

    // Subslice this block's Float4 row once.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // --- Pass 1: fused sum + sumsq via Float4 strided traversal ---
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        s += v[0] + v[1] + v[2] + v[3];
        sq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        i += block_dim::<DimX>();
    }

    // --- Block-wide sum reduce ---
    let warp_sum = warp.redux(ReduxAdd, s);
    {
        let mut sl = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = warp_sum;
        }
    }
    sync_threads();
    let sv_sum = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, sv_sum);

    // --- Block-wide sumsq reduce ---
    let warp_sq = warp.redux(ReduxAdd, sq);
    {
        let mut sl = smem_sq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = warp_sq;
        }
    }
    sync_threads();
    let sv_sq = if lane_id < num_warps {
        smem_sq[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sumsq = warp.redux(ReduxAdd, sv_sq);

    // Mean & rstd in fp32 registers — hoisted out of the write loop.
    // D = D4 * 4 in fp32 space.
    let inv_d = 1.0f32 / ((D4 as f32) * 4.0f32);
    let mean = block_sum * inv_d;
    let var = block_sumsq * inv_d - mean * mean;
    let rstd = (var + eps).rsqrt();

    // --- Pass 2: write normalized output as Float4 ---
    // reshape_map output layout [t0, i0, t1] over D4-unit indices:
    //   t0 ∈ [0, BLOCK)           — thread within block
    //   i0 ∈ [0, D4/BLOCK)        — per-thread slot index
    //   t1 ∈ [0, grid_dim)        — block (row) index
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
        out[0] = (v[0] - mean) * rstd;
        out[1] = (v[1] - mean) * rstd;
        out[2] = (v[2] - mean) * rstd;
        out[3] = (v[3] - mean) * rstd;
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
    assert_eq!(shape.len(), 2, "layer_norm: shape=[B, D]");
    let (b, d) = (shape[0], shape[1]);

    // BLOCK=256. Float4 path requires D % 4 == 0 and D4 % BLOCK == 0.
    const BLOCK: u32 = 256;
    assert!(
        d % 4 == 0,
        "layer_norm: D must be divisible by 4 for Float4"
    );
    let d4 = d / 4;
    assert!(
        d4 % BLOCK as usize == 0,
        "D4 must be divisible by BLOCK ({BLOCK}) for reshape_map output layout"
    );

    let n_total = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    // Host-side: repack f32 into Float4. (cudaMalloc guarantees 16B alignment.)
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); b * d4];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let d4u = d4 as u32;
    let eps: f32 = 1e-5;

    // 1 block per row.
    let bs: u32 = BLOCK;
    let gs: u32 = b as u32;

    // Untimed warmup launch.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
    }
    ctx.sync().unwrap();

    // Timed warmup.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: copy device → host before persisting.
    d_y4.copy_to_host(&mut h_y4).unwrap();
    drop(d_x4);

    // Unpack Float4 → f32.
    for (i, v) in h_y4.iter().enumerate() {
        h_y[i * 4] = v[0];
        h_y[i * 4 + 1] = v[1];
        h_y[i * 4 + 2] = v[2];
        h_y[i * 4 + 3] = v[3];
    }

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
