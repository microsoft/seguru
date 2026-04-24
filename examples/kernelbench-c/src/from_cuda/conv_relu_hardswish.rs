//! 57_Conv2d_ReLU_HardSwish — SeGuRu port translated from
//! `examples/kernelbench-c/cuda/conv_relu_hardswish.cu`.
//!
//! PyTorch reference:
//!     y = conv2d(x, W, b); y = relu(y); y = y * clamp((y+3)/6, 0, 1)
//!
//! Shapes (pilot):
//!     x: [B=128, Cin=8, H=128, W=128]
//!     W: [Cout=64, Cin=8, 3, 3]
//!     b: [Cout=64]
//!     y: [B, Cout, Ho=126, Wo=126]
//!
//! v2: shared-memory input patch tiling per the skill-doc "Convolution
//! Recipe" / "Shared Memory Tiling" sections.  v1 (one-thread-per-output,
//! all Cin reads from gmem) followed the raw CUDA pattern verbatim, but
//! SeGuRu's safety layer adds a per-access bounds check that's amortized
//! by moving the Cin·Kh·Kw accumulation loop over a shared patch.
//!
//! Recipe parameters (Cin=8, Kh=Kw=3, Ho=Wo=126):
//!   * TILE_OUT = 14 — output pixels per block side.  14*9 = 126 = Ho = Wo,
//!     so a 9x9 grid per (bi, co) cleanly partitions the output with no
//!     mid-tile masking.
//!   * PATCH = TILE_OUT + Kh - 1 = 16 — input patch side per block.
//!   * BDIM_X = BDIM_Y = 16 — 256 threads/block (8 warps).  Only the inner
//!     14×14 = 196 threads write output; the outer 60 threads participate
//!     only in the cooperative smem load.
//!   * CIN_CHUNK = 4 — channel slab per smem refill.  Patch smem =
//!     [CIN_CHUNK, PATCH, PATCH] = 4*16*16 = 1024 f32 = 4 KB/block.
//!     Cin=8 / CIN_CHUNK=4 = 2 refills per output.
//!   * Load layout: 256 threads × 4 slots = 1024 = PATCH_VOL exactly, so
//!     `reshape_map!([4] | [16, 16] => layout: [i0, t0, t1])` gives each
//!     thread 4 adjacent smem slots.  Per-thread 4 slots are 4 consecutive
//!     columns in the same patch row, so the gmem reads are 4 adjacent
//!     f32s — fully coalesced within a warp.
//!   * The input patch is always fully inside x's [H, W] region (for
//!     by ∈ 0..9: h0 = by*14 ∈ {0,14,..,112}, py ∈ 0..16 → g_h ≤ 127 < 128),
//!     so the load is branchless.
//!
//! Epilogue: conv_bias + ReLU + HardSwish fused into the final store.
//!
//! Skill-doc patterns applied:
//!   - `u32` indexing end-to-end (Golden Rule #1).
//!   - Shared-mem patch tile + `reshape_map!([4] | [16,16])` disjoint load
//!     (bounds-check-free stores).
//!   - Disjoint 6-axis output `reshape_map` via `(BDIM_X, TILE_OUT)` clamp
//!     so the outer 60 threads' stores are type-proven no-ops.
//!   - `unroll!` on the kh/kw inner loops.

use std::path::Path;
use std::time::Instant;

use crunchy::unroll;
use gpu::prelude::*;

const TILE_OUT: u32 = 14;
const PATCH: u32 = 16;
const BDIM_X: u32 = 16;
const BDIM_Y: u32 = 16;
const CIN_CHUNK: u32 = 4;
const PATCH_AREA: u32 = PATCH * PATCH; // 256
const PATCH_VOL: u32 = PATCH_AREA * CIN_CHUNK; // 1024
const KHW: u32 = 9; // Kh*Kw for 3x3

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn conv_relu_hardswish_fc_kernel(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    Bsz: u32,
    Cin: u32,
    H: u32,
    Wi: u32,
    Cout: u32,
    Ho: u32,
    Wo: u32,
) {
    let _ = Bsz;
    let _ = Ho;
    let _ = Wo;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bx = block_id::<DimX>();
    let by = block_id::<DimY>();
    let bz = block_id::<DimZ>();

    let co = bz % Cout;
    let bi = bz / Cout;

    let h0 = by * TILE_OUT;
    let w0 = bx * TILE_OUT;

    let x_batch_base = bi * Cin * H * Wi;
    let w_chan_base = co * Cin * KHW;

    // Shared-memory input patch: flat [CIN_CHUNK, PATCH, PATCH].
    let mut patch = gpu::GpuShared::<[f32; PATCH_VOL as usize]>::zero();

    // Load map: 256 threads × 4 slots = 1024 patch elements, disjoint.
    let load_map = reshape_map!([4] | [16, 16] => layout: [i0, t0, t1]);

    // Output map: [TILE_OUT, gx, TILE_OUT, gy, B*Cout] = [126, 126, B*Cout].
    // Threads with tx >= TILE_OUT or ty >= TILE_OUT have no valid slot; the
    // `y_thread[0] = ...` store for them is suppressed by the chunk
    // precondition (skill-doc Example 6 semantics).
    let out_map = reshape_map!(
        [1] | [(BDIM_X, TILE_OUT), grid_dim::<DimX>(), (BDIM_Y, TILE_OUT), grid_dim::<DimY>(), grid_dim::<DimZ>()]
        => layout: [i0, t0, t1, t2, t3, t4]
    );
    let mut y_thread = chunk_mut(y, out_map);

    // Per-thread derived load indices (all 4 slots share ci_off + py + patch
    // row, and land on 4 adjacent px values).
    let tid = ty * BDIM_X + tx;
    let slot_base = tid * 4;
    let ci_off_load = slot_base / PATCH_AREA;
    let rem = slot_base - ci_off_load * PATCH_AREA;
    let py = rem / PATCH;
    let px_base = rem - py * PATCH;
    let patch_row_h = h0 + py;
    let patch_row_w = w0 + px_base;

    let hw = H * Wi;
    let mut acc = 0.0f32;

    let mut ci_base: u32 = 0;
    while ci_base < Cin {
        // ---- Cooperative load: 4 adjacent f32s per thread.
        {
            let mut cp = patch.chunk_mut(load_map);
            let src_base =
                x_batch_base + (ci_base + ci_off_load) * hw + patch_row_h * Wi + patch_row_w;
            cp[0] = x[src_base as usize];
            cp[1] = x[(src_base + 1) as usize];
            cp[2] = x[(src_base + 2) as usize];
            cp[3] = x[(src_base + 3) as usize];
        }
        sync_threads();

        // ---- Compute: 36 FMAs per thread per refill.  Outer 60 threads of
        // each block accumulate garbage but their store is suppressed.
        let mut ci_off: u32 = 0;
        while ci_off < CIN_CHUNK {
            let patch_ci = ci_off * PATCH_AREA;
            let w_base = w_chan_base + (ci_base + ci_off) * KHW;
            unroll! { for kh in 0..3 {
                let row_off = patch_ci + (ty + kh as u32) * PATCH + tx;
                let w_row = w_base + (kh as u32) * 3;
                unroll! { for kw in 0..3 {
                    let p = patch[(row_off + kw as u32) as usize];
                    let wv = w[(w_row + kw as u32) as usize];
                    acc += p * wv;
                }}
            }}
            ci_off += 1;
        }
        sync_threads();
        ci_base += CIN_CHUNK;
    }

    // ---- Epilogue: conv_bias + ReLU + HardSwish.
    if tx < TILE_OUT && ty < TILE_OUT {
        let v_pre = acc + bias[co as usize];
        let r = if v_pre > 0.0 { v_pre } else { 0.0 };
        let mut hs = (r + 3.0) * (1.0 / 6.0);
        if hs < 0.0 {
            hs = 0.0;
        }
        if hs > 1.0 {
            hs = 1.0;
        }
        y_thread[0] = r * hs;
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
    assert_eq!(
        shape.len(),
        7,
        "conv_relu_hardswish: shape=[B, Cin, H, W, Cout, Kh, Kw]"
    );
    let (bsz, cin, hh, ww, cout, kh, kw) = (
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
    );
    assert_eq!(kh, 3, "kernel_size hardcoded to 3");
    assert_eq!(kw, 3, "kernel_size hardcoded to 3");

    let ho = hh - 2;
    let wo = ww - 2;

    assert_eq!(
        ho % TILE_OUT as usize,
        0,
        "Ho={} must be a multiple of TILE_OUT={}",
        ho,
        TILE_OUT
    );
    assert_eq!(
        wo % TILE_OUT as usize,
        0,
        "Wo={} must be a multiple of TILE_OUT={}",
        wo,
        TILE_OUT
    );
    assert_eq!(
        cin % CIN_CHUNK as usize,
        0,
        "Cin={} must be a multiple of CIN_CHUNK={}",
        cin,
        CIN_CHUNK
    );

    let h_x = super::super::read_bin(&in_dir.join("x.bin"), bsz * cin * hh * ww);
    let h_w = super::super::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b = super::super::read_bin(&in_dir.join("b.bin"), cout);
    let mut h_y = vec![0f32; bsz * cout * ho * wo];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let u_bsz = bsz as u32;
    let u_cin = cin as u32;
    let u_h = hh as u32;
    let u_w = ww as u32;
    let u_cout = cout as u32;
    let u_ho = ho as u32;
    let u_wo = wo as u32;

    let gx: u32 = u_wo / TILE_OUT;
    let gy: u32 = u_ho / TILE_OUT;
    let gz: u32 = u_bsz * u_cout;

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b);
    drop(d_w);
    drop(d_x);

    super::super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
