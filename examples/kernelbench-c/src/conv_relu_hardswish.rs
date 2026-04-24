//! 57_Conv2d_ReLU_HardSwish — SeGuRu port of PyTorch reference
//! `examples/kernelbench-c/problems/57_Conv2d_ReLU_HardSwish.py`.
//!
//! Regenerated against `docs/cuda-to-seguru-porting-skill.md` @ bf493b79
//! (Phase D.1). NOT using the Convolution Recipe's smem-patch tile: for
//! Cin=8 each thread does only 72 FMADDs which is below the
//! smem-amortization threshold (see skill-doc "When NOT to tile"). Uses
//! direct 1-thread-per-output with Map2D chunk_mut over the flattened
//! [B*Cout*Ho, Wo] output. ReLU + HardSwish are fused in the final store.
//!
//! 57_Conv2d_ReLU_HardSwish — fused nn.Conv2d + ReLU + HardSwish.
//!
//! PyTorch reference:
//!     y = F.conv2d(x, W, b)                           # [B, Cout, Ho, Wo]
//!     y = F.relu(y)
//!     y = y * torch.clamp((y + 3) / 6, 0, 1)          # HardSwish(y)
//!
//! Shapes (pilot):
//!     x: [B=128, Cin=8,  H=128, W=128]
//!     W: [Cout=64, Cin=8, 3, 3]
//!     b: [Cout=64]
//!     y: [B, Cout, Ho=126, Wo=126]   (no padding, no stride, kernel=3)
//!
//! Strategy: direct convolution, one output element per thread.  For Cin=8,
//! k=3 each thread issues 8·3·3 = 72 FMADDs (a tiny compute slice per thread),
//! so we favour a small block and a large grid rather than register tiling.
//! The output is flattened to a 2-D view of `[B*Cout*Ho, Wo]` so we can drive
//! the write with `Map2D::new(Wo)` and a 2-D grid (Map2D's precondition is
//! `global_dim_z == 1`, skill-doc §"Shared Memory Tiling" / `chunk_impl.rs`).
//! ReLU and HardSwish are fused into the final store — no intermediate
//! tensor is materialised.
//!
//! Skill-doc patterns used:
//!   - `u32` parameters and indexing throughout (Golden Rule #1).
//!   - `chunk_mut(y, Map2D::new(Wo))` with local index `y_chunk[(0, 0)]`
//!     (Golden Rule #6 — LOCAL indices in chunks).
//!   - Bounds via explicit `if wo < Wo && row < total_rows` guard (Rule #5).
//!   - Inner 3×3 loops are left as compile-time-bounded `while` loops so the
//!     codegen can unroll them (KSZ is a `const`).

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BLK_X: u32 = 16;
const BLK_Y: u32 = 16;
const KSZ: u32 = 3;

#[gpu::cuda_kernel]
pub fn conv_relu_hardswish_kernel(
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
    // Each thread owns one output element y[b, co, ho, wo].
    // The "row" axis flattens (b, co, ho); the "col" axis is wo.
    let mut y_chunk = chunk_mut(y, Map2D::new(Wo as usize));

    let wo = block_id::<DimX>() * BLK_X + thread_id::<DimX>();
    let row = block_id::<DimY>() * BLK_Y + thread_id::<DimY>();

    let total_rows = Bsz * Cout * Ho;

    if wo < Wo && row < total_rows {
        let bco = row / Ho;
        let ho = row - bco * Ho;
        let bi = bco / Cout;
        let co = bco - bi * Cout;

        // Σ_{ci, kh, kw} W[co, ci, kh, kw] · x[bi, ci, ho+kh, wo+kw]   + bias[co]
        let mut acc = bias[co as usize];

        let mut ci: u32 = 0;
        while ci < Cin {
            let x_base = ((bi * Cin + ci) * H) * Wi; // offset of x[bi, ci, :, :]
            let w_base = (co * Cin + ci) * (KSZ * KSZ); // offset of W[co, ci, :, :]

            let mut kh: u32 = 0;
            while kh < KSZ {
                let x_row = (x_base + (ho + kh) * Wi + wo) as usize;
                let w_row = (w_base + kh * KSZ) as usize;

                let mut kw: u32 = 0;
                while kw < KSZ {
                    acc += x[x_row + kw as usize] * w[w_row + kw as usize];
                    kw += 1;
                }
                kh += 1;
            }
            ci += 1;
        }

        // Fused epilogue: ReLU then HardSwish(r) = r * clamp((r + 3) / 6, 0, 1).
        let r = if acc > 0.0 { acc } else { 0.0 };
        let mut hs = (r + 3.0) * (1.0 / 6.0);
        if hs < 0.0 {
            hs = 0.0;
        }
        if hs > 1.0 {
            hs = 1.0;
        }
        y_chunk[(0, 0)] = r * hs;
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

    let h_x = super::read_bin(&in_dir.join("x.bin"), bsz * cin * hh * ww);
    let h_w = super::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b = super::read_bin(&in_dir.join("b.bin"), cout);
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

    let gx: u32 = u_wo.div_ceil(BLK_X);
    let gy: u32 = (u_bsz * u_cout * u_ho).div_ceil(BLK_Y);

    // Priming launch — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
        conv_relu_hardswish_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
        conv_relu_hardswish_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
        conv_relu_hardswish_kernel::launch(
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

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
