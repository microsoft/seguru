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
//! Strategy (mirrors the .cu): one thread per output element.  Output is
//! treated as a 2-D tensor of shape `[B*Cout*Ho, Wo]` so we can use `Map2D`
//! with a 2-D grid (z-dim = 1, as the map requires).  ReLU + HardSwish are
//! fused into the final store — no intermediate is materialized.
//!
//! Skill-doc patterns:
//!   - `u32` indexing end-to-end (Golden Rule #1).
//!   - Manual bounds guard `if wo < Wo && row < total_rows { ... }` (#5).
//!   - Output via `chunk_mut(y, Map2D::new(Wo))` with `y_chunk[(0,0)] = v`.
//!   - Kernel size constants (`KSZ=3`) kept `const` so the inner loop unrolls
//!     to 9 FMAs per Cin slice.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BLK_X: u32 = 16;
const BLK_Y: u32 = 16;
const KSZ: u32 = 3;

#[gpu::cuda_kernel]
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
    let mut y_chunk = chunk_mut(y, Map2D::new(Wo as usize));

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let wo = block_id::<DimX>() * BLK_X + tx;
    let row = block_id::<DimY>() * BLK_Y + ty; // = bco*Ho + ho

    let total_rows = Bsz * Cout * Ho;

    if wo < Wo && row < total_rows {
        let bco = row / Ho;
        let ho = row - bco * Ho;
        let bi = bco / Cout;
        let co = bco - bi * Cout;

        let mut acc = bias[co as usize];

        let mut ci: u32 = 0;
        while ci < Cin {
            let x_ci_base = ((bi * Cin + ci) * H) * Wi;
            let w_ci_base = (co * Cin + ci) * (KSZ * KSZ);

            // kh = 0
            let xr0 = (x_ci_base + (ho + 0) * Wi + wo) as usize;
            let wr0 = (w_ci_base + 0 * KSZ) as usize;
            acc += x[xr0] * w[wr0];
            acc += x[xr0 + 1] * w[wr0 + 1];
            acc += x[xr0 + 2] * w[wr0 + 2];

            // kh = 1
            let xr1 = (x_ci_base + (ho + 1) * Wi + wo) as usize;
            let wr1 = (w_ci_base + 1 * KSZ) as usize;
            acc += x[xr1] * w[wr1];
            acc += x[xr1 + 1] * w[wr1 + 1];
            acc += x[xr1 + 2] * w[wr1 + 2];

            // kh = 2
            let xr2 = (x_ci_base + (ho + 2) * Wi + wo) as usize;
            let wr2 = (w_ci_base + 2 * KSZ) as usize;
            acc += x[xr2] * w[wr2];
            acc += x[xr2 + 1] * w[wr2 + 1];
            acc += x[xr2 + 2] * w[wr2 + 2];

            ci += 1;
        }

        // ReLU
        let r = if acc > 0.0 { acc } else { 0.0 };
        // HardSwish on the ReLU output: r * clamp((r+3)/6, 0, 1).
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

    let gx: u32 = u_wo.div_ceil(BLK_X);
    let gy: u32 = (u_bsz * u_cout * u_ho).div_ceil(BLK_Y);

    // Priming launch (not counted).
    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BLK_X, BLK_Y, 1, 0);
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
