//! 57_Conv2d_ReLU_HardSwish — SeGuRu-from-CUDA refresh of
//! `examples/kernelbench-c/cuda/conv_relu_hardswish.cu`.
//!
//! PyTorch reference:
//!     y = conv2d(x, W, b); y = relu(y); y = y * clamp((y+3)/6, 0, 1)
//!
//! Shapes (from `python/compare.py::_conv_relu_hardswish`):
//!     x: [B=128, Cin=8, H=128, W=128]
//!     W: [Cout=64, Cin=8, 3, 3]
//!     b: [Cout=64]
//!     y: [B, Cout, Ho=126, Wo=126]
//!
//! Raw-CUDA parity version: direct 16x16 one-output-per-thread convolution,
//! matching the CUDA baseline's flattened `[B*Cout*Ho, Wo]` output geometry.
//! The earlier shared-memory 14x14 patch tile launched more blocks, left 60
//! threads per block idle for stores, and added barriers; for Cin=8 the direct
//! row-sliced 3x3 body is the faster and simpler port.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BDIM_X: u32 = 16;
const BDIM_Y: u32 = 16;
const KHW: u32 = 9;
const KSZ: u32 = 3;

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
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
    let mut y_thread = chunk_mut(y, Map2D::new(Wo as usize));

    let wo = block_id::<DimX>() * BDIM_X + thread_id::<DimX>();
    let row = block_id::<DimY>() * BDIM_Y + thread_id::<DimY>();
    let total_rows = Bsz * Cout * Ho;

    if wo < Wo && row < total_rows {
        let bco = row / Ho;
        let ho = row - bco * Ho;
        let bi = bco / Cout;
        let co = bco - bi * Cout;

        let hw = H * Wi;
        let x_batch_base = bi * Cin * hw;
        let w_chan_base = co * Cin * KHW;
        let wi = Wi as usize;

        let mut acc = bias[co as usize];
        let mut ci: u32 = 0;
        while ci < Cin {
            let x_ci = (x_batch_base + ci * hw + ho * Wi + wo) as usize;
            let w_ci = (w_chan_base + ci * KHW) as usize;

            let x0 = &x[x_ci..];
            let x1 = &x[(x_ci + wi)..];
            let x2 = &x[(x_ci + wi * 2)..];
            let wc = &w[w_ci..];

            acc += x0[0] * wc[0];
            acc += x0[1] * wc[1];
            acc += x0[2] * wc[2];
            acc += x1[0] * wc[3];
            acc += x1[1] * wc[4];
            acc += x1[2] * wc[5];
            acc += x2[0] * wc[6];
            acc += x2[1] * wc[7];
            acc += x2[2] * wc[8];

            ci += 1;
        }

        let r = if acc > 0.0 { acc } else { 0.0 };
        let mut hs = (r + 3.0) * (1.0 / 6.0);
        if hs < 0.0 {
            hs = 0.0;
        }
        if hs > 1.0 {
            hs = 1.0;
        }
        y_thread[(0, 0)] = r * hs;
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
    assert_eq!(kh, KSZ as usize, "kernel_size hardcoded to 3");
    assert_eq!(kw, KSZ as usize, "kernel_size hardcoded to 3");

    let ho = hh - kh + 1;
    let wo = ww - kw + 1;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), bsz * cin * hh * ww);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b = crate::read_bin(&in_dir.join("b.bin"), cout);
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

    let gx: u32 = u_wo.div_ceil(BDIM_X);
    let gy: u32 = (u_bsz * u_cout * u_ho).div_ceil(BDIM_Y);

    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        conv_relu_hardswish_fc_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, u_bsz, u_cin, u_h, u_w, u_cout, u_ho, u_wo,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
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

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
