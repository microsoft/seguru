//! 1_Conv2D_ReLU_BiasAdd — fused `conv2d -> relu -> +extra_bias`.
//!
//! PyTorch reference:
//!     y = F.conv2d(x, W, conv_bias)        // 3x3, stride 1, no padding
//!     y = F.relu(y)
//!     y = y + extra_bias                   // extra_bias broadcast per-channel
//!
//! Shapes (from `python/compare.py::_conv_relu_biasadd`):
//!     x       : [B=128, Cin=64, H=128, W=128]
//!     W       : [Cout=128, Cin=64, Kh=3, Kw=3]
//!     b       : [Cout]
//!     bias2   : [Cout, 1, 1]  (stored as Cout f32s)
//!     y       : [B, Cout, Ho=126, Wo=126]
//!
//! Kernel design (direct convolution, one output per thread):
//!   * Output has `total = B*Cout*Ho*Wo = 128*128*126*126 = 260_112_384`
//!     elements. That number is exactly divisible by our block size (256),
//!     so a 1-D launch `grid = total/256`, `block = 256` produces exactly one
//!     thread per output with no tail-masking and — crucially —
//!     `chunk_mut(y, MapContinuousLinear::new(1))` fully partitions `y`
//!     without bounds guards on the store (skill-doc "Memory Write Patterns /
//!     1D output — one element per thread").
//!   * Inner body: decompose the flat thread id back into (bi, co, ho, wo),
//!     then stream Cin*Kh*Kw = 576 FMAs per output from global memory with
//!     `while` loops (no iterators — keeps u32 arithmetic per Golden Rule #1).
//!   * Epilogue is fused: `relu(acc + b[co]) + bias2[co]`. The ReLU sits
//!     between the two biases, so we cannot precombine them.
//!
//! This is a deliberately simple, "globals-only" direct-conv. It's plenty for
//! the ~150 GFLOPs of work here and avoids the smem-tile-plus-broadcast-read
//! pattern which, given SeGuRu's compute-phase bounds checks (skill-doc
//! "Honest limitation on COMPUTE phase"), would not obviously win without
//! register tiling.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BS: u32 = 256;

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
pub fn conv_relu_biasadd_kernel(
    x: &[f32],
    w: &[f32],
    b1: &[f32],    // conv bias, [Cout]
    b2: &[f32],    // extra bias, [Cout] (view of [Cout, 1, 1])
    y: &mut [f32], // [B, Cout, Ho, Wo]
    B: u32,
    Cin: u32,
    H: u32,
    Wd: u32,
    Cout: u32,
    Kh: u32,
    Kw: u32,
    Ho: u32,
    Wo: u32,
) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let tid = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();

    let total = B * Cout * Ho * Wo;
    if tid >= total {
        // `total` is divisible by BS by construction (see host), but keep this
        // guard for future shape re-use.
        return;
    }

    // Decompose flat tid into (bi, co, ho, wo) for layout [B, Cout, Ho, Wo].
    let wo = tid % Wo;
    let t1 = tid / Wo;
    let ho = t1 % Ho;
    let t2 = t1 / Ho;
    let co = t2 % Cout;
    let bi = t2 / Cout;

    // Base offsets into x and w for this (bi, co).
    let x_batch_base = bi * Cin * H * Wd;
    let w_chan_base = co * Cin * Kh * Kw;
    let ci_xstride = H * Wd;
    let ci_wstride = Kh * Kw;

    let mut acc = 0.0f32;

    let mut ci: u32 = 0;
    while ci < Cin {
        let x_ci = x_batch_base + ci * ci_xstride;
        let w_ci = w_chan_base + ci * ci_wstride;
        let mut kh: u32 = 0;
        while kh < Kh {
            let x_row = x_ci + (ho + kh) * Wd + wo;
            let w_row = w_ci + kh * Kw;
            let mut kw: u32 = 0;
            while kw < Kw {
                acc += x[(x_row + kw) as usize] * w[(w_row + kw) as usize];
                kw += 1;
            }
            kh += 1;
        }
        ci += 1;
    }

    // Fused epilogue: conv_bias + ReLU + extra_bias.
    let mut v = acc + b1[co as usize];
    if v < 0.0 {
        v = 0.0;
    }
    v += b2[co as usize];

    y_chunk[0] = v;
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
        "conv_relu_biasadd: shape=[B, Cin, H, W, Cout, Kh, Kw]"
    );
    let (b, cin, h, wd, cout, kh, kw) = (
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
    );
    let ho = h - kh + 1;
    let wo = wd - kw + 1;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), b * cin * h * wd);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b1 = crate::read_bin(&in_dir.join("b.bin"), cout);
    let h_b2 = crate::read_bin(&in_dir.join("bias2.bin"), cout); // [Cout,1,1]
    let mut h_y = vec![0f32; b * cout * ho * wo];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b1 = ctx.new_tensor_view(h_b1.as_slice()).unwrap();
    let d_b2 = ctx.new_tensor_view(h_b2.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let cinn = cin as u32;
    let hh = h as u32;
    let ww = wd as u32;
    let co_u = cout as u32;
    let khu = kh as u32;
    let kwu = kw as u32;
    let hou = ho as u32;
    let wou = wo as u32;

    let total = bb * co_u * hou * wou;
    let gs: u32 = total.div_ceil(BS);

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BS, 1, 1, 0);
        conv_relu_biasadd_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b1, &d_b2, &mut d_y, bb, cinn, hh, ww, co_u, khu, kwu,
            hou, wou,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    // Warmup (timed for reporting).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BS, 1, 1, 0);
        conv_relu_biasadd_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b1, &d_b2, &mut d_y, bb, cinn, hh, ww, co_u, khu, kwu,
            hou, wou,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BS, 1, 1, 0);
        conv_relu_biasadd_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b1, &d_b2, &mut d_y, bb, cinn, hh, ww, co_u, khu, kwu,
            hou, wou,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: readback is NOT automatic.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b2);
    drop(d_b1);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
