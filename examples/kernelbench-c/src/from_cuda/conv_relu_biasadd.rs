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
//! Raw-CUDA parity version: direct 16x16 one-output-per-thread convolution,
//! matching `examples/kernelbench-c/cuda/conv_relu_biasadd.cu`. Edge tiles are
//! guarded with shape parameters rather than hardcoded benchmark dimensions.

use std::path::Path;
use std::time::Instant;

use gpu::chunk::ScopeUniqueMap;
use gpu::chunk_scope::{ChunkScope, TID_MAX_LEN};
use gpu::prelude::*;

const BDIM_X: u32 = 16;
const BDIM_Y: u32 = 16;
const KHW: u32 = 9; // Kh*Kw for 3x3

#[derive(Clone, Copy)]
struct ConvOutputMap {
    wo: u32,
    ho: u32,
}

// Each valid (global x, global y, global z) thread maps to exactly one
// y[z, y, x] element; edge-tile threads are invalidated before storing.
unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for ConvOutputMap {
    type IndexType = u32;
    type GlobalIndexType = u32;

    fn map(&self, idx: Self::IndexType, thread_ids: [u32; TID_MAX_LEN]) -> (bool, u32) {
        let x = CS::global_id_x(thread_ids);
        let y = CS::global_id_y(thread_ids);
        let z = CS::global_id_z(thread_ids);
        let valid = idx == 0 && x < self.wo && y < self.ho;
        (valid, (z * self.ho + y) * self.wo + x)
    }
}

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn conv_relu_biasadd_kernel(
    x: &[f32],
    w: &[f32],
    b1: &[f32],
    b2: &[f32],
    y: &mut [f32],
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
    let _ = B;
    let _ = Kh;
    let _ = Kw;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bx = block_id::<DimX>();
    let by = block_id::<DimY>();
    let bz = block_id::<DimZ>();

    let co = bz % Cout;
    let bi = bz / Cout;

    let ho = by * BDIM_Y + ty;
    let wo = bx * BDIM_X + tx;

    let mut y_thread = chunk_mut(y, ConvOutputMap { wo: Wo, ho: Ho });

    let hw = H * Wd;
    let x_batch_base = bi * Cin * hw;
    let w_chan_base = co * Cin * KHW;

    if wo < Wo && ho < Ho {
        let mut acc = 0.0f32;

        let mut ci: u32 = 0;
        while ci < Cin {
            let x_ci = (x_batch_base + ci * hw + ho * Wd + wo) as usize;
            let w_ci = (w_chan_base + ci * KHW) as usize;
            let wd = Wd as usize;

            let x0 = &x[x_ci..];
            let x1 = &x[(x_ci + wd)..];
            let x2 = &x[(x_ci + wd * 2)..];
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

        let mut v = acc + b1[co as usize];
        if v < 0.0 {
            v = 0.0;
        }
        v += b2[co as usize];
        y_thread[0] = v;
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
        "conv_relu_biasadd: shape=[B, Cin, H, W, Cout, Kh, Kw]"
    );
    let (b, cin, h, wd, cout, kh, kw) = (
        shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6],
    );
    let ho = h - kh + 1;
    let wo = wd - kw + 1;

    assert_eq!(kh, 3, "kernel specialized to Kh=3");
    assert_eq!(kw, 3, "kernel specialized to Kw=3");

    let h_x = crate::read_bin(&in_dir.join("x.bin"), b * cin * h * wd);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), cout * cin * kh * kw);
    let h_b1 = crate::read_bin(&in_dir.join("b.bin"), cout);
    let h_b2 = crate::read_bin(&in_dir.join("bias2.bin"), cout);
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

    let gx: u32 = wou.div_ceil(BDIM_X);
    let gy: u32 = hou.div_ceil(BDIM_Y);
    let gz: u32 = bb * co_u;

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
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
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
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
        let cfg = gpu_host::gpu_config!(gx, gy, gz, BDIM_X, BDIM_Y, 1, 0);
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
