//! 1_Conv2D_ReLU_BiasAdd — SeGuRu port of `cuda/conv_relu_biasadd.cu`.
//!
//! The reference `.cu` uses a 2-D output tile (TH=TW=16) with blockIdx.z as
//! (batch, out_channel). Because Wo=126 and Ho=126 are NOT multiples of 16,
//! the CUDA version guards every thread with `if (wo >= Wo || ho >= Ho)
//! return;` and stores directly into global memory. That pattern does not
//! map cleanly onto `chunk_mut` (the chunk map expects a contiguous partition
//! of the output; the 8×16 = 128 padded width disagrees with the real Wo=126,
//! so a `Map2D::new(Wo)` chunk would need a non-trivial `reshape_map` and
//! still lose cover for the tail threads).
//!
//! Rather than contort the chunk map, this translation collapses the CUDA
//! kernel's 3-D launch into a 1-D flat-thread-per-output launch. The inner
//! compute is an exact mirror of the CUDA body (same Cin/Kh/Kw accumulation
//! order, same fused `relu(acc + b[co]) + bias2[co]` epilogue). Because
//! `total = B*Cout*Ho*Wo = 260_112_384` is exactly divisible by BS=256, the
//! launch is dense and `chunk_mut(y, MapContinuousLinear::new(1))` partitions
//! the output perfectly without per-thread tail guards (skill-doc "1D output
//! — one element per thread").
//!
//! The direct-conv choice (no shared-memory tile, no register tile) matches
//! the `.cu` and keeps Stage-2 translation mechanical: the only semantic
//! change is grid geometry, which the skill doc calls out as the typical
//! Stage-2 degree of freedom.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BS: u32 = 256;

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
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
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let tid = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();

    let total = B * Cout * Ho * Wo;
    if tid >= total {
        return;
    }

    // Flat tid -> (bi, co, ho, wo) for layout [B, Cout, Ho, Wo].
    let wo = tid % Wo;
    let t1 = tid / Wo;
    let ho = t1 % Ho;
    let t2 = t1 / Ho;
    let co = t2 % Cout;
    let bi = t2 / Cout;

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

    // Epilogue (exact mirror of the .cu): conv_bias + ReLU + extra_bias.
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

    let total = bb * co_u * hou * wou;
    let gs: u32 = total.div_ceil(BS);

    // Prime.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BS, 1, 1, 0);
        conv_relu_biasadd_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b1, &d_b2, &mut d_y, bb, cinn, hh, ww, co_u, khu, kwu,
            hou, wou,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    // Warmup.
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

    // Timed.
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

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b2);
    drop(d_b1);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
