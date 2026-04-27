//! Depthwise Conv2d over [B,C,H,W] with weights [C,1,Kh,Kw].
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;

fn checked_len(name: &str, dims: &[usize]) -> usize {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .unwrap_or_else(|| panic!("{name} exceeds usize range"))
}

fn checked_u32(name: &str, value: usize) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name} exceeds u32 range: {value}"))
}

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_kernel(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    B: u32,
    C: u32,
    H: u32,
    Wd: u32,
    Kh: u32,
    Kw: u32,
    Ho: u32,
    Wo: u32,
    Total: u32,
) {
    let _ = B;
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let total = Total;
    if idx < total {
        let wo = idx % Wo;
        let mut t = idx / Wo;
        let ho = t % Ho;
        t /= Ho;
        let c = t % C;
        let b = t / C;

        let mut acc = 0.0f32;
        let mut kh = 0u32;
        while kh < Kh {
            let mut kw = 0u32;
            while kw < Kw {
                let x_idx = ((b * C + c) * H + (ho + kh)) * Wd + (wo + kw);
                let w_idx = (c * Kh + kh) * Kw + kw;
                acc += x[x_idx as usize] * w[w_idx as usize];
                kw += 1;
            }
            kh += 1;
        }
        out[0] = acc;
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
    assert_eq!(shape.len(), 6, "depthwise_conv2d: shape=[B,C,H,W,Kh,Kw]");
    let (b, c, h, wd, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    assert_eq!(c, 64, "depthwise_conv2d benchmark keeps C=64");
    assert_eq!(kh, 3, "depthwise_conv2d benchmark uses Kh=3");
    assert_eq!(kw, 3, "depthwise_conv2d benchmark uses Kw=3");
    assert!(h >= kh && wd >= kw, "kernel larger than input");
    let ho = h - kh + 1;
    let wo = wd - kw + 1;
    let x_len = checked_len("depthwise_conv2d input elements", &[b, c, h, wd]);
    let w_len = checked_len("depthwise_conv2d weight elements", &[c, kh, kw]);
    let y_len = checked_len("depthwise_conv2d output elements", &[b, c, ho, wo]);
    checked_u32("depthwise_conv2d input elements", x_len);
    checked_u32("depthwise_conv2d weight elements", w_len);

    let h_x = crate::read_bin(&in_dir.join("x.bin"), x_len);
    let h_w = crate::read_bin(&in_dir.join("w.bin"), w_len);
    let mut h_y = vec![0f32; y_len];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = checked_u32("depthwise_conv2d B", b);
    let cc = checked_u32("depthwise_conv2d C", c);
    let hh = checked_u32("depthwise_conv2d H", h);
    let ww = checked_u32("depthwise_conv2d W", wd);
    let khh = checked_u32("depthwise_conv2d Kh", kh);
    let kww = checked_u32("depthwise_conv2d Kw", kw);
    let hoo = checked_u32("depthwise_conv2d Ho", ho);
    let woo = checked_u32("depthwise_conv2d Wo", wo);
    let total = checked_u32("depthwise_conv2d output elements", y_len);
    let grid = total.div_ceil(BLOCK);

    {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        depthwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, cc, hh, ww, khh, kww, hoo, woo, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        depthwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, cc, hh, ww, khh, kww, hoo, woo, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        depthwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, cc, hh, ww, khh, kww, hoo, woo, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
