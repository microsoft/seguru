//! Port of cuda/pointwise_conv2d.cu: pointwise 1x1 Conv2d over [B,Cin,H,W].
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
pub fn pointwise_conv2d_kernel(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    B: u32,
    Cin: u32,
    H: u32,
    Wd: u32,
    Cout: u32,
    Total: u32,
) {
    let _ = B;
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let total = Total;
    if idx < total {
        let wi = idx % Wd;
        let mut t = idx / Wd;
        let h = t % H;
        t /= H;
        let co = t % Cout;
        let b = t / Cout;

        let mut acc = 0.0f32;
        let mut ci = 0u32;
        while ci < Cin {
            let x_idx = ((b * Cin + ci) * H + h) * Wd + wi;
            let w_idx = co * Cin + ci;
            acc += x[x_idx as usize] * w[w_idx as usize];
            ci += 1;
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
    assert_eq!(shape.len(), 5, "pointwise_conv2d: shape=[B,Cin,H,W,Cout]");
    let (b, cin, h, wd, cout) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    assert_eq!(cin, 64, "pointwise_conv2d benchmark keeps Cin=64");
    assert_eq!(cout, 128, "pointwise_conv2d benchmark keeps Cout=128");
    let x_len = checked_len("pointwise_conv2d input elements", &[b, cin, h, wd]);
    let w_len = checked_len("pointwise_conv2d weight elements", &[cout, cin]);
    let y_len = checked_len("pointwise_conv2d output elements", &[b, cout, h, wd]);
    checked_u32("pointwise_conv2d input elements", x_len);
    checked_u32("pointwise_conv2d weight elements", w_len);
    let total = checked_u32("pointwise_conv2d output elements", y_len);

    let h_x = crate::read_bin(&in_dir.join("x.bin"), x_len);
    let h_w = crate::read_bin(&in_dir.join("w.bin"), w_len);
    let mut h_y = vec![0f32; y_len];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = checked_u32("pointwise_conv2d B", b);
    let ci = checked_u32("pointwise_conv2d Cin", cin);
    let hh = checked_u32("pointwise_conv2d H", h);
    let ww = checked_u32("pointwise_conv2d W", wd);
    let co = checked_u32("pointwise_conv2d Cout", cout);
    let grid = total.div_ceil(BLOCK);

    {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        pointwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        pointwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(grid, 1, 1, BLOCK, 1, 1, 0);
        pointwise_conv2d_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &mut d_y, bb, ci, hh, ww, co, total,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
