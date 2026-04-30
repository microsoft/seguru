//! RMS normalization along dim=1 of a 4-D tensor (B, C, H, W).
//!
//! PyTorch reference:
//!     rms = sqrt(mean(x**2, dim=1, keepdim=True) + eps)   # shape (B,1,H,W)
//!     y   = x / rms                                       # shape (B,C,H,W)
//!
//! Memory layout (PyTorch contiguous, row-major):
//!     linear_index(b,c,h,w) = b*C*H*W + c*H*W + h*W + w
//! For a fixed (b, h, w), the C elements to reduce live at stride HW in memory.
//!
//! Strategy: two kernels, both elementwise w.r.t. their output.
//!   1. `rms_norm_reduce` — one thread per (b, h, w) pixel; sequentially sums
//!      the C strided values, writes inv_rms = 1/sqrt(mean(x^2) + eps).
//!   2. `rms_norm_apply`  — one thread per output element; multiplies x by the
//!      pre-computed inv_rms for its (b, h, w).
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - "Memory Write Patterns": `chunk_mut` + `MapContinuousLinear::new(1)`,
//!     LOCAL slot index `out[0]`.
//!   - Golden Rules #1, #6: `u32` kernel params and indices; bounds-guarded
//!     global thread id (no grid-stride writes).
//!   - "Common Pitfalls": separate `&[f32]` read and `&mut [f32]` write
//!     parameters (the apply kernel reads `x` and `inv_rms` and writes `y`).
//!   - Multi-Kernel Benchmarks: recreate `gpu_config!` before each launch
//!     (non-Copy).

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn rms_norm_reduce(x: &[f32], inv_rms: &mut [f32], B: u32, C: u32, HW: u32, eps: f32) {
    let mut out = chunk_mut(inv_rms, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let total = B * HW;
    if idx < total {
        let b = idx / HW;
        let hw = idx - b * HW;
        let base = b * C * HW + hw;
        let mut sumsq = 0.0f32;
        let mut c: u32 = 0;
        while c < C {
            let v = x[(base + c * HW) as usize];
            sumsq += v * v;
            c += 1;
        }
        let mean = sumsq / (C as f32);
        out[0] = 1.0f32 / (mean + eps).sqrt();
    }
}

#[gpu::cuda_kernel]
pub fn rms_norm_apply(x: &[f32], inv_rms: &[f32], y: &mut [f32], C: u32, HW: u32, N: u32) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < N {
        // idx = b*C*HW + c*HW + hw  →  inv index = b*HW + hw
        let chw = C * HW;
        let b = idx / chw;
        let rem = idx - b * chw;
        let hw = rem - (rem / HW) * HW;
        let inv = inv_rms[(b * HW + hw) as usize];
        out[0] = x[idx as usize] * inv;
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
    assert_eq!(shape.len(), 4, "rms_norm: shape=[B,C,H,W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = h * w;
    let n_total = b * c * hw;
    let n_pixels = b * hw;

    let h_x = super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];
    let mut h_inv = vec![0f32; n_pixels];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let mut d_inv = ctx.new_tensor_view(h_inv.as_mut_slice()).unwrap();

    let bb = b as u32;
    let cc = c as u32;
    let hwh = hw as u32;
    let nn = n_total as u32;
    let eps: f32 = 1e-5;

    let bs: u32 = 256;
    let gs_red: u32 = (n_pixels as u32).div_ceil(bs);
    let gs_app: u32 = nn.div_ceil(bs);

    // Warm up once before timing.
    {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, cc, hwh, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hwh, nn).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, cc, hwh, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hwh, nn).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(cfg_r, ctx, md, &d_x, &mut d_inv, bb, cc, hwh, eps).unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hwh, nn).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_inv);
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
