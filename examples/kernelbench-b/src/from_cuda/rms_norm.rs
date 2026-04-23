//! Port of cuda/rms_norm.cu — RMSNorm along dim=1 of a 4-D (B,C,H,W) tensor.
//!
//! Mirrors the CUDA implementation one-for-one:
//!   Pass 1 (rms_norm_reduce): one thread owns 4 consecutive w-pixels
//!     (a float4). Thread loops over C reading a float4 per c and
//!     accumulates 4 independent sum-of-squares; writes float4 of
//!     inv_rms = 1 / sqrt(mean(x^2) + eps).
//!   Pass 2 (rms_norm_apply):  one thread owns 4 consecutive output
//!     elements (a float4 into the full (B,C,HW) tensor). Broadcasts the
//!     (b, w)-indexed inv_rms float4 and writes y = x * inv_rms.
//!
//! Thread geometry: block size 256, 1 thread per quad. HW % 4 == 0 required.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

#[gpu::cuda_kernel]
pub fn rms_norm_reduce(
    x: &[Float4],
    inv_rms: &mut [Float4],
    C: u32,
    HW4: u32,
    total_quads: u32,
    eps: f32,
    inv_C: f32,
) {
    let mut out = chunk_mut(inv_rms, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < total_quads {
        // Decode quad -> (b, hw4). total_quads == B * HW4.
        let b = idx / HW4;
        let hw4 = idx - b * HW4;
        let base4 = b * C * HW4 + hw4;

        let mut s0: f32 = 0.0;
        let mut s1: f32 = 0.0;
        let mut s2: f32 = 0.0;
        let mut s3: f32 = 0.0;
        let mut c: u32 = 0;
        while c < C {
            let v: Float4 = x[(base4 + c * HW4) as usize];
            s0 += v[0] * v[0];
            s1 += v[1] * v[1];
            s2 += v[2] * v[2];
            s3 += v[3] * v[3];
            c += 1;
        }

        let r0 = (s0 * inv_C + eps).rsqrt();
        let r1 = (s1 * inv_C + eps).rsqrt();
        let r2 = (s2 * inv_C + eps).rsqrt();
        let r3 = (s3 * inv_C + eps).rsqrt();
        out[0] = Float4::new([r0, r1, r2, r3]);
    }
}

#[gpu::cuda_kernel]
pub fn rms_norm_apply(
    x: &[Float4],
    inv_rms: &[Float4],
    y: &mut [Float4],
    C: u32,
    HW4: u32,
    total_quads: u32,
) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < total_quads {
        // idx over (B, C, HW4). Decode (b, hw4).
        let chw4 = C * HW4;
        let b = idx / chw4;
        let rem = idx - b * chw4;
        let hw4 = rem - (rem / HW4) * HW4;

        let xv: Float4 = x[idx as usize];
        let iv: Float4 = inv_rms[(b * HW4 + hw4) as usize];
        let mut yv: Float4 = Float4::new([0.0; 4]);
        yv[0] = xv[0] * iv[0];
        yv[1] = xv[1] * iv[1];
        yv[2] = xv[2] * iv[2];
        yv[3] = xv[3] * iv[3];
        out[0] = yv;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path,
    out_dir: &Path,
    iters: usize,
    shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 4, "rms_norm: shape=[B,C,H,W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = h * w;
    assert!(hw % 4 == 0, "rms_norm: H*W must be divisible by 4");
    let hw4 = hw / 4;
    let n_total = b * c * hw;
    let n_pixels = b * hw;

    let h_x_f32 = crate::read_bin(&in_dir.join("x.bin"), n_total);
    let h_x: Vec<Float4> = h_x_f32
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y: Vec<Float4> = vec![Float4::new([0.0; 4]); n_total / 4];
    let mut h_inv: Vec<Float4> = vec![Float4::new([0.0; 4]); n_pixels / 4];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let mut d_inv = ctx.new_tensor_view(h_inv.as_mut_slice()).unwrap();

    let cc = c as u32;
    let hw4_u = hw4 as u32;
    let total_quads_red: u32 = (b * hw4) as u32;       // B * HW/4
    let total_quads_app: u32 = (b * c * hw4) as u32;   // B * C * HW/4
    let eps: f32 = 1e-5;
    let inv_c: f32 = 1.0 / (c as f32);

    let bs: u32 = 256;
    let gs_red: u32 = total_quads_red.div_ceil(bs);
    let gs_app: u32 = total_quads_app.div_ceil(bs);

    // Untimed warmup.
    {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(
            cfg_r, ctx, md, &d_x, &mut d_inv, cc, hw4_u, total_quads_red, eps, inv_c,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hw4_u, total_quads_app,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(
            cfg_r, ctx, md, &d_x, &mut d_inv, cc, hw4_u, total_quads_red, eps, inv_c,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hw4_u, total_quads_app,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg_r = gpu_host::gpu_config!(gs_red, 1, 1, bs, 1, 1, 0);
        rms_norm_reduce::launch(
            cfg_r, ctx, md, &d_x, &mut d_inv, cc, hw4_u, total_quads_red, eps, inv_c,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_app, 1, 1, bs, 1, 1, 0);
        rms_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_inv, &mut d_y, cc, hw4_u, total_quads_app,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(h_y.as_mut_slice()).unwrap();
    drop(d_inv);
    drop(d_x);

    let h_y_flat: &[f32] = h_y.as_slice().flatten();
    crate::write_bin(&out_dir.join("y.bin"), h_y_flat);

    (kernel_us, warmup_us)
}
