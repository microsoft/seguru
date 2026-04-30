//! GELU (tanh approx) — translated from cuda/gelu.cu.
//!
//! Source CUDA uses Float4 vectorized loads/stores with a grid-stride loop
//! capped at 4096 blocks of 256 threads. The skill doc forbids grid-stride
//! writes through `chunk_mut`, so we mirror the *per-thread* vectorized work
//! (one Float4 = 4 gelu_elem invocations) but launch enough threads to cover
//! all Float4 slots exactly once (1 Float4 per thread). Block size = 256,
//! matching the .cu.
//!
//! Input size (n = 2048 * 65536) is divisible by 4, so no scalar tail is
//! needed.
use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

#[gpu::cuda_kernel]
pub fn gelu_kernel_vec(x: &[Float4], y: &mut [Float4], n4: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n4 {
        let k0: f32 = 0.7978845608028654;
        let c: f32 = 0.044715;
        let v: Float4 = x[idx as usize];
        let mut out: Float4 = Float4::new([0.0; 4]);
        for k in 0..4 {
            let xv = v[k];
            let inner = k0 * (xv + c * xv * xv * xv);
            out[k] = 0.5 * xv * (1.0 + GPUDeviceFloatIntrinsics::tanh(inner));
        }
        y_chunk[0] = out;
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
    assert_eq!(shape.len(), 2, "gelu: shape=[batch, dim]");
    let n = shape[0] * shape[1];
    assert!(n % 4 == 0, "gelu(from_cuda): n must be a multiple of 4");
    let n4 = n / 4;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); n4];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let nn4 = n4 as u32;
    let bs: u32 = 256;
    let gs: u32 = nn4.div_ceil(bs);

    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        gelu_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y4, nn4).unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        gelu_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y4, nn4).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        gelu_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y4, nn4).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y4.copy_to_host(h_y4.as_mut_slice()).unwrap();
    drop(d_y4);
    drop(d_x4);

    let h_y_flat: &[f32] = h_y4.as_slice().flatten();
    crate::write_bin(&out_dir.join("y.bin"), h_y_flat);

    (kernel_us, warmup_us)
}
