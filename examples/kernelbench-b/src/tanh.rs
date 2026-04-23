//! Tanh elementwise kernel.
//!
//! PyTorch reference:
//!     y = torch.tanh(x)     // shape: [batch_size, dim]
//!
//! Skill-doc patterns used:
//!   - `chunk_mut` + `MapContinuousLinear::new(1)` for the 1D output write
//!     (Memory Write Patterns / Golden Rule #3).
//!   - `u32` kernel parameters and global-thread-id with bounds guard
//!     (Golden Rules #1, #5; "Grid-stride loops DO NOT work").
//!   - Device intrinsic `.tanh()` from `GPUDeviceFloatIntrinsics`
//!     (re-exported via `gpu::prelude::*`). A naive Rust `f32::tanh` would
//!     not work inside a `#[gpu::cuda_kernel]`.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn tanh_kernel(x: &[f32], y: &mut [f32], n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        let v = x[idx as usize];
        y_chunk[0] = v.tanh();
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
    assert_eq!(shape.len(), 2, "tanh: shape=[batch_size, dim]");
    let n = shape[0] * shape[1];

    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let nn = n as u32;
    let bs: u32 = 256;
    let gs: u32 = nn.div_ceil(bs);

    // Warm up once before timing.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        tanh_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup pass (timed for the warmup return value).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        tanh_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        tanh_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_x);

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
