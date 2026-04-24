//! Softplus elementwise: y = v > 20 ? v : log1p(exp(v)).
use std::path::Path;
use std::time::Instant;
use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn softplus_kernel(x: &[f32], y: &mut [f32], n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        let v = x[idx as usize];
        y_chunk[0] = if v > 20.0 { v } else {
            GPUDeviceFloatIntrinsics::log1p(GPUDeviceFloatIntrinsics::exp(v))
        };
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path, out_dir: &Path, iters: usize, shape: &[usize],
) -> (f64, f64) {
    let n: usize = shape.iter().product();
    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let nn = n as u32; let bs: u32 = 256; let gs: u32 = nn.div_ceil(bs);
    { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
      softplus_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let warmup_iters = 5; let wt = Instant::now();
    for _ in 0..warmup_iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        softplus_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;
    let t = Instant::now();
    for _ in 0..iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        softplus_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
