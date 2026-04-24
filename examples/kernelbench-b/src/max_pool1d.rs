//! MaxPool1d k=4 s=4: input [N, C, L] -> output [N, C, L/4].
//! 1 thread per output; each thread reads 4 contiguous inputs and outputs max.
use std::path::Path;
use std::time::Instant;
use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn max_pool1d_kernel(x: &[f32], y: &mut [f32], n_out: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n_out {
        let base = (idx * 4) as usize;
        let a = x[base];
        let b = x[base + 1];
        let c = x[base + 2];
        let d = x[base + 3];
        y_chunk[0] = a.max(b).max(c.max(d));
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path, out_dir: &Path, iters: usize, shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 3);
    let (n, c, l) = (shape[0], shape[1], shape[2]);
    assert!(l % 4 == 0);
    let lo = l / 4;
    let n_in = n * c * l;
    let n_out = n * c * lo;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_in);
    let mut h_y = vec![0f32; n_out];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let nn = n_out as u32; let bs: u32 = 256; let gs: u32 = nn.div_ceil(bs);
    { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
      max_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let wi = 5; let wt = Instant::now();
    for _ in 0..wi { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        max_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        max_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap(); }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
