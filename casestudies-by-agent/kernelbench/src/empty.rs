//! Empty kernel — purely measures SeGuRu/CUDA launch overhead.
use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn empty_kernel(_x: &[f32], y: &mut [f32], _n: u32) {
    // Minimal work: write 0 to y[0] from thread 0. Ensures compiler/loader emit the symbol.
    let mut c = chunk_mut(y, MapContinuousLinear::new(1));
    if block_id::<DimX>() == 0 && thread_id::<DimX>() == 0 {
        c[0] = 0.0;
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
    let n: usize = shape.iter().product();
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let nn = n as u32;

    // Warmup
    let warmup_iters = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        empty_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let warmup = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        empty_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (us, warmup)
}
