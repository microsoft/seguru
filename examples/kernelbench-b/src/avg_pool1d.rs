//! AvgPool1d k=8 s=1 p=4: input [N, C, L] -> output [N, C, L + 1].
use std::path::Path;
use std::time::Instant;
use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn avg_pool1d_kernel(x: &[f32], y: &mut [f32], total_out: u32, l: u32, lo: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < total_out {
        let pos = idx % lo;
        let bc = idx / lo;
        let base = bc * l;
        let start = pos as i32 - 4;
        let mut sum = 0.0f32;
        for k in 0..8 {
            let in_pos = start + k;
            if in_pos >= 0 && in_pos < l as i32 {
                sum += x[(base + in_pos as u32) as usize];
            }
        }
        y_chunk[0] = sum * 0.125;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path, out_dir: &Path, iters: usize, shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 3);
    let (n, c, l) = (shape[0], shape[1], shape[2]);
    let lo = l + 1;
    let n_in = n * c * l;
    let n_out = n * c * lo;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n_in);
    let mut h_y = vec![0f32; n_out];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let total_out = n_out as u32; let l_u32 = l as u32; let lo_u32 = lo as u32;
    let bs: u32 = 256; let gs: u32 = total_out.div_ceil(bs);
    { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
      avg_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, total_out, l_u32, lo_u32).unwrap(); }
    ctx.sync().unwrap();
    let wi = 5; let wt = Instant::now();
    for _ in 0..wi { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        avg_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, total_out, l_u32, lo_u32).unwrap(); }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        avg_pool1d_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, total_out, l_u32, lo_u32).unwrap(); }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
