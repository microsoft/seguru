//! Row-wise inclusive cumulative sum [B, D] -> [B, D].
//! Single-warp kernel: BLOCK=32 threads each own ITEMS=128 contiguous elems.
//! Warp inclusive scan via shuffle-up, then add exclusive offset to locals.
use std::path::Path;
use std::time::Instant;
use gpu::prelude::*;

const BLOCK: u32 = 32;
const ITEMS: u32 = 128;

#[gpu::cuda_kernel]
pub fn cumsum_kernel(x: &[f32], y: &mut [f32], _D: u32) {
    let tid = thread_id::<DimX>();
    let row = block_id::<DimX>() as usize;
    let d_total = (BLOCK * ITEMS) as usize;
    let x_row = &x[(row * d_total)..((row + 1) * d_total)];

    let mut local: [f32; 128] = [0.0; 128];
    let base_local = (tid * ITEMS) as usize;
    let mut run = 0.0f32;
    let mut j: u32 = 0;
    while j < ITEMS {
        run += x_row[base_local + j as usize];
        local[j as usize] = run;
        j += 1;
    }

    // Warp inclusive scan via shuffle-up on 32 lanes.
    let mut scan_val = run;
    let (v1, _)  = gpu::shuffle!(up, scan_val, 1,  32);
    if tid >= 1  { scan_val += v1; }
    let (v2, _)  = gpu::shuffle!(up, scan_val, 2,  32);
    if tid >= 2  { scan_val += v2; }
    let (v4, _)  = gpu::shuffle!(up, scan_val, 4,  32);
    if tid >= 4  { scan_val += v4; }
    let (v8, _)  = gpu::shuffle!(up, scan_val, 8,  32);
    if tid >= 8  { scan_val += v8; }
    let (v16, _) = gpu::shuffle!(up, scan_val, 16, 32);
    if tid >= 16 { scan_val += v16; }

    let offset = scan_val - run;

    let out_map = reshape_map!(
        [ITEMS] | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [i0, t0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut j: u32 = 0;
    while j < ITEMS {
        y_chunk[j] = local[j as usize] + offset;
        j += 1;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path, out_dir: &Path, iters: usize, shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2);
    let (b, d) = (shape[0], shape[1]);
    assert_eq!(d, (BLOCK * ITEMS) as usize, "cumsum: D must equal BLOCK*ITEMS (32*128=4096)");
    let n = b * d;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let dd = d as u32; let bs = BLOCK; let gs = b as u32;
    { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
      cumsum_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap(); }
    ctx.sync().unwrap();
    let wi = 5; let wt = Instant::now();
    for _ in 0..wi { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        cumsum_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap(); }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        cumsum_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap(); }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
