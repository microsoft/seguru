//! Mean over last dim: [B, D] -> [B]. Same reduction as sum_dim, divide by D.
use std::path::Path;
use std::time::Instant;
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Thread};
use gpu::prelude::*;
use gpu::vector::Float4;

#[gpu::cuda_kernel]
pub fn mean_dim_kernel(x: &[Float4], y: &mut [f32], D4: u32, inv_d: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smem = GpuShared::<[f32; 32]>::zero();
    let mut y_chunk = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        acc += v[0] + v[1] + v[2] + v[3];
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxAdd, acc);
    {
        let mut sl = smem
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 { sl[0] = ws; }
    }
    sync_threads();
    let sv = if lane_id < num_warps { smem[lane_id as usize] } else { 0.0 };
    let bs = warp.redux(ReduxAdd, sv);
    if tid == 0 { y_chunk[0] = bs * inv_d; }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path, out_dir: &Path, iters: usize, shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2);
    let (b, d) = (shape[0], shape[1]);
    assert!(d % 4 == 0);
    let n = b * d;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let h_x4: Vec<Float4> = h_x.chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]])).collect();
    let mut h_y = vec![0f32; b];
    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let d4 = (d / 4) as u32;
    let inv_d = 1.0f32 / (d as f32);
    let bs: u32 = 256; let gs: u32 = b as u32;
    { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
      mean_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4, inv_d).unwrap(); }
    ctx.sync().unwrap();
    let wi = 5; let wt = Instant::now();
    for _ in 0..wi { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        mean_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4, inv_d).unwrap(); }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters { let cfg = gpu_host::gpu_config!(gs,1,1,bs,1,1,0);
        mean_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, d4, inv_d).unwrap(); }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
