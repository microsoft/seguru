//! Argmin over dim=1 for [B, D1, D2] -> [B, D2] (int64).
use gpu::chunk_scope::{Block, Grid, Thread, build_chunk_scope};
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;

#[gpu::cuda_kernel]
pub fn argmin_dim_kernel(x: &[f32], y: &mut [i64], d1: u32, d2: u32) {
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);
    let tid = thread_id::<DimX>();
    let out = block_id::<DimX>();
    let b = out / d2;
    let k = out - b * d2;
    let mut s_val = GpuShared::<[f32; 256]>::zero();
    let mut s_idx = GpuShared::<[i32; 256]>::zero();
    let mut y_chunk = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    let mut best_v = f32::INFINITY;
    let mut best_i = i32::MAX;
    let mut i = tid;
    while i < d1 {
        let v = x[((b * d1 + i) * d2 + k) as usize];
        let ii = i as i32;
        if v < best_v || (v == best_v && ii < best_i) {
            best_v = v;
            best_i = ii;
        }
        i += block_dim::<DimX>();
    }

    {
        let mut sv = s_val.chunk_mut(MapContinuousLinear::new(1));
        sv[0] = best_v;
    }
    {
        let mut si = s_idx.chunk_mut(MapContinuousLinear::new(1));
        si[0] = best_i;
    }
    sync_threads();

    let mut stride = BLOCK / 2;
    while stride > 0 {
        let mut out_v = 0.0f32;
        let mut out_i = 0i32;
        if tid < stride {
            let lv = s_val[tid as usize];
            let li = s_idx[tid as usize];
            let rv = s_val[(tid + stride) as usize];
            let ri = s_idx[(tid + stride) as usize];
            out_v = lv;
            out_i = li;
            if rv < lv || (rv == lv && ri < li) {
                out_v = rv;
                out_i = ri;
            }
        }
        sync_threads();
        if tid < stride {
            {
                let mut sv = s_val.chunk_mut(MapContinuousLinear::new(1));
                sv[0] = out_v;
            }
            {
                let mut si = s_idx.chunk_mut(MapContinuousLinear::new(1));
                si[0] = out_i;
            }
        }
        sync_threads();
        stride >>= 1;
    }

    if tid == 0 {
        y_chunk[0] = s_idx[0] as i64;
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
    assert_eq!(shape.len(), 3);
    let (b, d1, d2) = (shape[0], shape[1], shape[2]);
    let n = b * d1 * d2;
    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0i64; b * d2];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let d1u = d1 as u32;
    let d2u = d2 as u32;
    let bs = BLOCK;
    let gs = (b * d2) as u32;
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        argmin_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d1u, d2u).unwrap();
    }
    ctx.sync().unwrap();
    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        argmin_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d1u, d2u).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        argmin_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d1u, d2u).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    super::write_bin_i64(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
