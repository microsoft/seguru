//! Port of cuda/mse_loss.cu — scalar mean((a-b)^2).
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 1024;

#[gpu::cuda_kernel]
pub fn mse_loss_kernel(a: &[f32], b: &[f32], out: &mut [f32], N: u32, inv_n: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smem = GpuShared::<[f32; 32]>::zero();
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < N {
        let d = a[i as usize] - b[i as usize];
        acc += d * d;
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxAdd, acc);
    {
        let mut sl = smem
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = ws;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem[lane_id as usize]
    } else {
        0.0
    };
    let bs = warp.redux(ReduxAdd, sv);
    if tid == 0 {
        let mut o = chunk_mut(out, MapContinuousLinear::new(1));
        o[0] = bs * inv_n;
    }
}

#[gpu::cuda_kernel]
pub fn mse_loss_kernel_vec(a: &[Float4], b: &[Float4], out: &mut [f32], N4: u32, inv_n: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smem = GpuShared::<[f32; 32]>::zero();
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < N4 {
        let av: Float4 = a[i as usize];
        let bv: Float4 = b[i as usize];
        let d0 = av[0] - bv[0];
        let d1 = av[1] - bv[1];
        let d2 = av[2] - bv[2];
        let d3 = av[3] - bv[3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxAdd, acc);
    {
        let mut sl = smem
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = ws;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem[lane_id as usize]
    } else {
        0.0
    };
    let bs = warp.redux(ReduxAdd, sv);
    if tid == 0 {
        let mut o = chunk_mut(out, MapContinuousLinear::new(1));
        o[0] = bs * inv_n;
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
    let h_a = super::read_bin(&in_dir.join("a.bin"), n);
    let h_b = super::read_bin(&in_dir.join("b.bin"), n);
    let mut h_y = vec![0f32; 1];
    let inv_n = 1.0f32 / (n as f32);
    let bs = BLOCK;
    let gs: u32 = 1;

    let (kernel_us, warmup_us) = if n % 4 == 0 {
        let n4 = n / 4;
        let h_a4: Vec<Float4> = h_a
            .chunks_exact(4)
            .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
            .collect();
        let h_b4: Vec<Float4> = h_b
            .chunks_exact(4)
            .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
            .collect();
        let d_a4 = ctx.new_tensor_view(h_a4.as_slice()).unwrap();
        let d_b4 = ctx.new_tensor_view(h_b4.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
        let n4u = n4 as u32;
        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel_vec::launch(cfg, ctx, md, &d_a4, &d_b4, &mut d_y, n4u, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let wi = 5;
        let wt = Instant::now();
        for _ in 0..wi {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel_vec::launch(cfg, ctx, md, &d_a4, &d_b4, &mut d_y, n4u, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel_vec::launch(cfg, ctx, md, &d_a4, &d_b4, &mut d_y, n4u, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
        d_y.copy_to_host(&mut h_y).unwrap();
        (kernel_us, warmup_us)
    } else {
        let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
        let nn = n as u32;
        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel::launch(cfg, ctx, md, &d_a, &d_b, &mut d_y, nn, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let wi = 5;
        let wt = Instant::now();
        for _ in 0..wi {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel::launch(cfg, ctx, md, &d_a, &d_b, &mut d_y, nn, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            mse_loss_kernel::launch(cfg, ctx, md, &d_a, &d_b, &mut d_y, nn, inv_n).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
        d_y.copy_to_host(&mut h_y).unwrap();
        (kernel_us, warmup_us)
    };

    super::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
