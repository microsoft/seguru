//! Port of cuda/batch_norm.cu: BatchNorm2d training-mode normalization over [B,C,H,W], no affine.
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Grid, Thread, build_chunk_scope};
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;
const EPS: f32 = 1e-5;

#[gpu::cuda_kernel]
pub fn batch_norm_stats(
    x: &[f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    B: u32,
    C: u32,
    HW: u32,
    eps: f32,
) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);
    let tid = thread_id::<DimX>();
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let c = block_id::<DimX>();
    let count = B * HW;
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    let mut i = tid;
    while i < count {
        let b = i / HW;
        let hw = i - b * HW;
        let v = x[(b * C * HW + c * HW + hw) as usize];
        sum += v;
        sumsq += v * v;
        i += block_dim::<DimX>();
    }

    let mut smem_sum = GpuShared::<[f32; 32]>::zero();
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();
    let warp_sum = warp.redux(ReduxAdd, sum);
    let warp_sumsq = warp.redux(ReduxAdd, sumsq);
    {
        let mut sum_c = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        let mut sq_c = smem_sumsq
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane == 0 {
            sum_c[0] = warp_sum;
            sq_c[0] = warp_sumsq;
        }
    }
    sync_threads();
    let v_sum = if lane < num_warps {
        smem_sum[lane as usize]
    } else {
        0.0
    };
    let v_sumsq = if lane < num_warps {
        smem_sumsq[lane as usize]
    } else {
        0.0
    };
    let total_sum = warp.redux(ReduxAdd, v_sum);
    let total_sumsq = warp.redux(ReduxAdd, v_sumsq);
    if tid == 0 {
        let inv_count = 1.0f32 / (count as f32);
        let m = total_sum * inv_count;
        let var0 = total_sumsq * inv_count - m * m;
        let var = if var0 < 0.0 { 0.0 } else { var0 };
        let mut mean_out = mean
            .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
            .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
        let mut rstd_out = rstd
            .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
            .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
        mean_out[0] = m;
        rstd_out[0] = (var + eps).rsqrt();
    }
}

#[gpu::cuda_kernel]
pub fn batch_norm_apply(
    x: &[f32],
    mean: &[f32],
    rstd: &[f32],
    y: &mut [f32],
    C: u32,
    HW: u32,
    N: u32,
) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < N {
        let c = (idx / HW) - (idx / (C * HW)) * C;
        out[0] = (x[idx as usize] - mean[c as usize]) * rstd[c as usize];
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
    assert_eq!(shape.len(), 4, "batch_norm: shape=[B,C,H,W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(c, 64, "batch_norm benchmark keeps C=64");
    let hw = h * w;
    let n = b * c * hw;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let mut h_mean = vec![0f32; c];
    let mut h_rstd = vec![0f32; c];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let mut d_mean = ctx.new_tensor_view(h_mean.as_mut_slice()).unwrap();
    let mut d_rstd = ctx.new_tensor_view(h_rstd.as_mut_slice()).unwrap();
    let bb = b as u32;
    let cc = c as u32;
    let hwu = hw as u32;
    let nn = n as u32;
    let gs_stats = cc;
    let gs_apply = nn.div_ceil(BLOCK);

    {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_stats::launch(
            cfg_s,
            ctx,
            md,
            &d_x,
            &mut d_mean,
            &mut d_rstd,
            bb,
            cc,
            hwu,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_mean, &*d_rstd, &mut d_y, cc, hwu, nn,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_stats::launch(
            cfg_s,
            ctx,
            md,
            &d_x,
            &mut d_mean,
            &mut d_rstd,
            bb,
            cc,
            hwu,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_mean, &*d_rstd, &mut d_y, cc, hwu, nn,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_stats::launch(
            cfg_s,
            ctx,
            md,
            &d_x,
            &mut d_mean,
            &mut d_rstd,
            bb,
            cc,
            hwu,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        batch_norm_apply::launch(
            cfg_a, ctx, md, &d_x, &*d_mean, &*d_rstd, &mut d_y, cc, hwu, nn,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
