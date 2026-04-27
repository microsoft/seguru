//! InstanceNorm2d training-style normalization over each [H,W] plane.
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;
const EPS: f32 = 1e-5;

#[gpu::cuda_kernel]
pub fn instance_norm_kernel(x: &[f32], y: &mut [f32], HW: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let tid = thread_id::<DimX>();
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * HW as usize)..((row + 1) * HW as usize)];
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    let mut i = tid;
    while i < HW {
        let v = x_row[i as usize];
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
    let inv_hw = 1.0f32 / (HW as f32);
    let mean = total_sum * inv_hw;
    let var0 = total_sumsq * inv_hw - mean * mean;
    let var = if var0 < 0.0 { 0.0 } else { var0 };
    let rstd = (var + eps).rsqrt();

    let out_map = reshape_map!(
        [HW / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut out = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut j = tid;
    while j < HW {
        out[slot] = (x_row[j as usize] - mean) * rstd;
        j += block_dim::<DimX>();
        slot += 1;
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
    assert_eq!(shape.len(), 4, "instance_norm: shape=[B,C,H,W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(c, 64, "instance_norm benchmark keeps C=64");
    let hw = h * w;
    assert_eq!(hw % BLOCK as usize, 0, "H*W must be divisible by BLOCK");
    let n = b * c * hw;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let hwu = hw as u32;
    let gs = (b * c) as u32;

    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BLOCK, 1, 1, 0);
        instance_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, hwu, EPS).unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BLOCK, 1, 1, 0);
        instance_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, hwu, EPS).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, BLOCK, 1, 1, 0);
        instance_norm_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, hwu, EPS).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
