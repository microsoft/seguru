//! Port of cuda/group_norm.cu: GroupNorm over [B,C,H,W] with 8 groups, no affine.
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Grid, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;
const EPS: f32 = 1e-5;
const GROUPS: usize = 8;

#[allow(clippy::too_many_arguments)]
#[gpu::cuda_kernel]
pub fn group_norm_stats_kernel(
    x: &[Float4],
    mean: &mut [f32],
    rstd: &mut [f32],
    group_elems4: u32,
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
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * group_elems4 as usize)..((row + 1) * group_elems4 as usize)];
    let mut local_sum = 0.0f32;
    let mut local_sumsq = 0.0f32;
    let mut i = tid;
    while i < group_elems4 {
        let v = x_row[i as usize];
        local_sum += v[0] + v[1] + v[2] + v[3];
        local_sumsq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        i += block_dim::<DimX>();
    }

    let mut smem_sum = GpuShared::<[f32; 32]>::zero();
    let mut smem_sumsq = GpuShared::<[f32; 32]>::zero();
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    let warp_sumsq = warp.redux(ReduxAdd, local_sumsq);
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
    let inv_n = 1.0f32 / ((group_elems4 as f32) * 4.0f32);
    let mean_v = total_sum * inv_n;
    let var0 = total_sumsq * inv_n - mean_v * mean_v;
    let var = if var0 < 0.0 { 0.0 } else { var0 };
    let rstd_v = (var + eps).rsqrt();

    let mut mean_chunk = mean
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    let mut rstd_chunk = rstd
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    if tid == 0 {
        mean_chunk[0] = mean_v;
        rstd_chunk[0] = rstd_v;
    }
}

#[allow(clippy::too_many_arguments)]
#[gpu::cuda_kernel]
pub fn group_norm_apply_kernel(
    x: &[Float4],
    mean: &[f32],
    rstd: &[f32],
    y: &mut [Float4],
    group_elems4: u32,
    total4: u32,
) {
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    if idx < total4 {
        let row = idx / group_elems4;
        let m = mean[row as usize];
        let rs = rstd[row as usize];
        let v = x[idx as usize];
        out[0] = Float4::new([
            (v[0] - m) * rs,
            (v[1] - m) * rs,
            (v[2] - m) * rs,
            (v[3] - m) * rs,
        ]);
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
    assert_eq!(shape.len(), 4, "group_norm: shape=[B,C,H,W]");
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(
        b > 0 && c > 0 && h > 0 && w > 0,
        "group_norm requires non-empty B, C, H, and W"
    );
    assert_eq!(c, 64, "group_norm benchmark keeps C=64");
    assert_eq!(c % GROUPS, 0, "C must be divisible by 8 groups");
    let hw = h.checked_mul(w).expect("group_norm H*W overflow");
    let channels_per_group = c / GROUPS;
    let group_elems = channels_per_group
        .checked_mul(hw)
        .expect("group_norm group_elems overflow");
    assert_eq!(
        group_elems % 4,
        0,
        "group elements must be divisible by Float4 width"
    );
    let n = b
        .checked_mul(c)
        .and_then(|v| v.checked_mul(hw))
        .expect("group_norm input element count overflow");
    assert_eq!(n % 4, 0, "total elements must be divisible by Float4 width");
    let stats_len = b
        .checked_mul(GROUPS)
        .expect("group_norm stats length overflow");
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_mean = vec![0f32; stats_len];
    let mut h_rstd = vec![0f32; stats_len];
    let mut h_y4 = vec![Float4::new([0.0; 4]); n / 4];

    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_mean = ctx.new_tensor_view(h_mean.as_mut_slice()).unwrap();
    let mut d_rstd = ctx.new_tensor_view(h_rstd.as_mut_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();

    let group_elems4 =
        u32::try_from(group_elems / 4).expect("group_norm group_elems4 exceeds u32 limit");
    let total4 = u32::try_from(n / 4).expect("group_norm total4 exceeds u32 limit");
    let gs_stats = u32::try_from(stats_len).expect("group_norm stats grid exceeds u32 limit");
    let gs_apply = u32::try_from((n / 4).div_ceil(BLOCK as usize))
        .expect("group_norm apply grid exceeds u32 limit");

    {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        group_norm_stats_kernel::launch(
            cfg_s,
            ctx,
            md,
            &d_x4,
            &mut d_mean,
            &mut d_rstd,
            group_elems4,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        group_norm_apply_kernel::launch(
            cfg_a,
            ctx,
            md,
            &d_x4,
            &d_mean,
            &d_rstd,
            &mut d_y4,
            group_elems4,
            total4,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        group_norm_stats_kernel::launch(
            cfg_s,
            ctx,
            md,
            &d_x4,
            &mut d_mean,
            &mut d_rstd,
            group_elems4,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        group_norm_apply_kernel::launch(
            cfg_a,
            ctx,
            md,
            &d_x4,
            &d_mean,
            &d_rstd,
            &mut d_y4,
            group_elems4,
            total4,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg_s = gpu_host::gpu_config!(gs_stats, 1, 1, BLOCK, 1, 1, 0);
        group_norm_stats_kernel::launch(
            cfg_s,
            ctx,
            md,
            &d_x4,
            &mut d_mean,
            &mut d_rstd,
            group_elems4,
            EPS,
        )
        .unwrap();
        let cfg_a = gpu_host::gpu_config!(gs_apply, 1, 1, BLOCK, 1, 1, 0);
        group_norm_apply_kernel::launch(
            cfg_a,
            ctx,
            md,
            &d_x4,
            &d_mean,
            &d_rstd,
            &mut d_y4,
            group_elems4,
            total4,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y4.copy_to_host(&mut h_y4).unwrap();
    for (i, v) in h_y4.iter().enumerate() {
        h_y[i * 4] = v[0];
        h_y[i * 4 + 1] = v[1];
        h_y[i * 4 + 2] = v[2];
        h_y[i * 4 + 3] = v[3];
    }
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
