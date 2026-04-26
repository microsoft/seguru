//! Row-wise softmax — port of `cuda/softmax.cu` with Float4 row traversal.
use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::{Float4, VecFlatten};

const BLOCK: u32 = 256;

#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[Float4], y: &mut [Float4], D4: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();

    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    let mut local_max = f32::NEG_INFINITY;
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 {
            let xk = v[k];
            let old_max = local_max;
            local_max = local_max.max(xk);
            local_sum *= GPUDeviceFloatIntrinsics::exp(old_max - local_max);
            local_sum += GPUDeviceFloatIntrinsics::exp(xk - local_max);
        }
        i += block_dim::<DimX>();
    }

    let warp_max = warp.redux(ReduxMax, local_max);
    {
        let mut s = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_max;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_max[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let block_max = warp.redux(ReduxMax, sv);

    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut s = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_sum;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, sv);
    let inv_sum = 1.0f32 / block_sum;

    let out_map = reshape_map!(
        [D4 / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let mut out = Float4::new([0.0; 4]);
        for k in 0..4 {
            out[k] = GPUDeviceFloatIntrinsics::exp(v[k] - block_max) * inv_sum;
        }
        y_chunk[slot] = out;
        i += block_dim::<DimX>();
        slot += 1;
    }
}

#[gpu::cuda_kernel]
pub fn softmax_kernel_scalar(x: &[f32], y: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    let mut local_max = f32::NEG_INFINITY;
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        let old_max = local_max;
        local_max = local_max.max(v);
        local_sum *= GPUDeviceFloatIntrinsics::exp(old_max - local_max);
        local_sum += GPUDeviceFloatIntrinsics::exp(v - local_max);
        i += block_dim::<DimX>();
    }

    let warp_max = warp.redux(ReduxMax, local_max);
    {
        let mut s = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_max;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_max[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let block_max = warp.redux(ReduxMax, sv);

    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut s = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = warp_sum;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, sv);
    let inv_sum = 1.0f32 / block_sum;

    let out_map = reshape_map!(
        [D / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D {
        y_chunk[slot] = GPUDeviceFloatIntrinsics::exp(x_row[i as usize] - block_max) * inv_sum;
        i += block_dim::<DimX>();
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
    assert_eq!(shape.len(), 2, "softmax: shape=[B,D]");
    let (b, d) = (shape[0], shape[1]);
    let n_total = b * d;
    let h_x = super::super::read_bin(&in_dir.join("x.bin"), n_total);
    let bs: u32 = BLOCK;
    let gs: u32 = b as u32;

    if d % 4 == 0 && (d / 4) % BLOCK as usize == 0 {
        let d4 = d / 4;
        let h_x4: Vec<Float4> = h_x
            .chunks_exact(4)
            .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); h_x4.len()];
        let d_x = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();
        let dd4: u32 = d4 as u32;

        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd4).unwrap();
        }
        ctx.sync().unwrap();

        let warmup_iters: usize = 5;
        let wt = Instant::now();
        for _ in 0..warmup_iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd4).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd4).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

        d_y.copy_to_host(h_y4.as_mut_slice()).unwrap();
        let h_y_flat: &[f32] = h_y4.as_slice().flatten();
        super::super::write_bin(&out_dir.join("y.bin"), h_y_flat);
        (kernel_us, warmup_us)
    } else {
        assert!(
            d % BLOCK as usize == 0,
            "softmax scalar fallback requires D ({d}) divisible by BLOCK ({BLOCK})"
        );
        let mut h_y = vec![0f32; n_total];
        let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
        let dd: u32 = d as u32;

        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel_scalar::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_iters: usize = 5;
        let wt = Instant::now();
        for _ in 0..warmup_iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel_scalar::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            softmax_kernel_scalar::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
        d_y.copy_to_host(&mut h_y).unwrap();
        super::super::write_bin(&out_dir.join("y.bin"), &h_y);
        (kernel_us, warmup_us)
    }
}
