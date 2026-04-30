//! Port of cuda/log_softmax.cu — same algorithm as softmax, final = (x-max)-log(sum).
use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use std::path::Path;
use std::time::Instant;

const BLOCK: u32 = 256;

#[gpu::cuda_kernel]
pub fn log_softmax_kernel(x: &[f32], y: &mut [f32], D: u32) {
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
        let om = local_max;
        local_max = local_max.max(v);
        local_sum *= GPUDeviceFloatIntrinsics::exp(om - local_max);
        local_sum += GPUDeviceFloatIntrinsics::exp(v - local_max);
        i += block_dim::<DimX>();
    }
    let wm = warp.redux(ReduxMax, local_max);
    {
        let mut s = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = wm;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_max[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let bm = warp.redux(ReduxMax, sv);
    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - bm);
    let ws = warp.redux(ReduxAdd, local_sum);
    {
        let mut s = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            s[0] = ws;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let bs = warp.redux(ReduxAdd, sv);
    let log_sum = GPUDeviceFloatIntrinsics::log(bs);
    let out_map = reshape_map!(
        [D / block_dim::<DimX>()] | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D {
        y_chunk[slot] = (x_row[i as usize] - bm) - log_sum;
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
    assert_eq!(shape.len(), 2);
    let (b, d) = (shape[0], shape[1]);
    assert!(d % BLOCK as usize == 0);
    let n = b * d;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    let dd = d as u32;
    let bs = BLOCK;
    let gs = b as u32;
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        log_softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        log_softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        log_softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
