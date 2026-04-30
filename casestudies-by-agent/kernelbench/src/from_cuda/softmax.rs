//! Row-wise softmax — direct port of `cuda/softmax.cu`.
//!
//! Single kernel, one block per row, BLOCK=256:
//!   Pass 1: online max+sum over each thread's strided slice.
//!   Block-wide max reduce (warp-XOR + smem + warp-XOR broadcast).
//!   Rescale each thread's local_sum from local_max to block_max.
//!   Block-wide sum reduce (same shape).
//!   Pass 2: write y[i] = exp(x[i] - block_max) * inv_sum.

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;

const BLOCK: u32 = 256;

#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[f32], y: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One slot per warp for cross-warp reductions (up to BLOCK=1024 → 32 warps).
    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // Subslice the row once — O(1) bounds checks for row reads.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Pass 1: online max+sum over a strided slice of the row ---
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

    // --- Block-wide max reduce ---
    let warp_max = warp.redux(ReduxMax, local_max);
    {
        let mut smem_max_chunk = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_max_chunk[0] = warp_max;
        }
    }
    sync_threads();

    // Every warp loads the num_warps warp-maxes, pads rest with -inf, and does
    // another XOR-butterfly. All lanes of all warps end up with block_max.
    let smem_val = if lane_id < num_warps {
        smem_max[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let block_max = warp.redux(ReduxMax, smem_val);

    // Rescale each thread's partial sum from local_max → block_max.
    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);

    // --- Block-wide sum reduce ---
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut smem_sum_chunk = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_sum_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    let smem_val = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, smem_val);

    let inv_sum = 1.0f32 / block_sum;

    // --- Pass 2: write normalised outputs ---
    //
    // Layout [t0, i0, t1]: pos = t0 + i0*BLOCK + t1*D
    //   t0 ∈ [0, BLOCK)          — thread within block
    //   i0 ∈ [0, D / BLOCK)      — slot index per thread
    //   t1 ∈ [0, grid_dim)       — block (== row) index
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
    assert!(
        d % BLOCK as usize == 0,
        "D ({d}) must be divisible by BLOCK ({BLOCK}) for reshape_map output"
    );
    let n_total = b * d;

    let h_x = super::super::read_bin(&in_dir.join("x.bin"), n_total);
    let mut h_y = vec![0f32; n_total];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb: u32 = b as u32;
    let dd: u32 = d as u32;

    let bs: u32 = BLOCK;
    let gs: u32 = bb; // one block per row

    // Warm up once before timing.
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_x);

    super::super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
