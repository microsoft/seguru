//! Port of cuda/argmax_dim.cu (two-pass; output i64).
use gpu::cg::{CGOperations, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Grid, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;
use std::path::Path;
use std::time::Instant;

#[gpu::device]
#[inline(always)]
fn warp_min_i32(mut v: i32) -> i32 {
    let (p16, _) = gpu::shuffle!(xor, v, 16, 32);
    if p16 < v {
        v = p16;
    }
    let (p8, _) = gpu::shuffle!(xor, v, 8, 32);
    if p8 < v {
        v = p8;
    }
    let (p4, _) = gpu::shuffle!(xor, v, 4, 32);
    if p4 < v {
        v = p4;
    }
    let (p2, _) = gpu::shuffle!(xor, v, 2, 32);
    if p2 < v {
        v = p2;
    }
    let (p1, _) = gpu::shuffle!(xor, v, 1, 32);
    if p1 < v {
        v = p1;
    }
    v
}

#[gpu::cuda_kernel]
pub fn argmax_dim_kernel(x: &[f32], y: &mut [i64], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smax = GpuShared::<[f32; 32]>::zero();
    let mut smin = GpuShared::<[i32; 32]>::zero();
    let mut y_chunk = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    let mut acc = f32::NEG_INFINITY;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        acc = acc.max(v);
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxMax, acc);
    {
        let mut sl = smax
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = ws;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smax[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let bm = warp.redux(ReduxMax, sv);

    let mut local_idx: i32 = i32::MAX;
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        if v == bm {
            let ii = i as i32;
            if ii < local_idx {
                local_idx = ii;
            }
        }
        i += block_dim::<DimX>();
    }
    let wi = warp_min_i32(local_idx);
    {
        let mut sl = smin
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = wi;
        }
    }
    sync_threads();
    if tid < 32 {
        let sv = if lane_id < num_warps {
            smin[lane_id as usize]
        } else {
            i32::MAX
        };
        let bmin = warp_min_i32(sv);
        if lane_id == 0 {
            y_chunk[0] = bmin as i64;
        }
    }
}

#[gpu::cuda_kernel]
pub fn argmax_dim_kernel_vec(x: &[Float4], y: &mut [i64], D4: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smax = GpuShared::<[f32; 32]>::zero();
    let mut smin = GpuShared::<[i32; 32]>::zero();
    let mut y_chunk = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    let mut acc = f32::NEG_INFINITY;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        acc = acc.max(v[0]).max(v[1]).max(v[2]).max(v[3]);
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxMax, acc);
    {
        let mut sl = smax
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = ws;
        }
    }
    sync_threads();
    let sv = if lane_id < num_warps {
        smax[lane_id as usize]
    } else {
        f32::NEG_INFINITY
    };
    let bm = warp.redux(ReduxMax, sv);

    let mut local_idx: i32 = i32::MAX;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let base = (i * 4) as i32;
        if v[0] == bm && base < local_idx {
            local_idx = base;
        }
        if v[1] == bm && base + 1 < local_idx {
            local_idx = base + 1;
        }
        if v[2] == bm && base + 2 < local_idx {
            local_idx = base + 2;
        }
        if v[3] == bm && base + 3 < local_idx {
            local_idx = base + 3;
        }
        i += block_dim::<DimX>();
    }
    let wi = warp_min_i32(local_idx);
    {
        let mut sl = smin
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            sl[0] = wi;
        }
    }
    sync_threads();
    if tid < 32 {
        let sv = if lane_id < num_warps {
            smin[lane_id as usize]
        } else {
            i32::MAX
        };
        let bmin = warp_min_i32(sv);
        if lane_id == 0 {
            y_chunk[0] = bmin as i64;
        }
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
    let n = b * d;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0i64; b];
    let bs: u32 = 256;
    let gs: u32 = b as u32;

    let (kernel_us, warmup_us) = if d % 4 == 0 {
        let d4 = d / 4;
        let h_x4: Vec<Float4> = h_x
            .chunks_exact(4)
            .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
            .collect();
        let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
        let d4u = d4 as u32;
        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y, d4u).unwrap();
        }
        ctx.sync().unwrap();
        let wi = 5;
        let wt = Instant::now();
        for _ in 0..wi {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y, d4u).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel_vec::launch(cfg, ctx, md, &d_x4, &mut d_y, d4u).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
        d_y.copy_to_host(&mut h_y).unwrap();
        (kernel_us, warmup_us)
    } else {
        let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
        let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
        let dd = d as u32;
        {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let wi = 5;
        let wt = Instant::now();
        for _ in 0..wi {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            argmax_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, dd).unwrap();
        }
        ctx.sync().unwrap();
        let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;
        d_y.copy_to_host(&mut h_y).unwrap();
        (kernel_us, warmup_us)
    };

    crate::write_bin_i64(&out_dir.join("y.bin"), &h_y);
    (kernel_us, warmup_us)
}
