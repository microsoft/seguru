//! Port of cuda/l1_norm.cu.
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::vector::Float4;
use std::path::Path;
use std::time::Instant;

#[gpu::cuda_kernel]
pub fn l1_norm_kernel(x: &[Float4], y: &mut [Float4], D4: u32, eps: f32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let mut smem = GpuShared::<[f32; 32]>::zero();
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];
    let mut s = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let a0 = if v[0] < 0.0 { -v[0] } else { v[0] };
        let a1 = if v[1] < 0.0 { -v[1] } else { v[1] };
        let a2 = if v[2] < 0.0 { -v[2] } else { v[2] };
        let a3 = if v[3] < 0.0 { -v[3] } else { v[3] };
        s += a0 + a1 + a2 + a3;
        i += block_dim::<DimX>();
    }
    let ws = warp.redux(ReduxAdd, s);
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
    let inv = 1.0f32 / (bs + eps);
    let out_map = reshape_map!(
        [D4 / block_dim::<DimX>()] | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        let mut out = Float4::new([0.0; 4]);
        out[0] = v[0] * inv;
        out[1] = v[1] * inv;
        out[2] = v[2] * inv;
        out[3] = v[3] * inv;
        y_chunk[slot] = out;
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
    const BLOCK: u32 = 256;
    assert!(d % 4 == 0);
    let d4 = d / 4;
    assert!(d4 % BLOCK as usize == 0);
    let n = b * d;
    let h_x = crate::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y4: Vec<Float4> = vec![Float4::new([0.0; 4]); b * d4];
    let d_x4 = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y4 = ctx.new_tensor_view(h_y4.as_mut_slice()).unwrap();
    let d4u = d4 as u32;
    let eps = 1e-12f32;
    let bs = BLOCK;
    let gs = b as u32;
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l1_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
    }
    ctx.sync().unwrap();
    let wi = 5;
    let wt = Instant::now();
    for _ in 0..wi {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l1_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / wi as f64;
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        l1_norm_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y4, d4u, eps).unwrap();
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
