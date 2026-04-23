//! SeGuRu port of `cuda/sum_dim.cu` — row-wise sum over last dim.
//!
//! One block per row, block size 256 (8 warps). Each thread grid-strides over
//! the row using `Float4` loads (mirrors `reinterpret_cast<const float4*>`).
//! Two-stage block reduction: warp-shuffle via `warp.redux(ReduxAdd, …)`, then
//! warp 0 reloads the per-warp partials from shared memory (with the
//! `lane < WARPS` partial-mask guard from the .cu) and warp-reduces again.
//! Thread 0 writes `y[row]`.
//!
//! Input shape `[B, D]` with `D % 4 == 0` (D=16384 in the benchmark, same
//! precondition the CUDA kernel relies on for the fast Float4 path).

use std::path::Path;
use std::time::Instant;

use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Thread};
use gpu::prelude::*;

/// One block per row. `BLOCK = 256` → 8 warps → 32-slot smem is safe up to
/// `BLOCK = 1024`, matching the template bound in the `.cu`.
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[Float4], y: &mut [f32], D4: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // One shared-memory slot per warp (up to 32 warps, mirrors warp_sums[WARPS]).
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // One output f32 per block — chained Grid→Block→Thread scope.
    let mut sum_out = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    // Row slice in Float4 units.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];

    // --- Thread-local grid-strided partial sum with Float4 loads ---
    // Mirrors the .cu loop: `for (i=tid; i<D4; i+=BLOCK) acc += v.x+v.y+v.z+v.w;`
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v: Float4 = x_row[i as usize];
        acc += v[0] + v[1] + v[2] + v[3];
        i += block_dim::<DimX>();
    }

    // --- Stage 1: warp-level reduction (replaces the __shfl_down_sync loop). ---
    let warp_sum = warp.redux(ReduxAdd, acc);

    // --- Stage 2: collect per-warp sums in shared memory. ---
    {
        let mut smem_chunk = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    // --- Warp 0 reduces the WARPS partial sums (partial-mask guard). ---
    let smem_val = if lane_id < num_warps {
        smem_sum[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, smem_val);

    if tid == 0 {
        sum_out[0] = block_sum;
    }
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir:  &Path,
    out_dir: &Path,
    iters:   usize,
    shape:   &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 2, "sum_dim: shape must be [B, D]");
    let (b, d) = (shape[0], shape[1]);
    assert!(d % 4 == 0, "sum_dim (from_cuda): D must be divisible by 4");
    let n_total = b * d;

    let h_x = crate::read_bin(&in_dir.join("x.bin"), n_total);

    // f32 -> Float4 host-side repack (untimed). Mirrors the .cu's
    // `reinterpret_cast<const float4*>` without aliasing unsafety.
    let h_x4: Vec<Float4> = h_x
        .chunks_exact(4)
        .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_y = vec![0f32; b];

    let d_x     = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let bb = b as u32;
    let d4 = (d / 4) as u32;
    let bs: u32 = 256;
    let gs: u32 = bb; // one block per row

    // Warmup (untimed).
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();

    // Warmup timing.
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, d4).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
