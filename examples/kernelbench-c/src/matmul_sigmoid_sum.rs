//! 56_Matmul_Sigmoid_Sum — fused nn.Linear + sigmoid + row-sum.
//!
//! PyTorch reference:
//!     y = torch.sigmoid(x @ W.T + b).sum(dim=1, keepdim=True)
//!
//! Shapes:
//!     x: [M, K]        (M=128, K=32768)
//!     W: [N, K]        (N=32768)
//!     b: [N]
//!     y: [M, 1]
//!
//! The [M, N] intermediate (16 MB) is never materialized — each block owns
//! one output row and accumulates `partial += sigmoid(acc + bias[n])` into
//! a scalar, then block-reduces to one f32 per row.
//!
//! Strategy (mirrors `cuda/matmul_sigmoid_sum.cu`):
//!   * grid = (M,), block = (BDIM=256,). One block per row m.
//!   * Outer loop sweeps N in chunks of NC = BDIM * TN = 1024.
//!   * Per chunk, TN=4 register accumulators per thread cover 4 n-columns
//!     (n = nc + j*BDIM + tid for j=0..4).
//!   * Inner loop tiles K by BK=256 into a `GpuShared<[f32; 256]>` x-tile.
//!   * Block reduction uses warp-redux + smem cross-warp, following the
//!     Golden Rule #7 row-reduction pattern (see sum_dim.rs template).
//!
//! Skill-doc patterns used (cuda-to-seguru-porting-skill.md):
//!   - Golden Rule #1: `u32` indices end-to-end.
//!   - Golden Rule #7: 1-block-per-row cooperative row reduction.
//!   - "Shared Memory Tiling": static `GpuShared` + `reshape_map` load
//!     (disjoint per-thread slot, no bounds check on the store).
//!   - "Warp reductions": `ThreadWarpTile::redux(ReduxAdd, _)`.
//!   - `chunk_to_scope(grid2block, _)` for one output per block.
//!
//! Compute dominates bandwidth: ≈137 GFMA per row. atol=0.2 in compare.py
//! because f32 row-summing over N=32768 drifts vs. Torch's reduced path.

use std::path::Path;
use std::time::Instant;

use crunchy::unroll;
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{build_chunk_scope, Block, Grid, Thread};
use gpu::prelude::*;

const BDIM: u32 = 256;
const BK: u32 = 256;
const TN: u32 = 4;
const NC: u32 = BDIM * TN; // 1024

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn matmul_sigmoid_sum_kernel(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    M: u32,
    N: u32,
    K: u32,
) {
    let _ = M;

    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);
    let grid2block = build_chunk_scope(Grid, Block);
    let block2thread = build_chunk_scope(Block, Thread);

    let m = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size();

    let mut xs = gpu::GpuShared::<[f32; BK as usize]>::zero();
    let mut smem_red = gpu::GpuShared::<[f32; 32]>::zero();

    // One output scalar per block — chained Grid→Block→Thread scope.
    let mut y_out = y
        .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
        .chunk_to_scope(block2thread, MapContinuousLinear::new(1));

    // Disjoint per-thread slot for x-tile loads: slot = tid.
    let xs_load_map = reshape_map!([1] | [BDIM] => layout: [i0, t0]);

    let mut partial = 0.0f32;
    let mut nc: u32 = 0;
    while nc < N {
        // TN register accumulators for this n-chunk.
        let mut acc = [0.0f32; TN as usize];

        let mut k_tile: u32 = 0;
        while k_tile < K {
            // Cooperative load: xs[tid] = x[m, k_tile + tid].
            {
                let mut cx = xs.chunk_mut(xs_load_map);
                cx[0] = x[(m * K + k_tile + tid) as usize];
            }
            sync_threads();

            // Inner compute: BK scalar k-steps, TN FMAs per step.
            let mut k: u32 = 0;
            while k < BK {
                let xk = xs[k as usize];
                unroll! { for j in 0..4 {
                    let n_idx = nc + (j as u32) * BDIM + tid;
                    acc[j] += xk * w[(n_idx * K + k_tile + k) as usize];
                }}
                k += 1;
            }

            sync_threads();
            k_tile += BK;
        }

        // Fused epilogue for this n-chunk: bias + sigmoid + fold into `partial`.
        unroll! { for j in 0..4 {
            let n_idx = nc + (j as u32) * BDIM + tid;
            let v = acc[j] + bias[n_idx as usize];
            let s = 1.0f32 / (1.0f32 + GPUDeviceFloatIntrinsics::exp(-v));
            partial += s;
        }}

        nc += NC;
    }

    // ---- Block reduce `partial` → one scalar via warp redux + smem.
    let warp_sum = warp.redux(ReduxAdd, partial);
    {
        let mut smem_chunk = smem_red
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 {
            smem_chunk[0] = warp_sum;
        }
    }
    sync_threads();

    let smem_val = if lane_id < num_warps {
        smem_red[lane_id as usize]
    } else {
        0.0f32
    };
    let block_sum = warp.redux(ReduxAdd, smem_val);

    if tid == 0 {
        y_out[0] = block_sum;
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
    assert_eq!(shape.len(), 3, "matmul_sigmoid_sum: shape=[M, K, N]");
    let (m, k, n) = (shape[0], shape[1], shape[2]);

    assert!(
        k % BK as usize == 0,
        "K ({}) must be a multiple of BK ({})",
        k,
        BK
    );
    assert!(
        n % NC as usize == 0,
        "N ({}) must be a multiple of NC ({})",
        n,
        NC
    );

    let h_x = crate::read_bin(&in_dir.join("x.bin"), m * k);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), n * k);
    let h_b = crate::read_bin(&in_dir.join("b.bin"), n);
    let mut h_y = vec![0f32; m]; // [M, 1] contiguous == M floats

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let mm = m as u32;
    let nn = n as u32;
    let kk = k as u32;

    let gs: u32 = mm; // one block per row
    let bs: u32 = BDIM;

    // Priming launch (untimed).
    {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        matmul_sigmoid_sum_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
            .unwrap();
    }
    ctx.sync().unwrap();

    // Warmup (timed for reporting).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        matmul_sigmoid_sum_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
            .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        matmul_sigmoid_sum_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
            .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7 (host-side): explicit readback.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
