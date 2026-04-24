//! 12_Gemm_Multiply_LeakyReLU — port of the hand-tuned CUDA kernel in
//! `cuda/gemm_mul_lrelu.cu` to SeGuRu.
//!
//! PyTorch reference:
//!     y = F.leaky_relu((x @ W.T + b) * multiplier, 0.1)
//!
//! Shapes:
//!     x: [M, K]
//!     W: [N, K]  (nn.Linear layout; inner product y[m,n] = Σ_k x[m,k]·W[n,k])
//!     b: [N]
//!     y: [M, N]
//!
//! Tile config (matches the .cu source):
//!   BM=128, BN=128, BK=8, TM=TN=8, block = 16×16 = 256 threads.
//!   Each thread owns an 8×8 output sub-tile → 64 FMAs per pair of shared
//!   loads, which amortizes the bounds-checked `tile[idx]` broadcast reads
//!   that SeGuRu's disjoint-partition model forces in the compute phase
//!   (see docs/cuda-to-seguru-porting-skill.md, "Honest limitation on
//!   COMPUTE phase").
//!
//! Deviations from the CUDA:
//!   * Shared tiles are stored non-transposed (As[m][k], Bs[n][k]) rather
//!     than transposed [k][m] / [k][n]. The reason is that with a non-
//!     transposed layout each thread's 4 tile writes land in 4 *contiguous*
//!     slots (`tid*4 + 0..3`), which matches a simple
//!     `reshape_map!([4] | [16,16] => layout: [i0, t0, t1])` and therefore
//!     eliminates bounds checks on the LOAD phase. The transposed layout
//!     would require per-thread strided writes, which `chunk_mut` cannot
//!     express as a disjoint partition. Compute reads remain scalar.
//!   * Loads use 4 scalar reads per tile per thread instead of a `float4`
//!     vectorized load — the NVVM backend still emits 128-bit loads for
//!     contiguous addresses in many cases and the control flow is simpler.
//!   * Disjoint output writes go through a 6-axis `reshape_map` so the
//!     8×8 per-thread store is proven in-range by the type system.

use std::path::Path;
use std::time::Instant;

use crunchy::unroll;
use gpu::prelude::*;

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 8;
const TM: u32 = 8;
const TN: u32 = 8;
const BDIM_X: u32 = 16; // threads per block along x (n dim)
const BDIM_Y: u32 = 16; // threads per block along y (m dim)

#[gpu::cuda_kernel]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn gemm_mul_lrelu_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[f32],
    y: &mut [f32],
    M: u32,
    N: u32,
    K: u32,
    multiplier: f32,
    negative_slope: f32,
) {
    let _ = M; // M, N are implied by the launch grid; keep for bin-compat.
    let _ = N;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bid_x = block_id::<DimX>();
    let bid_y = block_id::<DimY>();

    let bm = bid_y * BM; // first row of this block's output tile
    let bn = bid_x * BN; // first col of this block's output tile

    // Flat thread id in 0..256, used to partition the per-tile load work.
    let tid = ty * BDIM_X + tx;
    let a_row = tid >> 1; // 0..128 (one row of the 128-row tile per 2 threads)
    let a_col = (tid & 1) << 2; // either 0 or 4 — the 4-wide slice of K this thread loads

    // Shared tiles, non-transposed: As[m_local, k_local], Bs[n_local, k_local].
    let mut tile_a = gpu::GpuShared::<[f32; (BM * BK) as usize]>::zero(); // 1024
    let mut tile_b = gpu::GpuShared::<[f32; (BN * BK) as usize]>::zero(); // 1024

    // Per-thread disjoint slot for tile loads: 4 contiguous slots at tid*4.
    // layout [i0, t0, t1] → mem = (t1*16 + t0)*4 + i0 = tid*4 + i0.
    // That is exactly a_row*BK + a_col + i0 for this thread. ✓
    let load_map = reshape_map!([4] | [16, 16] => layout: [i0, t0, t1]);

    // Per-thread disjoint slot for the 8×8 output tile.
    // We want y[(bid_y*BM + ty*TM + i) * N + (bid_x*BN + tx*TN + j)] at
    // chunk index (j, i) for j∈0..8 and i∈0..8. Decompose the global linear
    // index from fastest-changing axis to slowest:
    //   i0=j (stride 1, size 8)
    //   t0=tx (stride 8, size 16)
    //   t1=bid_x (stride BN=128, size gx)
    //   i1=i (stride N, size 8)
    //   t2=ty (stride TM*N, size 16)
    //   t3=bid_y (stride BM*N, size gy)
    let out_map = reshape_map!(
        [8, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let mut y_thread = chunk_mut(y, out_map);

    // Register accumulator — 64 f32s per thread.
    let mut acc = [[0.0f32; TN as usize]; TM as usize];

    let num_tiles = K / BK;
    let mut tstep: u32 = 0;
    while tstep < num_tiles {
        let k_base4 = tstep * (BK >> 2);

        // ---- Load 4 consecutive K-lanes of X into tile_a.
        {
            let mut ca = tile_a.chunk_mut(load_map);
            let v: Float4 = x[((bm + a_row) * (K >> 2) + k_base4 + (a_col >> 2)) as usize];
            ca[0] = v[0]; ca[1] = v[1]; ca[2] = v[2]; ca[3] = v[3];
        }
        // ---- Load 4 consecutive K-lanes of W into tile_b.
        {
            let mut cb = tile_b.chunk_mut(load_map);
            let v: Float4 = w[((bn + a_row) * (K >> 2) + k_base4 + (a_col >> 2)) as usize];
            cb[0] = v[0]; cb[1] = v[1]; cb[2] = v[2]; cb[3] = v[3];
        }

        sync_threads();

        // ---- Compute: 8×8 register tile, BK=8 inner k steps per outer tile.
        // `unroll!` fully unrolls these loops so the backend keeps the 64-wide
        // accumulator and the 8-wide a/b register fans in registers instead of
        // spilling to local memory.
        let row_off = (ty * TM) as usize;
        let col_off = (tx * TN) as usize;

        unroll! { for kk in 0..8 {
            let mut a_reg = [0.0f32; TM as usize];
            let mut b_reg = [0.0f32; TN as usize];

            for ii in 0..8usize {
                a_reg[ii] = tile_a[(row_off + ii) * 8 + kk];
            }
            for jj in 0..8usize {
                b_reg[jj] = tile_b[(col_off + jj) * 8 + kk];
            }

            // 64 FMAs per k step — 512 FMAs per BK-tile per thread.
            unroll! { for ii in 0..8 {
                let ai = a_reg[ii];
                unroll! { for jj in 0..8 {
                    acc[ii][jj] += ai * b_reg[jj];
                }}
            }}
        }}

        sync_threads();
        tstep += 1;
    }

    // ---- Epilogue: fused bias + multiplier + leaky_relu, then store.
    let mut bias_reg = [0.0f32; TN as usize];
    unroll! { for j in 0..8 {
        bias_reg[j] = bias[(bn + tx * TN) as usize + j];
    }}

    unroll! { for i in 0..8 {
        unroll! { for j in 0..8 {
            let mut v = (acc[i][j] + bias_reg[j]) * multiplier;
            if v < 0.0 {
                v *= negative_slope;
            }
            // Shape is [i0=j, i1=i]; access order is (i0, i1).
            y_thread[(j as u32, i as u32)] = v;
        }}
    }}
}

pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &Path,
    out_dir: &Path,
    iters: usize,
    shape: &[usize],
) -> (f64, f64) {
    assert_eq!(shape.len(), 3, "gemm_mul_lrelu: shape=[M, K, N]");
    let (m, k, n) = (shape[0], shape[1], shape[2]);

    assert!(
        m % BM as usize == 0 && n % BN as usize == 0 && k % BK as usize == 0,
        "M, N, K must be multiples of {}, {}, {} respectively",
        BM,
        BN,
        BK
    );

    let h_x = crate::read_bin(&in_dir.join("x.bin"), m * k);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), n * k);
    let h_b = crate::read_bin(&in_dir.join("b.bin"), n);
    let mut h_y = vec![0f32; m * n];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let d_x4 = unsafe { &*(&d_x as *const _ as *const gpu_host::TensorView<'_, [Float4]>) };
    let d_w4 = unsafe { &*(&d_w as *const _ as *const gpu_host::TensorView<'_, [Float4]>) };

    let mm = m as u32;
    let nn = n as u32;
    let kk = k as u32;
    let multiplier: f32 = 2.0;
    let negative_slope: f32 = 0.1;

    // grid: (N/BN, M/BM). block: (16, 16).
    let gx: u32 = nn / BN;
    let gy: u32 = mm / BM;

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_mul_lrelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, multiplier, negative_slope,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    // Warmup (timed for reporting).
    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_mul_lrelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, multiplier, negative_slope,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    // Timed iterations.
    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_mul_lrelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, multiplier, negative_slope,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // Golden Rule #7: readback is NOT automatic.
    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
