//! 53_Gemm_Scaling_Hardtanh_GELU — SeGuRu port of PyTorch reference
//! `examples/kernelbench-c/problems/53_Gemm_Scaling_Hardtanh_GELU.py`.
//!
//! Regenerated against `docs/cuda-to-seguru-porting-skill.md` @ bf493b79
//! (Phase D.1). GEMM Recipe (128×128×8 register-tiled) with fused
//! epilogue: scale → hardtanh(min,max) → tanh-approx GELU.
//!
//! 53_Gemm_Scaling_Hardtanh_GELU — fused nn.Linear + scaling + hardtanh + GELU.
//!
//! PyTorch reference:
//!     y = gemm(x)                 # x @ W.T + b
//!     y = y * 0.5                 # scaling_factor
//!     y = F.hardtanh(y, -2.0, 2.0)
//!     y = F.gelu(y)
//!
//! Shapes:
//!     x: [M, K]  W: [N, K]  b: [N]  y: [M, N]   (M=2048, K=N=8192)
//!
//! Strategy (from docs/cuda-to-seguru-porting-skill.md — "GEMM / Matmul
//! Recipe"): register-tiled SGEMM with BM=BN=128, BK=8, TM=TN=8, block=16×16
//! = 256 threads. Each thread owns an 8×8 output sub-tile. Non-transposed
//! smem layout (As[m][k], Bs[n][k]) so each thread writes 4 contiguous slots
//! per tile and avoids store-side bounds checks. Epilogue (bias + scaling +
//! hardtanh + tanh-approx GELU) is fused into the per-thread output write.
//! Structurally mirrors `from_cuda/gemm_scale_htanh_gelu.rs`.
//!
//! Epilogue note: `GPUDeviceFloatIntrinsics` has `tanh` but not `erf`, so the
//! tanh approximation of GELU is used. Matches PyTorch's exact `F.gelu` to
//! within compare.py atol (5e-3) for |v| ≤ 2 after hardtanh.
//!
//! Scalars (scaling=0.5, hmin=-2.0, hmax=2.0) are hard-coded as module
//! constants; `main.rs` does not pass them to `run`.

use std::path::Path;
use std::time::Instant;

use crunchy::unroll;
use gpu::prelude::*;

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 8;
const TM: u32 = 8;
const TN: u32 = 8;
const BDIM_X: u32 = 16;
const BDIM_Y: u32 = 16;

// Fused-epilogue constants (the problem spec hard-codes these).
const SCALING: f32 = 0.5;
const HMIN: f32 = -2.0;
const HMAX: f32 = 2.0;

// Tanh-approximation GELU constants.
const GELU_K0: f32 = 0.7978845608028654; // sqrt(2/pi)
const GELU_K1: f32 = 0.044715;

#[gpu::cuda_kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn gemm_scale_htanh_gelu_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[f32],
    y: &mut [f32],
    M: u32,
    N: u32,
    K: u32,
) {
    let _ = M;
    let _ = N;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bid_x = block_id::<DimX>();
    let bid_y = block_id::<DimY>();

    let bm = bid_y * BM;
    let bn = bid_x * BN;

    // 256 threads cooperatively load a 128×8 tile = 1024 f32 → 4 per thread.
    // With BK=8, (a_row, a_col) maps tid to a contiguous 4-slot group in a
    // row of the non-transposed smem tile.
    let tid = ty * BDIM_X + tx;
    let a_row = tid >> 1;
    let a_col = (tid & 1) << 2;

    let mut tile_a = gpu::GpuShared::<[f32; (BM * BK) as usize]>::zero();
    let mut tile_b = gpu::GpuShared::<[f32; (BN * BK) as usize]>::zero();

    let load_map = reshape_map!([4] | [16, 16] => layout: [i0, t0, t1]);

    let out_map = reshape_map!(
        [8, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let mut y_thread = chunk_mut(y, out_map);

    let mut acc = [[0.0f32; TN as usize]; TM as usize];

    let num_tiles = K / BK;
    let mut tstep: u32 = 0;
    while tstep < num_tiles {
        let k_base4 = tstep * (BK >> 2);

        // LOAD — contiguous 4-element writes to smem, no bounds check.
        {
            let mut ca = tile_a.chunk_mut(load_map);
            let v: Float4 = x[((bm + a_row) * (K >> 2) + k_base4 + (a_col >> 2)) as usize];
            ca[0] = v[0]; ca[1] = v[1]; ca[2] = v[2]; ca[3] = v[3];
        }
        {
            let mut cb = tile_b.chunk_mut(load_map);
            let v: Float4 = w[((bn + a_row) * (K >> 2) + k_base4 + (a_col >> 2)) as usize];
            cb[0] = v[0]; cb[1] = v[1]; cb[2] = v[2]; cb[3] = v[3];
        }

        sync_threads();

        // COMPUTE — 64 FMAs per k iteration amortise broadcast-read checks.
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

    // ---- Epilogue: bias + scaling + hardtanh + tanh-approx GELU.
    let mut bias_reg = [0.0f32; TN as usize];
    unroll! { for j in 0..8 {
        bias_reg[j] = bias[(bn + tx * TN) as usize + j];
    }}

    unroll! { for i in 0..8 {
        unroll! { for j in 0..8 {
            let mut v = (acc[i][j] + bias_reg[j]) * SCALING;
            if v < HMIN { v = HMIN; }
            if v > HMAX { v = HMAX; }
            let inner = GELU_K0 * (v + GELU_K1 * v * v * v);
            let th = GPUDeviceFloatIntrinsics::tanh(inner);
            let out = 0.5 * v * (1.0 + th);
            y_thread[(j as u32, i as u32)] = out;
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
    assert_eq!(shape.len(), 3, "gemm_scale_htanh_gelu: shape=[M, K, N]");
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

    let d_x4_view = d_x
        .try_cast_slice::<Float4>()
        .expect("x Float4 view requires 16-byte alignment and length divisible by 4");
    let d_w4_view = d_w
        .try_cast_slice::<Float4>()
        .expect("w Float4 view requires 16-byte alignment and length divisible by 4");
    let d_x4 = &d_x4_view;
    let d_w4 = &d_w4_view;

    let mm = m as u32;
    let nn = n as u32;
    let kk = k as u32;

    let gx: u32 = nn / BN;
    let gy: u32 = mm / BM;

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    d_y.copy_to_host(&mut h_y).unwrap();
    drop(d_y);
    drop(d_b);
    drop(d_w);
    drop(d_x);

    crate::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
