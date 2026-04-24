//! 53_Gemm_Scaling_Hardtanh_GELU — fused nn.Linear + scaling + hardtanh + GELU.
//!
//! PyTorch reference:
//!     y = gemm(x)                 # x @ W.T + b
//!     y = y * 0.5                 # scaling_factor
//!     y = F.hardtanh(y, -2.0, 2.0)
//!     y = F.gelu(y)               # exact form: 0.5 * y * (1 + erf(y / sqrt(2)))
//!
//! Shapes:
//!     x: [M, K]  W: [N, K]  b: [N]  y: [M, N]   (M=2048, K=N=8192)
//!
//! Strategy: 16×16 shared-memory tiled GEMM with the scaling+hardtanh+GELU
//! epilogue fused into the per-thread output write. Mirrors
//! `gemm_mul_lrelu.rs`.
//!
//! Epilogue note: SeGuRu's `GPUDeviceFloatIntrinsics` exposes `tanh` but not
//! `erf` (see crates/gpu/src/device_intrinsic.rs). We therefore use the tanh
//! approximation of GELU:
//!     gelu(v) ≈ 0.5 * v * (1 + tanh(√(2/π) * (v + 0.044715 * v^3)))
//! which matches PyTorch's exact `F.gelu` output to within ~1e-4 after the
//! hardtanh clamp (|v| ≤ 2). The compare.py tolerance (5e-3) absorbs this.
//!
//! Scalars (scaling=0.5, hmin=-2.0, hmax=2.0) are hard-coded as module
//! constants; `main.rs` does not pass them to `run`.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BM: u32 = 16;
const BN: u32 = 16;
const BK: u32 = 16;

// Fused-epilogue constants (the problem spec hard-codes these).
const SCALING: f32 = 0.5;
const HMIN: f32 = -2.0;
const HMAX: f32 = 2.0;

// Tanh-approximation GELU constants.
const GELU_K0: f32 = 0.7978845608028654; // sqrt(2/pi)
const GELU_K1: f32 = 0.044715;

#[gpu::cuda_kernel]
pub fn gemm_scale_htanh_gelu_kernel(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    M: u32,
    N: u32,
    K: u32,
) {
    let mut y_chunk = chunk_mut(y, Map2D::new(N as usize));

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let col = block_id::<DimX>() * BN + tx; // n-index
    let row = block_id::<DimY>() * BM + ty; // m-index

    let mut tile_a = gpu::GpuShared::<[f32; (BM * BK) as usize]>::zero();
    let mut tile_b = gpu::GpuShared::<[f32; (BN * BK) as usize]>::zero();

    let load_map = reshape_map!([1] | [16, 16] => layout: [i0, t0, t1]);

    let mut acc = 0.0f32;
    let num_tiles = K / BK;

    let mut t: u32 = 0;
    while t < num_tiles {
        let k_base = t * BK;

        {
            let mut ca = tile_a.chunk_mut(load_map);
            let a_col = k_base + tx;
            ca[0] = if row < M && a_col < K {
                x[(row * K + a_col) as usize]
            } else {
                0.0
            };
        }
        {
            let mut cb = tile_b.chunk_mut(load_map);
            let n_row = block_id::<DimX>() * BN + ty;
            let b_col = k_base + tx;
            cb[0] = if n_row < N && b_col < K {
                w[(n_row * K + b_col) as usize]
            } else {
                0.0
            };
        }

        sync_threads();

        let mut k: u32 = 0;
        while k < BK {
            acc += tile_a[(ty * BK + k) as usize] * tile_b[(tx * BK + k) as usize];
            k += 1;
        }

        sync_threads();
        t += 1;
    }

    // Fused epilogue: bias add, scaling, hardtanh, tanh-approx GELU.
    if row < M && col < N {
        let b_val = bias[col as usize];
        let mut v = (acc + b_val) * SCALING;
        // hardtanh clamp to [HMIN, HMAX].
        if v < HMIN {
            v = HMIN;
        }
        if v > HMAX {
            v = HMAX;
        }
        // tanh-approx GELU.
        let inner = GELU_K0 * (v + GELU_K1 * v * v * v);
        let th = GPUDeviceFloatIntrinsics::tanh(inner);
        y_chunk[(0, 0)] = 0.5 * v * (1.0 + th);
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
    assert_eq!(shape.len(), 3, "gemm_scale_htanh_gelu: shape=[M, K, N]");
    let (m, k, n) = (shape[0], shape[1], shape[2]);

    let h_x = crate::read_bin(&in_dir.join("x.bin"), m * k);
    let h_w = crate::read_bin(&in_dir.join("W.bin"), n * k);
    let h_b = crate::read_bin(&in_dir.join("b.bin"), n);
    let mut h_y = vec![0f32; m * n];

    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let d_w = ctx.new_tensor_view(h_w.as_slice()).unwrap();
    let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();

    let mm = m as u32;
    let nn = n as u32;
    let kk = k as u32;

    let gx: u32 = nn.div_ceil(BN);
    let gy: u32 = mm.div_ceil(BM);

    // Priming launch (compilation + first-call overhead) — not counted.
    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        gemm_scale_htanh_gelu_kernel::launch(
            cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk,
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
