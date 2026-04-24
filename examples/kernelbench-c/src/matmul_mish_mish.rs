//! 29_Matmul_Mish_Mish — fused nn.Linear + mish + mish.
//!
//! PyTorch reference:
//!     y = F.mish(F.mish(x @ W.T + b))
//!     where mish(v) = v * tanh(softplus(v)) = v * tanh(log(1 + exp(v)))
//!
//! Shapes:
//!     x: [M, K]
//!     W: [N, K]  (nn.Linear layout; y[m,n] = Σ_k x[m,k]·W[n,k])
//!     b: [N]
//!     y: [M, N]
//!
//! Strategy: 16×16 shared-memory tiled GEMM with the bias + mish(mish(.))
//! epilogue fused into the per-thread output write.

use std::path::Path;
use std::time::Instant;

use gpu::prelude::*;

const BM: u32 = 16;
const BN: u32 = 16;
const BK: u32 = 16;

#[inline]
fn mish(v: f32) -> f32 {
    // Numerically-safe softplus: for large v, log1p(exp(v)) ~= v.
    let sp = if v > 20.0 {
        v
    } else {
        GPUDeviceFloatIntrinsics::log1p(GPUDeviceFloatIntrinsics::exp(v))
    };
    v * GPUDeviceFloatIntrinsics::tanh(sp)
}

#[gpu::cuda_kernel]
pub fn matmul_mish_mish_kernel(
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
    let col = block_id::<DimX>() * BN + tx;
    let row = block_id::<DimY>() * BM + ty;

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

    if row < M && col < N {
        let b_val = bias[col as usize];
        let v0 = acc + b_val;
        let v1 = mish(v0);
        let v2 = mish(v1);
        y_chunk[(0, 0)] = v2;
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
    assert_eq!(shape.len(), 3, "matmul_mish_mish: shape=[M, K, N]");
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

    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        matmul_mish_mish_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
            .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        matmul_mish_mish_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
            .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BN, BM, 1, 0);
        matmul_mish_mish_kernel::launch(cfg, ctx, md, &d_x, &d_w, &d_b, &mut d_y, mm, nn, kk)
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
