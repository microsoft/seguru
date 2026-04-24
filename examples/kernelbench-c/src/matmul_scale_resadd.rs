//! 40_Matmul_Scaling_ResidualAdd — SeGuRu port of PyTorch reference
//! `examples/kernelbench-c/problems/40_Matmul_Scaling_ResidualAdd.py`.
//!
//! Regenerated against `docs/cuda-to-seguru-porting-skill.md` @ bf493b79
//! (Phase D.1). GEMM Recipe (128×128×8 register-tiled). PyTorch computes
//! `y = (x@W.T+b) * scale + (x@W.T+b)`; algebraically this is
//! `(x@W.T+b) * (1+scale)`, fused as a single scalar multiply in the
//! epilogue.
//!
//! 40_Matmul_Scaling_ResidualAdd — fused nn.Linear + scale + residual-add.
//!
//! PyTorch reference:
//!     y0        = x @ W.T + b
//!     original  = y0.clone().detach()
//!     y         = y0 * scaling_factor + original           // == y0 * (1 + s)
//!
//! Fused form:  y[m, n] = (Σ_k x[m,k] · W[n,k] + b[n]) * (1 + scaling).
//!
//! Tile config (per docs/cuda-to-seguru-porting-skill.md "GEMM / Matmul Recipe"):
//!   BM=128, BN=128, BK=8, TM=TN=8, block = 16×16 = 256 threads,
//!   non-transposed shared-memory tiles, 8×8 register output tile.

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

const SCALING: f32 = 0.5;

#[gpu::cuda_kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn matmul_scale_resadd_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[f32],
    y: &mut [f32],
    M: u32,
    N: u32,
    K: u32,
    scale1p: f32,
) {
    let _ = M;
    let _ = N;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bid_x = block_id::<DimX>();
    let bid_y = block_id::<DimY>();

    let bm = bid_y * BM;
    let bn = bid_x * BN;

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

    // Fused epilogue: bias + (1 + scaling) scale.
    let mut bias_reg = [0.0f32; TN as usize];
    unroll! { for j in 0..8 {
        bias_reg[j] = bias[(bn + tx * TN) as usize + j];
    }}

    unroll! { for i in 0..8 {
        unroll! { for j in 0..8 {
            let v = (acc[i][j] + bias_reg[j]) * scale1p;
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
    assert_eq!(shape.len(), 3, "matmul_scale_resadd: shape=[M, K, N]");
    let (m, k, n) = (shape[0], shape[1], shape[2]);

    assert!(
        m % BM as usize == 0 && n % BN as usize == 0 && k % BK as usize == 0,
        "M, N, K must be multiples of {}, {}, {} respectively",
        BM,
        BN,
        BK
    );

    let h_x = super::read_bin(&in_dir.join("x.bin"), m * k);
    let h_w = super::read_bin(&in_dir.join("W.bin"), n * k);
    let h_b = super::read_bin(&in_dir.join("b.bin"), n);
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
    let scale1p: f32 = 1.0 + SCALING;

    let gx: u32 = nn / BN;
    let gy: u32 = mm / BM;

    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        matmul_scale_resadd_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, scale1p,
        )
        .unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        matmul_scale_resadd_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, scale1p,
        )
        .unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        matmul_scale_resadd_kernel::launch(
            cfg, ctx, md, d_x4, d_w4, &d_b, &mut d_y, mm, nn, kk, scale1p,
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

    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
