//! 76_Gemm_Add_ReLU — SeGuRu port of cuda/gemm_add_relu.cu.
//! Epilogue: y = relu(acc + b). Linear has bias=False in reference,
//! but the extra bias tensor has shape [N] so it folds into the GEMM bias.
//!
//! Shapes: x [M,K], W [N,K], b [N], y [M,N]. M=1024, K=N=8192.

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

#[gpu::cuda_kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn gemm_add_relu_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[Float4],
    y: &mut [Float4],
    M: u32,
    N: u32,
    K: u32,
) {
    let _ = M;

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

    // K in Float4 units; scale per-row stride and per-iteration step.
    let k4 = K >> 2;
    let a_col4 = a_col >> 2;

    // K-major shared layout for BM/BN=128: offset = k_lane * 128 + row_or_col.
    let load_map = reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0]);

    let mut acc = [[0.0f32; TN as usize]; TM as usize];

    let num_tiles = K / BK;
    let mut tstep: u32 = 0;
    while tstep < num_tiles {
        let k_base4 = tstep * (BK >> 2);

        {
            let mut ca = tile_a.chunk_mut(load_map);
            let idx_x = ((bm + a_row) * k4 + k_base4 + a_col4) as usize;
            let v: Float4 = x[idx_x];
            ca[0] = v[0];
            ca[1] = v[1];
            ca[2] = v[2];
            ca[3] = v[3];
        }
        {
            let mut cb = tile_b.chunk_mut(load_map);
            let idx_w = ((bn + a_row) * k4 + k_base4 + a_col4) as usize;
            let v: Float4 = w[idx_w];
            cb[0] = v[0];
            cb[1] = v[1];
            cb[2] = v[2];
            cb[3] = v[3];
        }

        sync_threads();

        let row_off = (ty * TM) as usize;
        let col_off = (tx * TN) as usize;

        unroll! { for kk in 0..8 {
            let mut a_reg = [0.0f32; TM as usize];
            let mut b_reg = [0.0f32; TN as usize];

            for ii in 0..8usize {
                a_reg[ii] = tile_a[kk * BM as usize + row_off + ii];
            }
            for jj in 0..8usize {
                b_reg[jj] = tile_b[kk * BN as usize + col_off + jj];
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

    // ---- Epilogue: fused bias + relu, written as Float4 (st.global.v4.f32).
    // bias reinterpreted as &[Float4]; each thread needs 2 Float4 of bias.
    let bn4 = bn >> 2;
    let col4_base = bn4 + tx * 2;
    let b0: Float4 = bias[col4_base as usize];
    let b1: Float4 = bias[(col4_base + 1) as usize];

    // y as &mut [Float4] with [M][N/4] row-major. Per-thread chunk is 2 Float4 × 8 rows.
    // stride check: i0=col4 inner (stride 1), t0=tx (stride TN/4=2), t1=bid_x (stride BN/4=32),
    //               i1=row inner (stride N/4), t2=ty (stride TM*N/4), t3=bid_y (stride BM*N/4).
    let _ = N;
    let out_map = reshape_map!(
        [2, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let mut y_tile = chunk_mut(y, out_map);

    unroll! { for i in 0..8 {
        let mut o0 = Float4::default();
        let mut o1 = Float4::default();
        unroll! { for j in 0..4 {
            let mut v = acc[i][j] + b0[j];
            if v < 0.0 { v = 0.0; }
            o0[j] = v;
        }}
        unroll! { for j in 0..4 {
            let mut v = acc[i][j + 4] + b1[j];
            if v < 0.0 { v = 0.0; }
            o1[j] = v;
        }}
        y_tile[(0u32, i as u32)] = o0;
        y_tile[(1u32, i as u32)] = o1;
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
    assert_eq!(shape.len(), 3, "gemm_add_relu: shape=[M, K, N]");
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

    // Reinterpret X and W as &[Float4] device views to emit ld.global.v4.f32.
    // K must be a multiple of 4 (asserted above: k % BK == 0, BK=8).
    let d_x4_view = d_x
        .try_cast_slice::<Float4>()
        .expect("x Float4 view requires 16-byte alignment and length divisible by 4");
    let d_w4_view = d_w
        .try_cast_slice::<Float4>()
        .expect("w Float4 view requires 16-byte alignment and length divisible by 4");
    let d_b4_view = d_b
        .try_cast_slice::<Float4>()
        .expect("bias Float4 view requires 16-byte alignment and length divisible by 4");
    let d_x4 = &d_x4_view;
    let d_w4 = &d_w4_view;
    let d_b4 = &d_b4_view;

    let mm = m as u32;
    let nn = n as u32;
    let kk = k as u32;

    let gx: u32 = nn / BN;
    let gy: u32 = mm / BM;

    {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        let mut d_y4_view = d_y
            .try_cast_slice_mut::<Float4>()
            .expect("y Float4 view requires 16-byte alignment and length divisible by 4");
        let d_y4 = &mut d_y4_view;
        gemm_add_relu_kernel::launch(cfg, ctx, md, d_x4, d_w4, d_b4, d_y4, mm, nn, kk).unwrap();
    }
    ctx.sync().unwrap();

    let warmup_iters: usize = 5;
    let wt = Instant::now();
    for _ in 0..warmup_iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        let mut d_y4_view = d_y
            .try_cast_slice_mut::<Float4>()
            .expect("y Float4 view requires 16-byte alignment and length divisible by 4");
        let d_y4 = &mut d_y4_view;
        gemm_add_relu_kernel::launch(cfg, ctx, md, d_x4, d_w4, d_b4, d_y4, mm, nn, kk).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / warmup_iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
        let mut d_y4_view = d_y
            .try_cast_slice_mut::<Float4>()
            .expect("y Float4 view requires 16-byte alignment and length divisible by 4");
        let d_y4 = &mut d_y4_view;
        gemm_add_relu_kernel::launch(cfg, ctx, md, d_x4, d_w4, d_b4, d_y4, mm, nn, kk).unwrap();
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
