use gpu::prelude::*;
use gpu::CacheStreamLoadStore;
use std::time::Instant;

// =====================================================================
// Kernel definitions — all 19 PolybenchGPU benchmarks
// =====================================================================

// --- conv2d ---
#[gpu::cuda_kernel]
pub fn bench_conv2d(a: &[f32], b: &mut [f32], ni: u32, nj: u32) {
    let mut b = chunk_mut(b, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let c11: f32 = 0.2;
    let c21: f32 = 0.5;
    let c31: f32 = -0.8;
    let c12: f32 = -0.3;
    let c22: f32 = 0.6;
    let c32: f32 = -0.9;
    let c13: f32 = 0.4;
    let c23: f32 = 0.7;
    let c33: f32 = 0.10;
    if i > 0 && i < ni - 1 && j > 0 && j < nj - 1 {
        b[0] = c11 * a[((i - 1) * nj + (j - 1)) as usize]
            + c21 * a[((i - 1) * nj + j) as usize]
            + c31 * a[((i - 1) * nj + (j + 1)) as usize]
            + c12 * a[(i * nj + (j - 1)) as usize]
            + c22 * a[(i * nj + j) as usize]
            + c32 * a[(i * nj + (j + 1)) as usize]
            + c13 * a[((i + 1) * nj + (j - 1)) as usize]
            + c23 * a[((i + 1) * nj + j) as usize]
            + c33 * a[((i + 1) * nj + (j + 1)) as usize];
    }
}

// --- conv3d ---
#[gpu::cuda_kernel]
pub fn bench_conv3d(a: &[f32], b: &mut [f32], ni: u32, nj: u32, nk: u32) {
    let mut b = chunk_mut(b, MapContinuousLinear::new(1));
    let k = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let ij = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let i = ij / nj;
    let j = ij % nj;
    if i > 0 && i < ni - 1 && j > 0 && j < nj - 1 && k > 0 && k < nk - 1 {
        let c11: f32 = 2.0;
        let c12: f32 = 5.0;
        let c13: f32 = -8.0;
        let c21: f32 = -3.0;
        let c22: f32 = 6.0;
        let c23: f32 = -9.0;
        let c31: f32 = 4.0;
        let c32: f32 = 7.0;
        let c33: f32 = 10.0;
        let s = nj * nk;
        b[0] = c11 * a[((i - 1) * s + (j - 1) * nk + (k - 1)) as usize]
            + c12 * a[((i - 1) * s + (j - 1) * nk + k) as usize]
            + c13 * a[((i - 1) * s + (j - 1) * nk + (k + 1)) as usize]
            + c21 * a[((i - 1) * s + j * nk + (k - 1)) as usize]
            + c22 * a[((i - 1) * s + j * nk + k) as usize]
            + c23 * a[((i - 1) * s + j * nk + (k + 1)) as usize]
            + c31 * a[((i - 1) * s + (j + 1) * nk + (k - 1)) as usize]
            + c32 * a[((i - 1) * s + (j + 1) * nk + k) as usize]
            + c33 * a[((i - 1) * s + (j + 1) * nk + (k + 1)) as usize]
            + c11 * a[(i * s + (j - 1) * nk + (k - 1)) as usize]
            + c12 * a[(i * s + (j - 1) * nk + k) as usize]
            + c13 * a[(i * s + (j - 1) * nk + (k + 1)) as usize]
            + c21 * a[(i * s + j * nk + (k - 1)) as usize]
            + c22 * a[(i * s + j * nk + k) as usize]
            + c23 * a[(i * s + j * nk + (k + 1)) as usize]
            + c31 * a[(i * s + (j + 1) * nk + (k - 1)) as usize]
            + c32 * a[(i * s + (j + 1) * nk + k) as usize]
            + c33 * a[(i * s + (j + 1) * nk + (k + 1)) as usize]
            + c11 * a[((i + 1) * s + (j - 1) * nk + (k - 1)) as usize]
            + c12 * a[((i + 1) * s + (j - 1) * nk + k) as usize]
            + c13 * a[((i + 1) * s + (j - 1) * nk + (k + 1)) as usize]
            + c21 * a[((i + 1) * s + j * nk + (k - 1)) as usize]
            + c22 * a[((i + 1) * s + j * nk + k) as usize]
            + c23 * a[((i + 1) * s + j * nk + (k + 1)) as usize]
            + c31 * a[((i + 1) * s + (j + 1) * nk + (k - 1)) as usize]
            + c32 * a[((i + 1) * s + (j + 1) * nk + k) as usize]
            + c33 * a[((i + 1) * s + (j + 1) * nk + (k + 1)) as usize];
    }
}

// --- gemm ---
#[gpu::cuda_kernel]
pub fn bench_gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j < nj {
        let mut val = c[(0, 0)] * beta;
        let a_row: &[f32] = &a[(i * nk) as usize..((i + 1) * nk) as usize];
        let mut b_idx = j as usize;
        for a_val in a_row {
            val += alpha * a_val * b[b_idx];
            b_idx += nj as usize;
        }
        c[(0, 0)] = val;
    }
}

// --- twomm ---
#[gpu::cuda_kernel]
pub fn bench_mm2_kernel1(
    a: &[f32],
    b: &[f32],
    tmp: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
    alpha: f32,
) {
    let mut tmp = chunk_mut(tmp, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j < nj {
        let mut val = 0.0f32;
        let a_row: &[f32] = &a[(i * nk) as usize..((i + 1) * nk) as usize];
        let mut b_idx = j as usize;
        for a_val in a_row {
            val += alpha * a_val * b[b_idx];
            b_idx += nj as usize;
        }
        tmp[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm2_kernel2(
    tmp: &[f32],
    c: &[f32],
    d: &mut [f32],
    ni: u32,
    nj: u32,
    nl: u32,
    beta: f32,
) {
    let mut d = chunk_mut(d, Map2D::new(nl as usize));
    let l = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && l < nl {
        let mut val = d[(0, 0)] * beta;
        let tmp_row: &[f32] = &tmp[(i * nj) as usize..((i + 1) * nj) as usize];
        let mut c_idx = l as usize;
        for t_val in tmp_row {
            val += t_val * c[c_idx];
            c_idx += nl as usize;
        }
        d[(0, 0)] = val;
    }
}

// --- threemm ---
#[gpu::cuda_kernel]
pub fn bench_mm3_kernel1(
    a: &[f32],
    b: &[f32],
    e: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
) {
    let mut e = chunk_mut(e, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j < nj {
        let mut val = 0.0f32;
        let a_row: &[f32] = &a[(i * nk) as usize..((i + 1) * nk) as usize];
        let mut b_idx = j as usize;
        for a_val in a_row {
            val += a_val * b[b_idx];
            b_idx += nj as usize;
        }
        e[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm3_kernel2(
    c: &[f32],
    d: &[f32],
    f: &mut [f32],
    nj: u32,
    nl: u32,
    nm: u32,
) {
    let mut f = chunk_mut(f, Map2D::new(nl as usize));
    let l = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let j = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j < nj && l < nl {
        let mut val = 0.0f32;
        let c_row: &[f32] = &c[(j * nm) as usize..((j + 1) * nm) as usize];
        let mut d_idx = l as usize;
        for c_val in c_row {
            val += c_val * d[d_idx];
            d_idx += nl as usize;
        }
        f[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm3_kernel3(
    e: &[f32],
    f: &[f32],
    g: &mut [f32],
    ni: u32,
    nj: u32,
    nl: u32,
) {
    let mut g = chunk_mut(g, Map2D::new(nl as usize));
    let l = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && l < nl {
        let mut val = 0.0f32;
        let e_row: &[f32] = &e[(i * nj) as usize..((i + 1) * nj) as usize];
        let mut f_idx = l as usize;
        for e_val in e_row {
            val += e_val * f[f_idx];
            f_idx += nl as usize;
        }
        g[(0, 0)] = val;
    }
}

// --- atax ---
#[gpu::cuda_kernel]
pub fn bench_atax_kernel1(a: &[f32], x: &[f32], tmp: &mut [f32], nx: u32, ny: u32) {
    let mut tmp = chunk_mut(tmp, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < nx {
        let mut sum = 0.0f32;
        let a_row: &[f32] = &a[(i * ny) as usize..((i + 1) * ny) as usize];
        let mut j_idx: usize = 0;
        while j_idx < ny as usize {
            sum += a_row[j_idx] * x[j_idx];
            j_idx += 1;
        }
        tmp[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_atax_kernel2(a: &[f32], tmp: &[f32], y: &mut [f32], nx: u32, ny: u32) {
    let mut y = chunk_mut(y, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < nx {
            sum += a[(i * ny + j) as usize] * tmp[i as usize];
            i += 1;
        }
        y[0] = sum;
    }
}

// --- bicg ---
#[gpu::cuda_kernel]
pub fn bench_bicg_kernel1(a: &[f32], r: &[f32], s: &mut [f32], nx: u32, ny: u32) {
    let mut s = chunk_mut(s, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < nx {
            sum += r[i as usize] * a[(i * ny + j) as usize];
            i += 1;
        }
        s[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_bicg_kernel2(a: &[f32], p: &[f32], q: &mut [f32], nx: u32, ny: u32) {
    let mut q = chunk_mut(q, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < nx {
        let mut sum = 0.0f32;
        let a_row: &[f32] = &a[(i * ny) as usize..((i + 1) * ny) as usize];
        let mut j_idx: usize = 0;
        while j_idx < ny as usize {
            sum += a_row[j_idx] * p[j_idx];
            j_idx += 1;
        }
        q[0] = sum;
    }
}

// --- mvt ---
#[gpu::cuda_kernel]
pub fn bench_mvt_kernel1(a: &[f32], x1: &mut [f32], y1: &[f32], n: u32) {
    let mut x1 = chunk_mut(x1, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < n {
        let mut sum = x1[0];
        let a_row: &[f32] = &a[(i * n) as usize..((i + 1) * n) as usize];
        let mut j_idx: usize = 0;
        while j_idx < n as usize {
            sum += a_row[j_idx] * y1[j_idx];
            j_idx += 1;
        }
        x1[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mvt_kernel2(a: &[f32], x2: &mut [f32], y2: &[f32], n: u32) {
    let mut x2 = chunk_mut(x2, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < n {
        let mut sum = x2[0];
        let mut j: u32 = 0;
        while j < n {
            sum += a[(j * n + i) as usize] * y2[j as usize];
            j += 1;
        }
        x2[0] = sum;
    }
}

// --- gesummv ---
#[gpu::cuda_kernel]
pub fn bench_gesummv(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: u32,
    alpha: f32,
    beta: f32,
) {
    let mut y = chunk_mut(y, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < n {
        let mut sum_a = 0.0f32;
        let mut sum_b = 0.0f32;
        let a_row: &[f32] = &a[(i * n) as usize..((i + 1) * n) as usize];
        let b_row: &[f32] = &b[(i * n) as usize..((i + 1) * n) as usize];
        let mut j_idx: usize = 0;
        while j_idx < n as usize {
            sum_a += a_row[j_idx] * x[j_idx];
            sum_b += b_row[j_idx] * x[j_idx];
            j_idx += 1;
        }
        y[0] = alpha * sum_a + beta * sum_b;
    }
}

// --- syr2k ---
#[gpu::cuda_kernel]
pub fn bench_syr2k(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: u32,
    nj: u32,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(ni as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j <= i {
        let mut val = c[(0, 0)] * beta;
        let a_row_i: &[f32] = &a[(i * nj) as usize..((i + 1) * nj) as usize];
        let b_row_j: &[f32] = &b[(j * nj) as usize..((j + 1) * nj) as usize];
        let b_row_i: &[f32] = &b[(i * nj) as usize..((i + 1) * nj) as usize];
        let a_row_j: &[f32] = &a[(j * nj) as usize..((j + 1) * nj) as usize];
        for k in 0..nj as usize {
            val += alpha * a_row_i[k] * b_row_j[k] + alpha * b_row_i[k] * a_row_j[k];
        }
        c[(0, 0)] = val;
    }
}

// --- syrk ---
#[gpu::cuda_kernel]
pub fn bench_syrk(
    a: &[f32],
    c: &mut [f32],
    ni: u32,
    nj: u32,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(ni as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j <= i {
        let mut val = c[(0, 0)] * beta;
        let a_row_i: &[f32] = &a[(i * nj) as usize..((i + 1) * nj) as usize];
        let a_row_j: &[f32] = &a[(j * nj) as usize..((j + 1) * nj) as usize];
        for k in 0..nj as usize {
            val += alpha * a_row_i[k] * a_row_j[k];
        }
        c[(0, 0)] = val;
    }
}

// --- corr ---
const CORR_FLOAT_N: f32 = 3214212.01;
const CORR_EPS: f32 = 0.005;

#[gpu::cuda_kernel]
pub fn bench_corr_mean(data: &[f32], mean: &mut [f32], m: u32, n: u32) {
    let mut mean = chunk_mut(mean, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j) as usize].ldcs();
            i += 1;
        }
        mean[0] = sum / CORR_FLOAT_N;
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_std(data: &[f32], mean: &[f32], stddev: &mut [f32], m: u32, n: u32) {
    let mut stddev = chunk_mut(stddev, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < m {
        let mut sum = 0.0f32;
        let mean_j = mean[j as usize];
        let mut i: u32 = 0;
        while i < n {
            let diff = data[(i * m + j) as usize].ldcs() - mean_j;
            sum += diff * diff;
            i += 1;
        }
        sum /= CORR_FLOAT_N;
        let s = sum.sqrt();
        stddev[0] = if s <= CORR_EPS { 1.0 } else { s };
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_reduce(mean: &[f32], stddev: &[f32], data: &mut [f32], m: u32, n: u32) {
    let mut data = chunk_mut(data, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < n && j < m {
        let val = data[0] - mean[j as usize];
        data[0] = val / (CORR_FLOAT_N.sqrt() * stddev[j as usize]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_corr(data: &[f32], symmat: &mut [f32], m: u32, n: u32) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m as usize));
    let j2 = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let j1 = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j1) as usize].ldcs() * data[(i * m + j2) as usize].ldcs();
            i += 1;
        }
        symmat[(0, 0)] = sum;
    }
}

// --- covar ---
const COVAR_FLOAT_N: f32 = 3214212.01;

#[gpu::cuda_kernel]
pub fn bench_covar_mean(data: &[f32], mean: &mut [f32], m: u32, n: u32) {
    let mut mean = chunk_mut(mean, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j) as usize].ldcs();
            i += 1;
        }
        mean[0] = sum / COVAR_FLOAT_N;
    }
}

#[gpu::cuda_kernel]
pub fn bench_covar_reduce(mean: &[f32], data: &mut [f32], m: u32, n: u32) {
    let mut data = chunk_mut(data, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < n && j < m {
        data[0] = data[0] - mean[j as usize];
    }
}

#[gpu::cuda_kernel]
pub fn bench_covar_covar(data: &[f32], symmat: &mut [f32], m: u32, n: u32) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m as usize));
    let j2 = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let j1 = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j1) as usize].ldcs() * data[(i * m + j2) as usize].ldcs();
            i += 1;
        }
        symmat[(0, 0)] = sum;
    }
}

// --- doitgen ---
#[gpu::cuda_kernel]
pub fn bench_doitgen_kernel1(
    a: &[f32],
    c4: &[f32],
    sum_arr: &mut [f32],
    nr: u32,
    nq: u32,
    np: u32,
) {
    let mut sum_arr = chunk_mut(sum_arr, MapContinuousLinear::new(1));
    let p = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let qr = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        let mut val = 0.0f32;
        let a_row: &[f32] = &a[(r * (nq * np) + q * np) as usize..(r * (nq * np) + q * np + np) as usize];
        let mut c4_idx = p as usize;
        for a_val in a_row {
            val += a_val * c4[c4_idx];
            c4_idx += np as usize;
        }
        sum_arr[0] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_doitgen_kernel2(
    sum_arr: &[f32],
    a: &mut [f32],
    nr: u32,
    nq: u32,
    np: u32,
) {
    let mut a = chunk_mut(a, MapContinuousLinear::new(1));
    let p = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let qr = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        a[0] = sum_arr[(r * (nq * np) + q * np + p) as usize];
    }
}

// --- fdtd2d ---
#[gpu::cuda_kernel]
pub fn bench_fdtd_step1(
    fict: &[f32],
    ey: &mut [f32],
    hz: &[f32],
    nx: u32,
    ny: u32,
    t: u32,
) {
    let mut ey = chunk_mut(ey, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx && j < ny {
        if i == 0 {
            ey[0] = fict[t as usize];
        } else {
            ey[0] = ey[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[((i - 1) * ny + j) as usize]);
        }
    }
}

#[gpu::cuda_kernel]
pub fn bench_fdtd_step2(ex: &mut [f32], hz: &[f32], nx: u32, ny: u32) {
    let mut ex = chunk_mut(ex, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx && j > 0 && j < ny {
        ex[0] = ex[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[(i * ny + (j - 1)) as usize]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_fdtd_step3(ex: &[f32], ey: &[f32], hz: &mut [f32], nx: u32, ny: u32) {
    let mut hz = chunk_mut(hz, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx - 1 && j < ny - 1 {
        hz[0] = hz[0]
            - 0.7
                * (ex[(i * ny + (j + 1)) as usize] - ex[(i * ny + j) as usize] + ey[((i + 1) * ny + j) as usize]
                    - ey[(i * ny + j) as usize]);
    }
}

// --- gramschm ---
#[gpu::cuda_kernel]
pub fn bench_gramschm_kernel2(
    a: &[f32],
    r_kk: f32,
    q: &mut [f32],
    nj: u32,
    ni: u32,
    k: u32,
) {
    let mut q = chunk_mut(q, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j == k {
        q[(0, 0)] = a[(i * nj + k) as usize] / r_kk;
    }
}

#[gpu::cuda_kernel]
pub fn bench_gramschm_kernel3a(
    q: &[f32],
    a: &[f32],
    r: &mut [f32],
    ni: u32,
    nj: u32,
    k: u32,
) {
    let mut r = chunk_mut(r, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if row == k && j > k && j < nj {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < ni {
            sum += q[(i * nj + k) as usize] * a[(i * nj + j) as usize];
            i += 1;
        }
        r[(0, 0)] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_gramschm_kernel3b(
    q: &[f32],
    r: &[f32],
    a: &mut [f32],
    ni: u32,
    nj: u32,
    k: u32,
) {
    let mut a = chunk_mut(a, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j > k && j < nj && i < ni {
        a[0] = a[0] - q[(i * nj + k) as usize] * r[(k * nj + j) as usize];
    }
}

// --- jacobi1d ---
#[gpu::cuda_kernel]
pub fn bench_jacobi1d_kernel1(a: &[f32], b: &mut [f32], n: u32) {
    let mut b = chunk_mut(b, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i > 0 && i < n - 1 {
        b[0] = 0.33333 * (a[(i - 1) as usize] + a[i as usize] + a[(i + 1) as usize]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_jacobi1d_kernel2(a: &mut [f32], b: &[f32], n: u32) {
    let mut a = chunk_mut(a, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i > 0 && i < n - 1 {
        a[0] = b[i as usize];
    }
}

// --- jacobi2d ---
#[gpu::cuda_kernel]
pub fn bench_jacobi2d_kernel1(a: &[f32], b: &mut [f32], n: u32) {
    let mut b = chunk_mut(b, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i > 0 && i < n - 1 && j > 0 && j < n - 1 {
        b[(0, 0)] = 0.2
            * (a[(i * n + j) as usize]
                + a[(i * n + (j - 1)) as usize]
                + a[(i * n + (j + 1)) as usize]
                + a[((i + 1) * n + j) as usize]
                + a[((i - 1) * n + j) as usize]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_jacobi2d_kernel2(a: &mut [f32], b: &[f32], n: u32) {
    let mut a = chunk_mut(a, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i > 0 && i < n - 1 && j > 0 && j < n - 1 {
        a[(0, 0)] = b[(i * n + j) as usize];
    }
}

// --- lu ---
#[gpu::cuda_kernel]
pub fn bench_lu_kernel1(pivot: &[f32], row_tail: &mut [f32], rem: u32) {
    let mut row_tail = chunk_mut(row_tail, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < rem {
        row_tail[0] = row_tail[0] / pivot[0];
    }
}

#[gpu::cuda_kernel]
pub fn bench_lu_copy_col(a: &[f32], col: &mut [f32], n: u32, k: u32) {
    let mut col = chunk_mut(col, MapContinuousLinear::new(1));
    let i_tail = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let rem = n - k - 1;
    if i_tail < rem {
        col[0] = a[((i_tail + k + 1) * n + k) as usize];
    }
}

#[gpu::cuda_kernel]
pub fn bench_lu_kernel2(row_tail: &[f32], col: &[f32], rows_below: &mut [f32], n: u32, k: u32) {
    let j_tail = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i_tail = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let rem = n - k - 1;
    let mut rows_below = chunk_mut(
        rows_below,
        reshape_map!([1] | [(rem, n), rem] => layout: [i0, t0, t1], offset: k + 1),
    );
    if i_tail < rem && j_tail < rem {
        rows_below[0] = rows_below[0] - col[i_tail as usize] * row_tail[j_tail as usize];
    }
}

// =====================================================================
// Benchmark runner — all benchmarks inline in cuda_ctx closure
// =====================================================================

fn main() {
    gpu_host::cuda_ctx(0, |ctx, m| {
        // --- conv2d (NI=4096, NJ=4096) ---
        {
            let ni: usize = 4096;
            let nj: usize = 4096;
            let iters = 100;
            let h_a: Vec<f32> = (0..ni * nj).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_b = vec![0.0f32; ni * nj];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let mut d_b = ctx.new_tensor_view(h_b.as_mut_slice()).unwrap();
            let bx: u32 = 32;
            let by: u32 = 8;
            let gx = (nj as u32 + bx - 1) / bx;
            let gy = (ni as u32 + by - 1) / by;
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_conv2d::launch(cfg, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_conv2d::launch(cfg, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("conv2d SeGuRu: {:.3} us/iter (NI={}, NJ={}, {} iters)", us, ni, nj, iters);
        }

        // --- conv3d (NI=NJ=NK=256) ---
        {
            let ni: usize = 256;
            let nj: usize = 256;
            let nk: usize = 256;
            let iters = 100;
            let sz = ni * nj * nk;
            let h_a: Vec<f32> = (0..sz).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_b = vec![0.0f32; sz];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let mut d_b = ctx.new_tensor_view(h_b.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let gx = (nk as u32 + bx - 1) / bx;
            let gy = ((ni * nj) as u32 + by - 1) / by;
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_conv3d::launch(cfg, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32, nk as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_conv3d::launch(cfg, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32, nk as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("conv3d SeGuRu: {:.3} us/iter (NI={}, NJ={}, NK={}, {} iters)", us, ni, nj, nk, iters);
        }

        // --- gemm (NI=NJ=NK=512) ---
        {
            let ni: usize = 512;
            let nj: usize = 512;
            let nk: usize = 512;
            let iters = 100;
            let h_a = vec![1.0f32; ni * nk];
            let h_b = vec![1.0f32; nk * nj];
            let mut h_c = vec![0.0f32; ni * nj];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let gx = (nj as u32 + bx - 1) / bx;
            let gy = (ni as u32 + by - 1) / by;
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_gemm::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, nk as u32, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_gemm::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, nk as u32, 1.0f32, 0.0f32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("gemm SeGuRu: {:.3} us/iter (N={}, {} iters)", us, ni, iters);
        }

        // --- twomm (N=1024) ---
        {
            let n: usize = 1024;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_b = vec![1.0f32; n * n];
            let h_c = vec![1.0f32; n * n];
            let mut h_tmp = vec![0.0f32; n * n];
            let mut h_d = vec![0.0f32; n * n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let d_c = ctx.new_tensor_view(h_c.as_slice()).unwrap();
            let mut d_tmp = ctx.new_tensor_view(h_tmp.as_mut_slice()).unwrap();
            let mut d_d = ctx.new_tensor_view(h_d.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let g = (n as u32 + bx - 1) / bx;
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm2_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_tmp, n as u32, n as u32, n as u32, 1.0f32).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm2_kernel2::launch(cfg, ctx, m, &d_tmp, &d_c, &mut d_d, n as u32, n as u32, n as u32, 1.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm2_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_tmp, n as u32, n as u32, n as u32, 1.0f32).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm2_kernel2::launch(cfg, ctx, m, &d_tmp, &d_c, &mut d_d, n as u32, n as u32, n as u32, 1.0f32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("twomm SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- threemm (N=512) ---
        {
            let n: usize = 512;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_b = vec![1.0f32; n * n];
            let h_c = vec![1.0f32; n * n];
            let h_d = vec![1.0f32; n * n];
            let mut h_e = vec![0.0f32; n * n];
            let mut h_f = vec![0.0f32; n * n];
            let mut h_g = vec![0.0f32; n * n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let d_c = ctx.new_tensor_view(h_c.as_slice()).unwrap();
            let d_d = ctx.new_tensor_view(h_d.as_slice()).unwrap();
            let mut d_e = ctx.new_tensor_view(h_e.as_mut_slice()).unwrap();
            let mut d_f = ctx.new_tensor_view(h_f.as_mut_slice()).unwrap();
            let mut d_g = ctx.new_tensor_view(h_g.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let g = (n as u32 + bx - 1) / bx;
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm3_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_e, n as u32, n as u32, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm3_kernel2::launch(cfg, ctx, m, &d_c, &d_d, &mut d_f, n as u32, n as u32, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm3_kernel3::launch(cfg, ctx, m, &d_e, &d_f, &mut d_g, n as u32, n as u32, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_e, n as u32, n as u32, n as u32).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel2::launch(cfg, ctx, m, &d_c, &d_d, &mut d_f, n as u32, n as u32, n as u32).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel3::launch(cfg, ctx, m, &d_e, &d_f, &mut d_g, n as u32, n as u32, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("threemm SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- atax (N=4096) ---
        {
            let n: usize = 4096;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_x = vec![1.0f32; n];
            let mut h_tmp = vec![0.0f32; n];
            let mut h_y = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
            let mut d_tmp = ctx.new_tensor_view(h_tmp.as_mut_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_atax_kernel1::launch(cfg, ctx, m, &d_a, &d_x, &mut d_tmp, n as u32, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_atax_kernel2::launch(cfg, ctx, m, &d_a, &d_tmp, &mut d_y, n as u32, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_atax_kernel1::launch(cfg, ctx, m, &d_a, &d_x, &mut d_tmp, n as u32, n as u32).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_atax_kernel2::launch(cfg, ctx, m, &d_a, &d_tmp, &mut d_y, n as u32, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("atax SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- bicg (N=4096) ---
        {
            let n: usize = 4096;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_r = vec![1.0f32; n];
            let h_p = vec![1.0f32; n];
            let mut h_s = vec![0.0f32; n];
            let mut h_q = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_r = ctx.new_tensor_view(h_r.as_slice()).unwrap();
            let d_p = ctx.new_tensor_view(h_p.as_slice()).unwrap();
            let mut d_s = ctx.new_tensor_view(h_s.as_mut_slice()).unwrap();
            let mut d_q = ctx.new_tensor_view(h_q.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_bicg_kernel1::launch(cfg, ctx, m, &d_a, &d_r, &mut d_s, n as u32, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_bicg_kernel2::launch(cfg, ctx, m, &d_a, &d_p, &mut d_q, n as u32, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_bicg_kernel1::launch(cfg, ctx, m, &d_a, &d_r, &mut d_s, n as u32, n as u32).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_bicg_kernel2::launch(cfg, ctx, m, &d_a, &d_p, &mut d_q, n as u32, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("bicg SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- mvt (N=4096) ---
        {
            let n: usize = 4096;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_y1 = vec![1.0f32; n];
            let h_y2 = vec![1.0f32; n];
            let mut h_x1 = vec![0.0f32; n];
            let mut h_x2 = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_y1 = ctx.new_tensor_view(h_y1.as_slice()).unwrap();
            let d_y2 = ctx.new_tensor_view(h_y2.as_slice()).unwrap();
            let mut d_x1 = ctx.new_tensor_view(h_x1.as_mut_slice()).unwrap();
            let mut d_x2 = ctx.new_tensor_view(h_x2.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_mvt_kernel1::launch(cfg, ctx, m, &d_a, &mut d_x1, &d_y1, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_mvt_kernel2::launch(cfg, ctx, m, &d_a, &mut d_x2, &d_y2, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_mvt_kernel1::launch(cfg, ctx, m, &d_a, &mut d_x1, &d_y1, n as u32).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_mvt_kernel2::launch(cfg, ctx, m, &d_a, &mut d_x2, &d_y2, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("mvt SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- gesummv (N=4096) ---
        {
            let n: usize = 4096;
            let iters = 100;
            let h_a = vec![1.0f32; n * n];
            let h_b = vec![1.0f32; n * n];
            let h_x = vec![1.0f32; n];
            let mut h_y = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_gesummv::launch(cfg, ctx, m, &d_a, &d_b, &d_x, &mut d_y, n as u32, 1.0f32, 1.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_gesummv::launch(cfg, ctx, m, &d_a, &d_b, &d_x, &mut d_y, n as u32, 1.0f32, 1.0f32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("gesummv SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }

        // --- syr2k (NI=NJ=1024) ---
        {
            let ni: usize = 1024;
            let nj: usize = 1024;
            let iters = 100;
            let h_a = vec![1.0f32; ni * nj];
            let h_b = vec![1.0f32; ni * nj];
            let mut h_c = vec![0.0f32; ni * ni];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let g = (ni as u32 + bx - 1) / bx;
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_syr2k::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_syr2k::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, 1.0f32, 0.0f32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("syr2k SeGuRu: {:.3} us/iter (NI={}, NJ={}, {} iters)", us, ni, nj, iters);
        }

        // --- syrk (NI=NJ=1024) ---
        {
            let ni: usize = 1024;
            let nj: usize = 1024;
            let iters = 100;
            let h_a = vec![1.0f32; ni * nj];
            let mut h_c = vec![0.0f32; ni * ni];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();
            let bx: u32 = 16;
            let by: u32 = 16;
            let g = (ni as u32 + bx - 1) / bx;
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_syrk::launch(cfg, ctx, m, &d_a, &mut d_c, ni as u32, nj as u32, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_syrk::launch(cfg, ctx, m, &d_a, &mut d_c, ni as u32, nj as u32, 1.0f32, 0.0f32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("syrk SeGuRu: {:.3} us/iter (NI={}, NJ={}, {} iters)", us, ni, nj, iters);
        }

        // --- corr (M=N=2048) ---
        {
            let mn: usize = 2048;
            let nn: usize = 2048;
            let iters = 100;
            let h_data: Vec<f32> = (0..nn * mn).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_data_gpu = h_data.clone();
            let mut h_mean = vec![0.0f32; mn];
            let mut h_stddev = vec![0.0f32; mn];
            let mut h_symmat = vec![0.0f32; mn * mn];
            let d_data_ro = ctx.new_tensor_view(h_data.as_slice()).unwrap();
            let mut d_data = ctx.new_tensor_view(h_data_gpu.as_mut_slice()).unwrap();
            let mut d_mean = ctx.new_tensor_view(h_mean.as_mut_slice()).unwrap();
            let mut d_stddev = ctx.new_tensor_view(h_stddev.as_mut_slice()).unwrap();
            let mut d_symmat = ctx.new_tensor_view(h_symmat.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let gm = (mn as u32 + bs - 1) / bs;
            let gn = (nn as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
            bench_corr_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn as u32, nn as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
            bench_corr_std::launch(cfg, ctx, m, &d_data_ro, &d_mean, &mut d_stddev, mn as u32, nn as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
            bench_corr_reduce::launch(cfg, ctx, m, &d_mean, &d_stddev, &mut d_data, mn as u32, nn as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
            bench_corr_corr::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn as u32, nn as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                d_data.copy_from_host(&h_data).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_corr_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn as u32, nn as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_corr_std::launch(cfg, ctx, m, &d_data_ro, &d_mean, &mut d_stddev, mn as u32, nn as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
                bench_corr_reduce::launch(cfg, ctx, m, &d_mean, &d_stddev, &mut d_data, mn as u32, nn as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
                bench_corr_corr::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn as u32, nn as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("corr SeGuRu: {:.3} us/iter (M={}, N={}, {} iters)", us, mn, nn, iters);
        }

        // --- covar (M=N=2048) ---
        {
            let mn: usize = 2048;
            let nn: usize = 2048;
            let iters = 100;
            let h_data: Vec<f32> = (0..nn * mn).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_data_gpu = h_data.clone();
            let mut h_mean = vec![0.0f32; mn];
            let mut h_symmat = vec![0.0f32; mn * mn];
            let d_data_ro = ctx.new_tensor_view(h_data.as_slice()).unwrap();
            let mut d_data = ctx.new_tensor_view(h_data_gpu.as_mut_slice()).unwrap();
            let mut d_mean = ctx.new_tensor_view(h_mean.as_mut_slice()).unwrap();
            let mut d_symmat = ctx.new_tensor_view(h_symmat.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let gm = (mn as u32 + bs - 1) / bs;
            let gn = (nn as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
            bench_covar_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn as u32, nn as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
            bench_covar_reduce::launch(cfg, ctx, m, &d_mean, &mut d_data, mn as u32, nn as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
            bench_covar_covar::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn as u32, nn as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                d_data.copy_from_host(&h_data).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_covar_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn as u32, nn as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
                bench_covar_reduce::launch(cfg, ctx, m, &d_mean, &mut d_data, mn as u32, nn as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
                bench_covar_covar::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn as u32, nn as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("covar SeGuRu: {:.3} us/iter (M={}, N={}, {} iters)", us, mn, nn, iters);
        }

        // --- doitgen (NR=NQ=NP=128) ---
        {
            let nr: usize = 128;
            let nq: usize = 128;
            let np: usize = 128;
            let iters = 100;
            let sz = nr * nq * np;
            let h_a: Vec<f32> = (0..sz).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let h_c4: Vec<f32> = (0..np * np).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_a_gpu = h_a.clone();
            let mut h_sum = vec![0.0f32; sz];
            let d_a_ro = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let mut d_a = ctx.new_tensor_view(h_a_gpu.as_mut_slice()).unwrap();
            let d_c4 = ctx.new_tensor_view(h_c4.as_slice()).unwrap();
            let mut d_sum = ctx.new_tensor_view(h_sum.as_mut_slice()).unwrap();
            let bx: u32 = 128.min(np as u32);
            let by: u32 = (1024 / bx).min((nr * nq) as u32);
            let gx = (np as u32 + bx - 1) / bx;
            let gy = ((nr * nq) as u32 + by - 1) / by;
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_doitgen_kernel1::launch(cfg, ctx, m, &d_a_ro, &d_c4, &mut d_sum, nr as u32, nq as u32, np as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_doitgen_kernel2::launch(cfg, ctx, m, &d_sum, &mut d_a, nr as u32, nq as u32, np as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_doitgen_kernel1::launch(cfg, ctx, m, &d_a_ro, &d_c4, &mut d_sum, nr as u32, nq as u32, np as u32).unwrap();
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_doitgen_kernel2::launch(cfg, ctx, m, &d_sum, &mut d_a, nr as u32, nq as u32, np as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("doitgen SeGuRu: {:.3} us/iter (NR={}, NQ={}, NP={}, {} iters)", us, nr, nq, np, iters);
        }

        // --- fdtd2d (NX=NY=2048, TMAX=500) ---
        {
            let nx: usize = 2048;
            let ny: usize = 2048;
            let tmax: usize = 500;
            let iters = 1;
            let sz = nx * ny;
            let fict: Vec<f32> = (0..tmax).map(|t| t as f32).collect();
            let mut h_ex = vec![0.0f32; sz];
            let mut h_ey = vec![0.0f32; sz];
            let mut h_hz = vec![0.0f32; sz];
            for i in 0..nx {
                for j in 0..ny {
                    h_ex[i * ny + j] = (i * (j + 1)) as f32 / nx as f32;
                    h_ey[i * ny + j] = (i * (j + 2)) as f32 / ny as f32;
                    h_hz[i * ny + j] = (i * (j + 3)) as f32 / nx as f32;
                }
            }
            let d_fict = ctx.new_tensor_view(fict.as_slice()).unwrap();
            let mut d_ex = ctx.new_tensor_view(h_ex.as_mut_slice()).unwrap();
            let mut d_ey = ctx.new_tensor_view(h_ey.as_mut_slice()).unwrap();
            let mut d_hz = ctx.new_tensor_view(h_hz.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let gx = (ny as u32 + bs - 1) / bs;
            let gy = (nx as u32 + bs - 1) / bs;
            // Warmup
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
            bench_fdtd_step1::launch(cfg, ctx, m, &d_fict, &mut d_ey, &d_hz, nx as u32, ny as u32, 0u32).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
            bench_fdtd_step2::launch(cfg, ctx, m, &mut d_ex, &d_hz, nx as u32, ny as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
            bench_fdtd_step3::launch(cfg, ctx, m, &d_ex, &d_ey, &mut d_hz, nx as u32, ny as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for t in 0..tmax {
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step1::launch(cfg, ctx, m, &d_fict, &mut d_ey, &d_hz, nx as u32, ny as u32, t as u32).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step2::launch(cfg, ctx, m, &mut d_ex, &d_hz, nx as u32, ny as u32).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step3::launch(cfg, ctx, m, &d_ex, &d_ey, &mut d_hz, nx as u32, ny as u32).unwrap();
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("fdtd2d SeGuRu: {:.3} us/iter (NX={}, NY={}, TMAX={}, {} iters)", us, nx, ny, tmax, iters);
        }

        // --- gramschm (NI=NJ=2048) ---
        {
            let ni: usize = 2048;
            let nj: usize = 2048;
            let iters = 1;
            let mut h_a: Vec<f32> = vec![0.0; ni * nj];
            for i in 0..ni {
                for j in 0..nj {
                    h_a[i * nj + j] = ((i * j + 1) as f32) / (ni * nj) as f32;
                }
            }
            let mut h_r = vec![0.0f32; nj * nj];
            let mut h_q = vec![0.0f32; ni * nj];
            let mut d_a = ctx.new_tensor_view(h_a.as_mut_slice()).unwrap();
            let mut d_r = ctx.new_tensor_view(h_r.as_mut_slice()).unwrap();
            let mut d_q = ctx.new_tensor_view(h_q.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            // Warmup (1 iteration)
            {
                let mut h_a_tmp = vec![0.0f32; ni * nj];
                d_a.copy_to_host(&mut h_a_tmp).unwrap();
                let mut nrm = 0.0f32;
                for i in 0..ni {
                    nrm += h_a_tmp[i * nj] * h_a_tmp[i * nj];
                }
                d_r.copy_to_host(&mut h_r).unwrap();
                h_r[0] = nrm.sqrt();
                d_r.copy_from_host(&h_r).unwrap();
                let r_kk = h_r[0];
                let gx = (nj as u32 + bs - 1) / bs;
                let gy = (ni as u32 + bs - 1) / bs;
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                bench_gramschm_kernel2::launch(cfg, ctx, m, &d_a, r_kk, &mut d_q, nj as u32, ni as u32, 0u32).unwrap();
                let gy2 = (nj as u32 + bs - 1) / bs;
                let cfg = gpu_host::gpu_config!(gx, gy2, 1, bs, bs, 1, 0);
                bench_gramschm_kernel3a::launch(cfg, ctx, m, &d_q, &d_a, &mut d_r, ni as u32, nj as u32, 0u32).unwrap();
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                bench_gramschm_kernel3b::launch(cfg, ctx, m, &d_q, &d_r, &mut d_a, ni as u32, nj as u32, 0u32).unwrap();
                ctx.sync().unwrap();
            }
            // Reset
            for i in 0..ni {
                for j in 0..nj {
                    h_a[i * nj + j] = ((i * j + 1) as f32) / (ni * nj) as f32;
                }
            }
            h_r = vec![0.0f32; nj * nj];
            h_q = vec![0.0f32; ni * nj];
            d_a.copy_from_host(&h_a).unwrap();
            d_r.copy_from_host(&h_r).unwrap();
            d_q.copy_from_host(&h_q).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let mut h_a_tmp = vec![0.0f32; ni * nj];
                for k in 0..nj {
                    d_a.copy_to_host(&mut h_a_tmp).unwrap();
                    let mut nrm = 0.0f32;
                    for ii in 0..ni {
                        nrm += h_a_tmp[ii * nj + k] * h_a_tmp[ii * nj + k];
                    }
                    d_r.copy_to_host(&mut h_r).unwrap();
                    h_r[k * nj + k] = nrm.sqrt();
                    d_r.copy_from_host(&h_r).unwrap();
                    let r_kk = h_r[k * nj + k];

                    let gx = (nj as u32 + bs - 1) / bs;
                    let gy = (ni as u32 + bs - 1) / bs;
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_gramschm_kernel2::launch(cfg, ctx, m, &d_a, r_kk, &mut d_q, nj as u32, ni as u32, k as u32).unwrap();
                    let gy2 = (nj as u32 + bs - 1) / bs;
                    let cfg = gpu_host::gpu_config!(gx, gy2, 1, bs, bs, 1, 0);
                    bench_gramschm_kernel3a::launch(cfg, ctx, m, &d_q, &d_a, &mut d_r, ni as u32, nj as u32, k as u32).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_gramschm_kernel3b::launch(cfg, ctx, m, &d_q, &d_r, &mut d_a, ni as u32, nj as u32, k as u32).unwrap();
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("gramschm SeGuRu: {:.3} us/iter (NI={}, NJ={}, {} iters)", us, ni, nj, iters);
        }

        // --- jacobi1d (N=4096, TSTEPS=10000) ---
        {
            let n: usize = 4096;
            let tsteps: usize = 10000;
            let iters = 1;
            let mut h_a: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
            let mut h_b = vec![0.0f32; n];
            let mut d_a = ctx.new_tensor_view(h_a.as_mut_slice()).unwrap();
            let mut d_b = ctx.new_tensor_view(h_b.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_jacobi1d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_jacobi1d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for _ in 0..tsteps {
                    let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                    bench_jacobi1d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n as u32).unwrap();
                    let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                    bench_jacobi1d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n as u32).unwrap();
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("jacobi1d SeGuRu: {:.3} us/iter (N={}, TSTEPS={}, {} iters)", us, n, tsteps, iters);
        }

        // --- jacobi2d (N=1000, TSTEPS=20) ---
        {
            let n: usize = 1000;
            let tsteps: usize = 20;
            let iters = 1;
            let mut h_a: Vec<f32> = (0..n * n).map(|i| (i % 1024) as f32 / 1024.0).collect();
            let mut h_b = vec![0.0f32; n * n];
            let mut d_a = ctx.new_tensor_view(h_a.as_mut_slice()).unwrap();
            let mut d_b = ctx.new_tensor_view(h_b.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let g = (n as u32 + bs - 1) / bs;
            let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
            bench_jacobi2d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
            bench_jacobi2d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n as u32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for _ in 0..tsteps {
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_jacobi2d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n as u32).unwrap();
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_jacobi2d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n as u32).unwrap();
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("jacobi2d SeGuRu: {:.3} us/iter (N={}, TSTEPS={}, {} iters)", us, n, tsteps, iters);
        }

        // --- lu (N=2048) ---
        {
            let n: usize = 2048;
            let iters = 1;
            let mut h_a: Vec<f32> = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    h_a[i * n + j] = if i == j {
                        (n as f32) + 1.0
                    } else {
                        (i as f32 + j as f32) / n as f32
                    };
                }
            }
            let mut h_a_gpu = h_a.clone();
            let mut h_col = vec![0.0f32; n];
            let mut d_a = ctx.new_tensor_view(h_a_gpu.as_mut_slice()).unwrap();
            let mut d_col = ctx.new_tensor_view(h_col.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let row_bs: u32 = 256;
            macro_rules! launch_lu_step {
                ($k:expr) => {{
                    let k = $k;
                    let rem = n - k - 1;
                    if rem > 0 {
                        {
                            let row_tail_start = k * n + k + 1;
                            let row_tail_end = (k + 1) * n;
                            let (prefix, mut tail_and_after) = d_a.split_at_mut(row_tail_start);
                            let pivot = prefix.index(row_tail_start - 1..row_tail_start);
                            let (mut row_tail, _) =
                                tail_and_after.split_at_mut(row_tail_end - row_tail_start);
                            let grid = (rem as u32 + row_bs - 1) / row_bs;
                            let cfg = gpu_host::gpu_config!(grid, 1, 1, row_bs, 1, 1, 0);
                            bench_lu_kernel1::launch(
                                cfg,
                                ctx,
                                m,
                                &pivot,
                                &mut row_tail,
                                rem as u32,
                            )
                            .unwrap();
                        }
                        ctx.sync().unwrap();
                        {
                            let grid = (rem as u32 + row_bs - 1) / row_bs;
                            let cfg = gpu_host::gpu_config!(grid, 1, 1, row_bs, 1, 1, 0);
                            bench_lu_copy_col::launch(
                                cfg,
                                ctx,
                                m,
                                &d_a,
                                &mut d_col,
                                n as u32,
                                k as u32,
                            )
                            .unwrap();
                        }
                        {
                            let row_tail_start = k * n + k + 1;
                            let row_tail_end = (k + 1) * n;
                            let split_at = (k + 1) * n;
                            let (prefix, mut rows_below) = d_a.split_at_mut(split_at);
                            let row_tail = prefix.index(row_tail_start..row_tail_end);
                            let col = d_col.index(..rem);
                            let grid = (rem as u32 + bs - 1) / bs;
                            let cfg = gpu_host::gpu_config!(grid, grid, 1, bs, bs, 1, 0);
                            bench_lu_kernel2::launch(
                                cfg,
                                ctx,
                                m,
                                &row_tail,
                                &col,
                                &mut rows_below,
                                n as u32,
                                k as u32,
                            )
                            .unwrap();
                        }
                        ctx.sync().unwrap();
                    }
                }};
            }
            // Warmup
            {
                launch_lu_step!(0usize);
            }
            // Reset
            d_a.copy_from_host(&h_a_gpu).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                for k in 0..n {
                    launch_lu_step!(k);
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("lu SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }
    });
}
