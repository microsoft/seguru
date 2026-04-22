use gpu::prelude::*;
use std::time::Instant;

// =====================================================================
// Kernel definitions — all 19 PolybenchGPU benchmarks
// =====================================================================

// --- conv2d ---
#[gpu::cuda_kernel]
pub fn bench_conv2d(a: &[f32], b: &mut [f32], ni: usize, nj: usize) {
    let mut b = chunk_mut(b, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
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
        b[0] = c11 * a[(i - 1) * nj + (j - 1)]
            + c21 * a[(i - 1) * nj + j]
            + c31 * a[(i - 1) * nj + (j + 1)]
            + c12 * a[i * nj + (j - 1)]
            + c22 * a[i * nj + j]
            + c32 * a[i * nj + (j + 1)]
            + c13 * a[(i + 1) * nj + (j - 1)]
            + c23 * a[(i + 1) * nj + j]
            + c33 * a[(i + 1) * nj + (j + 1)];
    }
}

// --- conv3d ---
#[gpu::cuda_kernel]
pub fn bench_conv3d(a: &[f32], b: &mut [f32], ni: usize, nj: usize, nk: usize) {
    let mut b = chunk_mut(b, MapLinear::new(1));
    let k = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let ij = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
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
        b[0] = c11 * a[(i - 1) * s + (j - 1) * nk + (k - 1)]
            + c12 * a[(i - 1) * s + (j - 1) * nk + k]
            + c13 * a[(i - 1) * s + (j - 1) * nk + (k + 1)]
            + c21 * a[(i - 1) * s + j * nk + (k - 1)]
            + c22 * a[(i - 1) * s + j * nk + k]
            + c23 * a[(i - 1) * s + j * nk + (k + 1)]
            + c31 * a[(i - 1) * s + (j + 1) * nk + (k - 1)]
            + c32 * a[(i - 1) * s + (j + 1) * nk + k]
            + c33 * a[(i - 1) * s + (j + 1) * nk + (k + 1)]
            + c11 * a[i * s + (j - 1) * nk + (k - 1)]
            + c12 * a[i * s + (j - 1) * nk + k]
            + c13 * a[i * s + (j - 1) * nk + (k + 1)]
            + c21 * a[i * s + j * nk + (k - 1)]
            + c22 * a[i * s + j * nk + k]
            + c23 * a[i * s + j * nk + (k + 1)]
            + c31 * a[i * s + (j + 1) * nk + (k - 1)]
            + c32 * a[i * s + (j + 1) * nk + k]
            + c33 * a[i * s + (j + 1) * nk + (k + 1)]
            + c11 * a[(i + 1) * s + (j - 1) * nk + (k - 1)]
            + c12 * a[(i + 1) * s + (j - 1) * nk + k]
            + c13 * a[(i + 1) * s + (j - 1) * nk + (k + 1)]
            + c21 * a[(i + 1) * s + j * nk + (k - 1)]
            + c22 * a[(i + 1) * s + j * nk + k]
            + c23 * a[(i + 1) * s + j * nk + (k + 1)]
            + c31 * a[(i + 1) * s + (j + 1) * nk + (k - 1)]
            + c32 * a[(i + 1) * s + (j + 1) * nk + k]
            + c33 * a[(i + 1) * s + (j + 1) * nk + (k + 1)];
    }
}

// --- gemm ---
#[gpu::cuda_kernel]
pub fn bench_gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j < nj {
        let mut val = c[(0, 0)] * beta;
        let mut k: usize = 0;
        while k < nk {
            val += alpha * a[i * nk + k] * b[k * nj + j];
            k += 1;
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
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
) {
    let mut tmp = chunk_mut(tmp, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j < nj {
        let mut val = 0.0f32;
        let mut k: usize = 0;
        while k < nk {
            val += alpha * a[i * nk + k] * b[k * nj + j];
            k += 1;
        }
        tmp[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm2_kernel2(
    tmp: &[f32],
    c: &[f32],
    d: &mut [f32],
    ni: usize,
    nj: usize,
    nl: usize,
    beta: f32,
) {
    let mut d = chunk_mut(d, Map2D::new(nl));
    let l = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && l < nl {
        let mut val = d[(0, 0)] * beta;
        let mut j: usize = 0;
        while j < nj {
            val += tmp[i * nj + j] * c[j * nl + l];
            j += 1;
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
    ni: usize,
    nj: usize,
    nk: usize,
) {
    let mut e = chunk_mut(e, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j < nj {
        let mut val = 0.0f32;
        let mut k: usize = 0;
        while k < nk {
            val += a[i * nk + k] * b[k * nj + j];
            k += 1;
        }
        e[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm3_kernel2(
    c: &[f32],
    d: &[f32],
    f: &mut [f32],
    nj: usize,
    nl: usize,
    nm: usize,
) {
    let mut f = chunk_mut(f, Map2D::new(nl));
    let l = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let j = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j < nj && l < nl {
        let mut val = 0.0f32;
        let mut m: usize = 0;
        while m < nm {
            val += c[j * nm + m] * d[m * nl + l];
            m += 1;
        }
        f[(0, 0)] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mm3_kernel3(
    e: &[f32],
    f: &[f32],
    g: &mut [f32],
    ni: usize,
    nj: usize,
    nl: usize,
) {
    let mut g = chunk_mut(g, Map2D::new(nl));
    let l = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && l < nl {
        let mut val = 0.0f32;
        let mut j: usize = 0;
        while j < nj {
            val += e[i * nj + j] * f[j * nl + l];
            j += 1;
        }
        g[(0, 0)] = val;
    }
}

// --- atax ---
#[gpu::cuda_kernel]
pub fn bench_atax_kernel1(a: &[f32], x: &[f32], tmp: &mut [f32], nx: usize, ny: usize) {
    let mut tmp = chunk_mut(tmp, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < nx {
        let mut sum = 0.0f32;
        let mut j: usize = 0;
        while j < ny {
            sum += a[i * ny + j] * x[j];
            j += 1;
        }
        tmp[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_atax_kernel2(a: &[f32], tmp: &[f32], y: &mut [f32], nx: usize, ny: usize) {
    let mut y = chunk_mut(y, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < nx {
            sum += a[i * ny + j] * tmp[i];
            i += 1;
        }
        y[0] = sum;
    }
}

// --- bicg ---
#[gpu::cuda_kernel]
pub fn bench_bicg_kernel1(a: &[f32], r: &[f32], s: &mut [f32], nx: usize, ny: usize) {
    let mut s = chunk_mut(s, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < nx {
            sum += r[i] * a[i * ny + j];
            i += 1;
        }
        s[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_bicg_kernel2(a: &[f32], p: &[f32], q: &mut [f32], nx: usize, ny: usize) {
    let mut q = chunk_mut(q, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < nx {
        let mut sum = 0.0f32;
        let mut j: usize = 0;
        while j < ny {
            sum += a[i * ny + j] * p[j];
            j += 1;
        }
        q[0] = sum;
    }
}

// --- mvt ---
#[gpu::cuda_kernel]
pub fn bench_mvt_kernel1(a: &[f32], x1: &mut [f32], y1: &[f32], n: usize) {
    let mut x1 = chunk_mut(x1, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < n {
        let mut sum = x1[0];
        let mut j: usize = 0;
        while j < n {
            sum += a[i * n + j] * y1[j];
            j += 1;
        }
        x1[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_mvt_kernel2(a: &[f32], x2: &mut [f32], y2: &[f32], n: usize) {
    let mut x2 = chunk_mut(x2, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < n {
        let mut sum = x2[0];
        let mut j: usize = 0;
        while j < n {
            sum += a[j * n + i] * y2[j];
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
    n: usize,
    alpha: f32,
    beta: f32,
) {
    let mut y = chunk_mut(y, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < n {
        let mut sum_a = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut j: usize = 0;
        while j < n {
            sum_a += a[i * n + j] * x[j];
            sum_b += b[i * n + j] * x[j];
            j += 1;
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
    ni: usize,
    nj: usize,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(ni));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j <= i {
        let mut val = c[(0, 0)] * beta;
        let mut k: usize = 0;
        while k < nj {
            val += alpha * a[i * nj + k] * b[j * nj + k]
                + alpha * b[i * nj + k] * a[j * nj + k];
            k += 1;
        }
        c[(0, 0)] = val;
    }
}

// --- syrk ---
#[gpu::cuda_kernel]
pub fn bench_syrk(
    a: &[f32],
    c: &mut [f32],
    ni: usize,
    nj: usize,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(ni));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j <= i {
        let mut val = c[(0, 0)] * beta;
        let mut k: usize = 0;
        while k < nj {
            val += alpha * a[i * nj + k] * a[j * nj + k];
            k += 1;
        }
        c[(0, 0)] = val;
    }
}

// --- corr ---
const CORR_FLOAT_N: f32 = 3214212.01;
const CORR_EPS: f32 = 0.005;

#[gpu::cuda_kernel]
pub fn bench_corr_mean(data: &[f32], mean: &mut [f32], m: usize, n: usize) {
    let mut mean = chunk_mut(mean, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j];
            i += 1;
        }
        mean[0] = sum / CORR_FLOAT_N;
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_std(data: &[f32], mean: &[f32], stddev: &mut [f32], m: usize, n: usize) {
    let mut stddev = chunk_mut(stddev, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < m {
        let mut sum = 0.0f32;
        let mean_j = mean[j];
        let mut i: usize = 0;
        while i < n {
            let diff = data[i * m + j] - mean_j;
            sum += diff * diff;
            i += 1;
        }
        sum /= CORR_FLOAT_N;
        let s = sum.sqrt();
        stddev[0] = if s <= CORR_EPS { 1.0 } else { s };
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_reduce(mean: &[f32], stddev: &[f32], data: &mut [f32], m: usize, n: usize) {
    let mut data = chunk_mut(data, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < n && j < m {
        let val = data[0] - mean[j];
        data[0] = val / (CORR_FLOAT_N.sqrt() * stddev[j]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_corr_corr(data: &[f32], symmat: &mut [f32], m: usize, n: usize) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m));
    let j2 = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let j1 = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j1] * data[i * m + j2];
            i += 1;
        }
        symmat[(0, 0)] = sum;
    }
}

// --- covar ---
const COVAR_FLOAT_N: f32 = 3214212.01;

#[gpu::cuda_kernel]
pub fn bench_covar_mean(data: &[f32], mean: &mut [f32], m: usize, n: usize) {
    let mut mean = chunk_mut(mean, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j];
            i += 1;
        }
        mean[0] = sum / COVAR_FLOAT_N;
    }
}

#[gpu::cuda_kernel]
pub fn bench_covar_reduce(mean: &[f32], data: &mut [f32], m: usize, n: usize) {
    let mut data = chunk_mut(data, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < n && j < m {
        data[0] = data[0] - mean[j];
    }
}

#[gpu::cuda_kernel]
pub fn bench_covar_covar(data: &[f32], symmat: &mut [f32], m: usize, n: usize) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m));
    let j2 = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let j1 = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j1] * data[i * m + j2];
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
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut sum_arr = chunk_mut(sum_arr, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        let mut val = 0.0f32;
        let mut s: usize = 0;
        while s < np {
            val += a[r * (nq * np) + q * np + s] * c4[s * np + p];
            s += 1;
        }
        sum_arr[0] = val;
    }
}

#[gpu::cuda_kernel]
pub fn bench_doitgen_kernel2(
    sum_arr: &[f32],
    a: &mut [f32],
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        a[0] = sum_arr[r * (nq * np) + q * np + p];
    }
}

// --- fdtd2d ---
#[gpu::cuda_kernel]
pub fn bench_fdtd_step1(
    fict: &[f32],
    ey: &mut [f32],
    hz: &[f32],
    nx: usize,
    ny: usize,
    t: usize,
) {
    let mut ey = chunk_mut(ey, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < nx && j < ny {
        if i == 0 {
            ey[0] = fict[t];
        } else {
            ey[0] = ey[0] - 0.5 * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
        }
    }
}

#[gpu::cuda_kernel]
pub fn bench_fdtd_step2(ex: &mut [f32], hz: &[f32], nx: usize, ny: usize) {
    let mut ex = chunk_mut(ex, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < nx && j > 0 && j < ny {
        ex[0] = ex[0] - 0.5 * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_fdtd_step3(ex: &[f32], ey: &[f32], hz: &mut [f32], nx: usize, ny: usize) {
    let mut hz = chunk_mut(hz, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < nx - 1 && j < ny - 1 {
        hz[0] = hz[0]
            - 0.7
                * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j]
                    - ey[i * ny + j]);
    }
}

// --- gramschm ---
#[gpu::cuda_kernel]
pub fn bench_gramschm_kernel2(
    a: &[f32],
    r_kk: f32,
    q: &mut [f32],
    nj: usize,
    ni: usize,
    k: usize,
) {
    let mut q = chunk_mut(q, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j == k {
        q[(0, 0)] = a[i * nj + k] / r_kk;
    }
}

#[gpu::cuda_kernel]
pub fn bench_gramschm_kernel3a(
    q: &[f32],
    a: &[f32],
    r: &mut [f32],
    ni: usize,
    nj: usize,
    k: usize,
) {
    let mut r = chunk_mut(r, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if row == k && j > k && j < nj {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < ni {
            sum += q[i * nj + k] * a[i * nj + j];
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
    ni: usize,
    nj: usize,
    k: usize,
) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j > k && j < nj && i < ni {
        a[0] = a[0] - q[i * nj + k] * r[k * nj + j];
    }
}

// --- jacobi1d ---
#[gpu::cuda_kernel]
pub fn bench_jacobi1d_kernel1(a: &[f32], b: &mut [f32], n: usize) {
    let mut b = chunk_mut(b, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i > 0 && i < n - 1 {
        b[0] = 0.33333 * (a[i - 1] + a[i] + a[i + 1]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_jacobi1d_kernel2(a: &mut [f32], b: &[f32], n: usize) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i > 0 && i < n - 1 {
        a[0] = b[i];
    }
}

// --- jacobi2d ---
#[gpu::cuda_kernel]
pub fn bench_jacobi2d_kernel1(a: &[f32], b: &mut [f32], n: usize) {
    let mut b = chunk_mut(b, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i > 0 && i < n - 1 && j > 0 && j < n - 1 {
        b[(0, 0)] = 0.2
            * (a[i * n + j] + a[i * n + j - 1] + a[i * n + j + 1] + a[(i + 1) * n + j]
                + a[(i - 1) * n + j]);
    }
}

#[gpu::cuda_kernel]
pub fn bench_jacobi2d_kernel2(a: &mut [f32], b: &[f32], n: usize) {
    let mut a = chunk_mut(a, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i > 0 && i < n - 1 && j > 0 && j < n - 1 {
        a[(0, 0)] = b[i * n + j];
    }
}

// --- lu ---
#[gpu::cuda_kernel]
pub fn bench_lu_kernel1(a_read: &[f32], a_write: &mut [f32], n: usize, k: usize) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i == k && j > k && j < n {
        a_write[(0, 0)] = a_read[k * n + j] / a_read[k * n + k];
    }
}

#[gpu::cuda_kernel]
pub fn bench_lu_kernel2(a_read: &[f32], a_write: &mut [f32], n: usize, k: usize) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i > k && j > k && i < n && j < n {
        a_write[(0, 0)] = a_read[i * n + j] - a_read[i * n + k] * a_read[k * n + j];
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
            bench_conv2d::launch(cfg, ctx, m, &d_a, &mut d_b, ni, nj).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_conv2d::launch(cfg, ctx, m, &d_a, &mut d_b, ni, nj).unwrap();
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
            bench_conv3d::launch(cfg, ctx, m, &d_a, &mut d_b, ni, nj, nk).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_conv3d::launch(cfg, ctx, m, &d_a, &mut d_b, ni, nj, nk).unwrap();
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
            bench_gemm::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni, nj, nk, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_gemm::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni, nj, nk, 1.0f32, 0.0f32).unwrap();
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
            bench_mm2_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_tmp, n, n, n, 1.0f32).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm2_kernel2::launch(cfg, ctx, m, &d_tmp, &d_c, &mut d_d, n, n, n, 1.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm2_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_tmp, n, n, n, 1.0f32).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm2_kernel2::launch(cfg, ctx, m, &d_tmp, &d_c, &mut d_d, n, n, n, 1.0f32).unwrap();
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
            bench_mm3_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_e, n, n, n).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm3_kernel2::launch(cfg, ctx, m, &d_c, &d_d, &mut d_f, n, n, n).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
            bench_mm3_kernel3::launch(cfg, ctx, m, &d_e, &d_f, &mut d_g, n, n, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel1::launch(cfg, ctx, m, &d_a, &d_b, &mut d_e, n, n, n).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel2::launch(cfg, ctx, m, &d_c, &d_d, &mut d_f, n, n, n).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_mm3_kernel3::launch(cfg, ctx, m, &d_e, &d_f, &mut d_g, n, n, n).unwrap();
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
            bench_atax_kernel1::launch(cfg, ctx, m, &d_a, &d_x, &mut d_tmp, n, n).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_atax_kernel2::launch(cfg, ctx, m, &d_a, &d_tmp, &mut d_y, n, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_atax_kernel1::launch(cfg, ctx, m, &d_a, &d_x, &mut d_tmp, n, n).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_atax_kernel2::launch(cfg, ctx, m, &d_a, &d_tmp, &mut d_y, n, n).unwrap();
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
            bench_bicg_kernel1::launch(cfg, ctx, m, &d_a, &d_r, &mut d_s, n, n).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_bicg_kernel2::launch(cfg, ctx, m, &d_a, &d_p, &mut d_q, n, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_bicg_kernel1::launch(cfg, ctx, m, &d_a, &d_r, &mut d_s, n, n).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_bicg_kernel2::launch(cfg, ctx, m, &d_a, &d_p, &mut d_q, n, n).unwrap();
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
            bench_mvt_kernel1::launch(cfg, ctx, m, &d_a, &mut d_x1, &d_y1, n).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_mvt_kernel2::launch(cfg, ctx, m, &d_a, &mut d_x2, &d_y2, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_mvt_kernel1::launch(cfg, ctx, m, &d_a, &mut d_x1, &d_y1, n).unwrap();
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_mvt_kernel2::launch(cfg, ctx, m, &d_a, &mut d_x2, &d_y2, n).unwrap();
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
            bench_gesummv::launch(cfg, ctx, m, &d_a, &d_b, &d_x, &mut d_y, n, 1.0f32, 1.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_gesummv::launch(cfg, ctx, m, &d_a, &d_b, &d_x, &mut d_y, n, 1.0f32, 1.0f32).unwrap();
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
            bench_syr2k::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni, nj, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_syr2k::launch(cfg, ctx, m, &d_a, &d_b, &mut d_c, ni, nj, 1.0f32, 0.0f32).unwrap();
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
            bench_syrk::launch(cfg, ctx, m, &d_a, &mut d_c, ni, nj, 1.0f32, 0.0f32).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(g, g, 1, bx, by, 1, 0);
                bench_syrk::launch(cfg, ctx, m, &d_a, &mut d_c, ni, nj, 1.0f32, 0.0f32).unwrap();
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
            bench_corr_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn, nn).unwrap();
            let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
            bench_corr_std::launch(cfg, ctx, m, &d_data_ro, &d_mean, &mut d_stddev, mn, nn).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
            bench_corr_reduce::launch(cfg, ctx, m, &d_mean, &d_stddev, &mut d_data, mn, nn).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
            bench_corr_corr::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn, nn).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                d_data.copy_from_host(&h_data).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_corr_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn, nn).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_corr_std::launch(cfg, ctx, m, &d_data_ro, &d_mean, &mut d_stddev, mn, nn).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
                bench_corr_reduce::launch(cfg, ctx, m, &d_mean, &d_stddev, &mut d_data, mn, nn).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
                bench_corr_corr::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn, nn).unwrap();
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
            bench_covar_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn, nn).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
            bench_covar_reduce::launch(cfg, ctx, m, &d_mean, &mut d_data, mn, nn).unwrap();
            let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
            bench_covar_covar::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn, nn).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                d_data.copy_from_host(&h_data).unwrap();
                let cfg = gpu_host::gpu_config!(gm, 1, 1, bs, 1, 1, 0);
                bench_covar_mean::launch(cfg, ctx, m, &d_data_ro, &mut d_mean, mn, nn).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gn, 1, bs, bs, 1, 0);
                bench_covar_reduce::launch(cfg, ctx, m, &d_mean, &mut d_data, mn, nn).unwrap();
                let cfg = gpu_host::gpu_config!(gm, gm, 1, bs, bs, 1, 0);
                bench_covar_covar::launch(cfg, ctx, m, &d_data, &mut d_symmat, mn, nn).unwrap();
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
            bench_doitgen_kernel1::launch(cfg, ctx, m, &d_a_ro, &d_c4, &mut d_sum, nr, nq, np).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_doitgen_kernel2::launch(cfg, ctx, m, &d_sum, &mut d_a, nr, nq, np).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_doitgen_kernel1::launch(cfg, ctx, m, &d_a_ro, &d_c4, &mut d_sum, nr, nq, np).unwrap();
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_doitgen_kernel2::launch(cfg, ctx, m, &d_sum, &mut d_a, nr, nq, np).unwrap();
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
            bench_fdtd_step1::launch(cfg, ctx, m, &d_fict, &mut d_ey, &d_hz, nx, ny, 0).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
            bench_fdtd_step2::launch(cfg, ctx, m, &mut d_ex, &d_hz, nx, ny).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
            bench_fdtd_step3::launch(cfg, ctx, m, &d_ex, &d_ey, &mut d_hz, nx, ny).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for t in 0..tmax {
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step1::launch(cfg, ctx, m, &d_fict, &mut d_ey, &d_hz, nx, ny, t).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step2::launch(cfg, ctx, m, &mut d_ex, &d_hz, nx, ny).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_fdtd_step3::launch(cfg, ctx, m, &d_ex, &d_ey, &mut d_hz, nx, ny).unwrap();
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
                bench_gramschm_kernel2::launch(cfg, ctx, m, &d_a, r_kk, &mut d_q, nj, ni, 0).unwrap();
                let gy2 = (nj as u32 + bs - 1) / bs;
                let cfg = gpu_host::gpu_config!(gx, gy2, 1, bs, bs, 1, 0);
                bench_gramschm_kernel3a::launch(cfg, ctx, m, &d_q, &d_a, &mut d_r, ni, nj, 0).unwrap();
                let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                bench_gramschm_kernel3b::launch(cfg, ctx, m, &d_q, &d_r, &mut d_a, ni, nj, 0).unwrap();
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
                    bench_gramschm_kernel2::launch(cfg, ctx, m, &d_a, r_kk, &mut d_q, nj, ni, k).unwrap();
                    let gy2 = (nj as u32 + bs - 1) / bs;
                    let cfg = gpu_host::gpu_config!(gx, gy2, 1, bs, bs, 1, 0);
                    bench_gramschm_kernel3a::launch(cfg, ctx, m, &d_q, &d_a, &mut d_r, ni, nj, k).unwrap();
                    let cfg = gpu_host::gpu_config!(gx, gy, 1, bs, bs, 1, 0);
                    bench_gramschm_kernel3b::launch(cfg, ctx, m, &d_q, &d_r, &mut d_a, ni, nj, k).unwrap();
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
            bench_jacobi1d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n).unwrap();
            let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_jacobi1d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for _ in 0..tsteps {
                    let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                    bench_jacobi1d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n).unwrap();
                    let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                    bench_jacobi1d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n).unwrap();
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
            bench_jacobi2d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n).unwrap();
            let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
            bench_jacobi2d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n).unwrap();
            ctx.sync().unwrap();
            let start = Instant::now();
            for _ in 0..iters {
                for _ in 0..tsteps {
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_jacobi2d_kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n).unwrap();
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_jacobi2d_kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n).unwrap();
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
            let mut h_a_read = h_a.clone();
            let mut d_a_write = ctx.new_tensor_view(h_a_gpu.as_mut_slice()).unwrap();
            let mut d_a_read = ctx.new_tensor_view(h_a_read.as_mut_slice()).unwrap();
            let bs: u32 = 16;
            let g = (n as u32 + bs - 1) / bs;
            // Warmup
            {
                let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                bench_lu_kernel1::launch(cfg, ctx, m, &d_a_read, &mut d_a_write, n, 0).unwrap();
                d_a_write.copy_to_host(&mut h_a_gpu).unwrap();
                d_a_read.copy_from_host(&h_a_gpu).unwrap();
                let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                bench_lu_kernel2::launch(cfg, ctx, m, &d_a_read, &mut d_a_write, n, 0).unwrap();
                d_a_write.copy_to_host(&mut h_a_gpu).unwrap();
                d_a_read.copy_from_host(&h_a_gpu).unwrap();
                ctx.sync().unwrap();
            }
            // Reset
            for i in 0..n {
                for j in 0..n {
                    h_a_gpu[i * n + j] = if i == j {
                        (n as f32) + 1.0
                    } else {
                        (i as f32 + j as f32) / n as f32
                    };
                }
            }
            h_a_read = h_a_gpu.clone();
            d_a_write.copy_from_host(&h_a_gpu).unwrap();
            d_a_read.copy_from_host(&h_a_read).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                for k in 0..n {
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_lu_kernel1::launch(cfg, ctx, m, &d_a_read, &mut d_a_write, n, k).unwrap();
                    d_a_write.copy_to_host(&mut h_a_gpu).unwrap();
                    d_a_read.copy_from_host(&h_a_gpu).unwrap();
                    let cfg = gpu_host::gpu_config!(g, g, 1, bs, bs, 1, 0);
                    bench_lu_kernel2::launch(cfg, ctx, m, &d_a_read, &mut d_a_write, n, k).unwrap();
                    d_a_write.copy_to_host(&mut h_a_gpu).unwrap();
                    d_a_read.copy_from_host(&h_a_gpu).unwrap();
                }
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("lu SeGuRu: {:.3} us/iter (N={}, {} iters)", us, n, iters);
        }
    });
}
