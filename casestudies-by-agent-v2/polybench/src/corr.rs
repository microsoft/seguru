//! CORR: correlation matrix of an `N x M` data set.
//!
//! `mean` and `stddev` are computed one thread per variable — the data is
//! row-major with the variable index innermost, so a warp's loads are
//! contiguous and neither reduction needs cross-thread communication. The
//! data is then standardised in place and the correlation matrix is the
//! Gram matrix of the standardised data, which reuses the tiled
//! [`crate::covar::covar_symmat`] kernel. A final one-thread-per-row kernel
//! forces the diagonal to exactly 1, as PolyBench does.

use crunchy::unroll;
use gpu::*;

use crate::covar::{BDIM, CENTER_BX, CENTER_BY, KTILE, RED_BDIM, TILE, covar_mean, covar_symmat};

const EPS: f32 = 0.1;

/// `stddev[j] = sqrt(sum_i (data[i][j] - mean[j])^2 / n)`, clamped to 1 when
/// it degenerates.
#[gpu::cuda_kernel]
pub fn corr_stddev(data: &[f32], mean: &[f32], stddev: &mut [f32], n: u32, m: u32, inv_n: f32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let j = block_id::<DimX>() * RED_BDIM + thread_id::<DimX>();
    let mu = mean[j as usize];

    let mut acc = [0.0f32; 4];
    let mut i = 0u32;
    while i < n {
        unroll! {
            for u in 0..4 {
                let d = data[((i + u as u32) * m + j) as usize] - mu;
                acc[u] += d * d;
            }
        }
        i += 4;
    }
    let var = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * inv_n;
    let s = var.sqrt();

    let mut out = chunk_mut(stddev, MapContinuousLinear::new(1));
    out[0] = if s <= EPS { 1.0 } else { s };
}

/// `data[i][j] = (data[i][j] - mean[j]) / (sqrt(n) * stddev[j])`.
#[gpu::cuda_kernel]
pub fn corr_standardise(
    data: &mut [f32],
    mean: &[f32],
    stddev: &[f32],
    sqrt_n: f32,
) {
    assert!(Config::BDIM_X == CENTER_BX);
    assert!(Config::BDIM_Y == CENTER_BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * CENTER_BX + thread_id::<DimX>();

    let mut out =
        chunk_mut(data, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    out[0] = (out[0] - mean[j as usize]) / (sqrt_n * stddev[j as usize]);
}

/// `symmat[j][j] = 1`.
///
/// The map gives thread `j` the single element at `j * (m + 1)`, the `j`-th
/// diagonal entry, which is disjoint across threads by construction.
#[gpu::cuda_kernel]
pub fn corr_unit_diagonal(symmat: &mut [f32], m: u32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let gx = grid_dim::<DimX>();
    let mut out =
        chunk_mut(symmat, reshape_map!([(1, m + 1)] | [256, gx] => layout: [i0, t0, t1]));
    out[0] = 1.0;
}

/// CPU reference: returns the `m x m` correlation matrix.
pub fn corr_cpu(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut d = data.to_vec();
    let nf = n as f32;
    let mut mean = vec![0.0f32; m];
    for j in 0..m {
        let mut s = 0.0f32;
        for i in 0..n {
            s += d[i * m + j];
        }
        mean[j] = s / nf;
    }
    let mut stddev = vec![0.0f32; m];
    for j in 0..m {
        let mut s = 0.0f32;
        for i in 0..n {
            let t = d[i * m + j] - mean[j];
            s += t * t;
        }
        let sd = (s / nf).sqrt();
        stddev[j] = if sd <= EPS { 1.0 } else { sd };
    }
    for i in 0..n {
        for j in 0..m {
            d[i * m + j] = (d[i * m + j] - mean[j]) / (nf.sqrt() * stddev[j]);
        }
    }
    let mut symmat = vec![0.0f32; m * m];
    for j1 in 0..m {
        for j2 in 0..m {
            let mut s = 0.0f32;
            for i in 0..n {
                s += d[i * m + j1] * d[i * m + j2];
            }
            symmat[j1 * m + j2] = s;
        }
    }
    for j in 0..m {
        symmat[j * m + j] = 1.0;
    }
    symmat
}

pub fn corr_gpu(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    assert!(n % KTILE as usize == 0, "corr requires n % {KTILE} == 0");
    assert!(m % TILE as usize == 0, "corr requires m % {TILE} == 0");
    assert!(m % RED_BDIM as usize == 0, "corr requires m % {RED_BDIM} == 0");

    gpu_host::cuda_ctx(0, |ctx, mo| {
        let mut dd = ctx.new_tensor_view(data).unwrap();
        let z = vec![0.0f32; m];
        let mut dmean = ctx.new_tensor_view(z.as_slice()).unwrap();
        let mut dstd = ctx.new_tensor_view(z.as_slice()).unwrap();
        let zsym = vec![0.0f32; m * m];
        let mut dsym = ctx.new_tensor_view(zsym.as_slice()).unwrap();

        let red_grid = (m / RED_BDIM as usize) as u32;
        let cfg = gpu_host::gpu_config!(red_grid, 1, 1, @const RED_BDIM, 1, 1, 0);
        covar_mean::launch(cfg, ctx, mo, &dd, &mut dmean, n as u32, m as u32, 1.0 / n as f32)
            .unwrap();

        let cfg = gpu_host::gpu_config!(red_grid, 1, 1, @const RED_BDIM, 1, 1, 0);
        corr_stddev::launch(
            cfg, ctx, mo, &dd, &dmean, &mut dstd, n as u32, m as u32, 1.0 / n as f32,
        )
        .unwrap();

        let cfg = gpu_host::gpu_config!(
            (m / CENTER_BX as usize) as u32, (n / CENTER_BY as usize) as u32, 1,
            @const CENTER_BX, @const CENTER_BY, 1, 0);
        corr_standardise::launch(cfg, ctx, mo, &mut dd, &dmean, &dstd, (n as f32).sqrt()).unwrap();

        let g = (m / TILE as usize) as u32;
        let cfg = gpu_host::gpu_config!(g, g, 1, @const BDIM, @const BDIM, 1, 0);
        covar_symmat::launch(cfg, ctx, mo, &dd, &mut dsym, n as u32, m as u32).unwrap();

        let cfg = gpu_host::gpu_config!(red_grid, 1, 1, @const RED_BDIM, 1, 1, 0);
        corr_unit_diagonal::launch(cfg, ctx, mo, &mut dsym, m as u32).unwrap();

        let mut h = vec![0.0f32; m * m];
        dsym.copy_to_host(&mut h).unwrap();
        h
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn corr_matches_cpu() {
        let (n, m) = (512usize, 256usize);
        let data = seq(n * m, 161);
        let want = corr_cpu(&data, n, m);
        let got = corr_gpu(&data, n, m);
        assert_close(&got, &want, 1e-4, "corr");
    }
}
