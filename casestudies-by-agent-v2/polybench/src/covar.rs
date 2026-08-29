//! COVAR: covariance matrix of an `N x M` data set (`N` observations of `M`
//! variables).
//!
//! Three kernels:
//!
//! 1. `covar_mean` — one thread per variable. Because the data is row-major
//!    with the variable index innermost, thread-per-variable means the loads
//!    of one warp are contiguous, so the column reduction needs no
//!    cross-thread communication *and* stays fully coalesced.
//! 2. `covar_center` — subtract the mean, one thread per element.
//! 3. `covar_symmat` — `symmat = D^T D`. This is the same register-blocked
//!    64x64 / 4x4 tiled kernel as [`crate::gemm`], specialised for a
//!    transposed left operand: both shared tiles are filled by reading `D`
//!    along its unit-stride axis, so no transpose ever materialises in memory.
//!    It is reused by [`crate::corr`].

use crunchy::unroll;
use gpu::*;

pub const BDIM: u32 = 16;
pub const TILE: u32 = 64;
pub const KTILE: u32 = 16;
pub const TT: u32 = TILE / BDIM;
pub const RED_BDIM: u32 = 256;
pub const CENTER_BX: u32 = 32;
pub const CENTER_BY: u32 = 8;

const SMEM: usize = (KTILE * TILE) as usize;

/// `mean[j] = (sum_i data[i][j]) / n`.
#[gpu::cuda_kernel]
pub fn covar_mean(data: &[f32], mean: &mut [f32], n: u32, m: u32, inv_n: f32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let j = block_id::<DimX>() * RED_BDIM + thread_id::<DimX>();

    let mut acc = [0.0f32; 4];
    let mut i = 0u32;
    while i < n {
        unroll! {
            for u in 0..4 {
                acc[u] += data[((i + u as u32) * m + j) as usize];
            }
        }
        i += 4;
    }
    let mut out = chunk_mut(mean, MapContinuousLinear::new(1));
    out[0] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * inv_n;
}

/// `data[i][j] -= mean[j]`.
#[gpu::cuda_kernel]
pub fn covar_center(data: &mut [f32], mean: &[f32]) {
    assert!(Config::BDIM_X == CENTER_BX);
    assert!(Config::BDIM_Y == CENTER_BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * CENTER_BX + thread_id::<DimX>();

    let mut out =
        chunk_mut(data, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    out[0] = out[0] - mean[j as usize];
}

/// `symmat = D^T D` for `D` of shape `n x m`, `symmat` of shape `m x m`.
#[gpu::cuda_kernel]
pub fn covar_symmat(d: &[f32], symmat: &mut [f32], n: u32, m: u32) {
    assert!(Config::BDIM_X == BDIM);
    assert!(Config::BDIM_Y == BDIM);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();

    let mut cc = chunk_mut(
        symmat,
        reshape_map!([4, 4] | [16, gx, 16, gy] => layout: [t0, i1, t1, t2, i0, t3]),
    );

    let row0 = block_id::<DimY>() * TILE;
    let col0 = block_id::<DimX>() * TILE;

    let mut as_s = GpuShared::<[f32; SMEM]>::init(0.0f32);
    let mut bs_s = GpuShared::<[f32; SMEM]>::init(0.0f32);
    // Row `ty` of the slab, columns `tx + 16*j`: unit-stride in `d`.
    let col_map = reshape_map!([4] | [16, 16] => layout: [t0, i0, t1]);

    let mut acc = [[0.0f32; TT as usize]; TT as usize];

    for slab in 0..(n / KTILE) {
        let kt = slab * KTILE;
        sync_threads();
        {
            let mut ac = as_s.chunk_mut(col_map);
            let mut bc = bs_s.chunk_mut(col_map);
            unroll! {
                for j in 0..4 {
                    let jj = j as u32;
                    ac[jj] = d[((kt + ty) * m + row0 + tx + BDIM * jj) as usize];
                    bc[jj] = d[((kt + ty) * m + col0 + tx + BDIM * jj) as usize];
                }
            }
        }
        sync_threads();

        let av = &*as_s;
        let bv = &*bs_s;
        for kh in 0..(KTILE / 4) {
            unroll! {
                for kl in 0..4 {
                    let base = (kh * 4 + kl as u32) * TILE;
                    let mut af = [0.0f32; TT as usize];
                    let mut bf = [0.0f32; TT as usize];
                    unroll! {
                        for i in 0..4 {
                            af[i] = av[(base + ty + BDIM * (i as u32)) as usize];
                            bf[i] = bv[(base + tx + BDIM * (i as u32)) as usize];
                        }
                    }
                    unroll! {
                        for i in 0..4 {
                            unroll! {
                                for j in 0..4 {
                                    acc[i][j] += af[i] * bf[j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    unroll! {
        for i in 0..4 {
            unroll! {
                for j in 0..4 {
                    cc[(i as u32, j as u32)] = acc[i][j];
                }
            }
        }
    }
}

/// CPU reference: returns the `m x m` covariance matrix.
pub fn covar_cpu(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    let mut d = data.to_vec();
    let mut mean = vec![0.0f32; m];
    for j in 0..m {
        let mut s = 0.0f32;
        for i in 0..n {
            s += d[i * m + j];
        }
        mean[j] = s / n as f32;
    }
    for i in 0..n {
        for j in 0..m {
            d[i * m + j] -= mean[j];
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
    symmat
}

pub fn covar_gpu(data: &[f32], n: usize, m: usize) -> Vec<f32> {
    // Zero padding is not usable here: padded rows would be centred to
    // `-mean[j]` and then pollute `symmat`. The grid is required to map
    // exactly onto the data instead.
    assert!(n % KTILE as usize == 0, "covar requires n % {KTILE} == 0");
    assert!(m % TILE as usize == 0, "covar requires m % {TILE} == 0");
    assert!(m % RED_BDIM as usize == 0, "covar requires m % {RED_BDIM} == 0");

    gpu_host::cuda_ctx(0, |ctx, mo| {
        let mut dd = ctx.new_tensor_view(data).unwrap();
        let zmean = vec![0.0f32; m];
        let mut dmean = ctx.new_tensor_view(zmean.as_slice()).unwrap();
        let zsym = vec![0.0f32; m * m];
        let mut dsym = ctx.new_tensor_view(zsym.as_slice()).unwrap();

        let cfg = gpu_host::gpu_config!(
            (m / RED_BDIM as usize) as u32, 1, 1, @const RED_BDIM, 1, 1, 0);
        covar_mean::launch(cfg, ctx, mo, &dd, &mut dmean, n as u32, m as u32, 1.0 / n as f32)
            .unwrap();

        let cfg = gpu_host::gpu_config!(
            (m / CENTER_BX as usize) as u32, (n / CENTER_BY as usize) as u32, 1,
            @const CENTER_BX, @const CENTER_BY, 1, 0);
        covar_center::launch(cfg, ctx, mo, &mut dd, &dmean).unwrap();

        let g = (m / TILE as usize) as u32;
        let cfg = gpu_host::gpu_config!(g, g, 1, @const BDIM, @const BDIM, 1, 0);
        covar_symmat::launch(cfg, ctx, mo, &dd, &mut dsym, n as u32, m as u32).unwrap();

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
    fn covar_matches_cpu() {
        let (n, m) = (512usize, 256usize);
        let data = seq(n * m, 151);
        let want = covar_cpu(&data, n, m);
        let got = covar_gpu(&data, n, m);
        assert_close(&got, &want, 1e-4, "covar");
    }
}
