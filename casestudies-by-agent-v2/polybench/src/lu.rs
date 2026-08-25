//! LU decomposition, in place, without pivoting.
//!
//! The `k` loop is inherently sequential, so each step is two launches:
//!
//! 1. `lu_rowcol` snapshots the pivot row (already divided by `A[k][k]`) and
//!    the pivot column into two small scratch buffers.
//! 2. `lu_update` applies the rank-1 update to the whole matrix.
//!
//! Splitting it this way is what makes the update kernel expressible at all
//! under SeGuRu's aliasing rules: a kernel cannot hold `A` as both `&[f32]`
//! and `&mut [f32]`, and every thread of `lu_update` needs `A[i][k]` and
//! `A[k][j]`, which belong to other threads' chunks. Routing them through
//! read-only scratch buffers removes the aliasing *and* the race, and it lets
//! the update kernel fold the pivot-row write-back into the same pass.

use gpu::*;

pub const RED_BDIM: u32 = 256;
pub const BX: u32 = 32;
pub const BY: u32 = 8;

/// `row[j] = A[k][j] / A[k][k]` for `j > k` (verbatim otherwise), and
/// `col[i] = A[i][k]`.
#[gpu::cuda_kernel]
pub fn lu_rowcol(a: &[f32], row: &mut [f32], col: &mut [f32], n: u32, k: u32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let t = block_id::<DimX>() * RED_BDIM + thread_id::<DimX>();

    let pivot = a[(k * n + k) as usize];
    let rv = a[(k * n + t) as usize];
    let cv = a[(t * n + k) as usize];

    let mut rout = chunk_mut(row, MapContinuousLinear::new(1));
    rout[0] = if t > k { rv / pivot } else { rv };
    let mut cout = chunk_mut(col, MapContinuousLinear::new(1));
    cout[0] = cv;
}

/// `A[k][*] = row`, and `A[i][j] -= col[i] * row[j]` for `i > k, j > k`.
#[gpu::cuda_kernel]
pub fn lu_update(a: &mut [f32], row: &[f32], col: &[f32], k: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out =
        chunk_mut(a, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    let cur = out[0];
    let upd = cur - col[i as usize] * row[j as usize];
    let v = if i == k {
        row[j as usize]
    } else if i > k && j > k {
        upd
    } else {
        cur
    };
    out[0] = v;
}

/// CPU reference.
pub fn lu_cpu(a: &[f32], n: usize) -> Vec<f32> {
    let mut a = a.to_vec();
    for k in 0..n {
        let pivot = a[k * n + k];
        for j in k + 1..n {
            a[k * n + j] /= pivot;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                a[i * n + j] -= a[i * n + k] * a[k * n + j];
            }
        }
    }
    a
}

/// `n` must be a multiple of [`RED_BDIM`].
pub fn lu_gpu(a: &[f32], n: usize) -> Vec<f32> {
    assert!(n % RED_BDIM as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let mut da = ctx.new_tensor_view(a).unwrap();
        let z = vec![0.0f32; n];
        let mut drow = ctx.new_tensor_view(z.as_slice()).unwrap();
        let mut dcol = ctx.new_tensor_view(z.as_slice()).unwrap();
        let red_grid = (n / RED_BDIM as usize) as u32;
        let gx = (n / BX as usize) as u32;
        let gy = (n / BY as usize) as u32;

        for k in 0..n as u32 {
            let cfg = gpu_host::gpu_config!(red_grid, 1, 1, @const RED_BDIM, 1, 1, 0);
            lu_rowcol::launch(cfg, ctx, m, &da, &mut drow, &mut dcol, n as u32, k).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            lu_update::launch(cfg, ctx, m, &mut da, &drow, &dcol, k).unwrap();
        }

        let mut h = vec![0.0f32; n * n];
        da.copy_to_host(&mut h).unwrap();
        h
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn lu_matches_cpu() {
        let n = 512usize;
        let mut a = seq(n * n, 171);
        // Make the matrix strongly diagonally dominant so that pivot-free LU
        // is numerically well behaved in f32.
        for i in 0..n {
            a[i * n + i] += n as f32;
        }
        let want = lu_cpu(&a, n);
        let got = lu_gpu(&a, n);
        assert_close(&got, &want, 1e-4, "lu");
    }
}
