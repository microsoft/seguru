//! Jacobi 2D: `TSTEPS` sweeps of the five-point stencil
//! `B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])`
//! followed by a copy back into `A`, both over the interior only.
//!
//! One thread per point on a 32x8 CTA: `x` walks the unit-stride axis so the
//! three horizontally neighbouring loads are overlapping coalesced streams,
//! and the vertical neighbours hit the L2 from the adjacent CTA rows.

use gpu::*;

use crate::ix;

pub const BX: u32 = 32;
pub const BY: u32 = 8;

#[gpu::cuda_kernel]
pub fn jacobi2d_step(a: &[f32], b: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    // The grid maps exactly onto the array, so the plain linear map already
    // addresses `b[i * n + j]`: `MapContinuousLinear` computes
    // `gid_x + gid_y * gdim_x`. The equivalent `reshape_map!` has to
    // un-flatten the linear thread id and emits a runtime `div.u32`
    // (see README, experiment B).
    let mut out = chunk_mut(b, MapContinuousLinear::new(1));

    // See `crate::ix`.
    let total = n * n;
    if total == 0 || a.len() < total as usize {
        return;
    }
    let a = &a[..total as usize];
    let last = total - 1;

    let jm = j.max(1) - 1;
    let jp = (j + 1).min(n - 1);
    let im = i.max(1) - 1;
    let ip = (i + 1).min(n - 1);

    let v = 0.2f32
        * (a[ix(i * n + j, last)]
            + a[ix(i * n + jm, last)]
            + a[ix(i * n + jp, last)]
            + a[ix(ip * n + j, last)]
            + a[ix(im * n + j, last)]);
    let interior = i > 0 && i + 1 < n && j > 0 && j + 1 < n;
    out[0] = if interior { v } else { out[0] };
}

#[gpu::cuda_kernel]
pub fn jacobi2d_copy(b: &[f32], a: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out = chunk_mut(a, MapContinuousLinear::new(1));
    let total = n * n;
    if total == 0 || b.len() < total as usize {
        return;
    }
    let b = &b[..total as usize];
    let last = total - 1;

    let interior = i > 0 && i + 1 < n && j > 0 && j + 1 < n;
    out[0] = if interior { b[ix(i * n + j, last)] } else { out[0] };
}

/// CPU reference returning `(A, B)` after `tsteps` sweeps.
pub fn jacobi2d_cpu(a: &[f32], b: &[f32], n: usize, tsteps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for _ in 0..tsteps {
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                b[i * n + j] = 0.2
                    * (a[i * n + j]
                        + a[i * n + j - 1]
                        + a[i * n + j + 1]
                        + a[(i + 1) * n + j]
                        + a[(i - 1) * n + j]);
            }
        }
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                a[i * n + j] = b[i * n + j];
            }
        }
    }
    (a, b)
}

/// `n` must be a multiple of `BX * BY / gcd` — in practice a multiple of 32.
pub fn jacobi2d_gpu(a: &[f32], b: &[f32], n: usize, tsteps: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(n % BX as usize == 0 && n % BY as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let mut da = ctx.new_tensor_view(a).unwrap();
        let mut db = ctx.new_tensor_view(b).unwrap();
        let gx = (n / BX as usize) as u32;
        let gy = (n / BY as usize) as u32;
        for _ in 0..tsteps {
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            jacobi2d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            jacobi2d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
        }
        let mut ha = vec![0.0f32; n * n];
        let mut hb = vec![0.0f32; n * n];
        da.copy_to_host(&mut ha).unwrap();
        db.copy_to_host(&mut hb).unwrap();
        (ha, hb)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn jacobi2d_matches_cpu() {
        let (n, tsteps) = (512usize, 10usize);
        let a = seq(n * n, 81);
        let b = seq(n * n, 82);
        let (wa, wb) = jacobi2d_cpu(&a, &b, n, tsteps);
        let (ga, gb) = jacobi2d_gpu(&a, &b, n, tsteps);
        assert_close(&ga, &wa, 1e-4, "jacobi2d a");
        assert_close(&gb, &wb, 1e-4, "jacobi2d b");
    }
}
