//! 2D convolution: a 3x3 stencil `B = conv(A)` over the interior of an
//! `NI x NJ` array, with the border left at zero.
//!
//! Each thread owns a column of `ROWS = 4` consecutive outputs and slides a
//! 3x3 register window down it, so the three input rows are fetched once and
//! reused three times: 18 loads per 4 outputs instead of 36. The `j-1/j/j+1`
//! accesses are three overlapping unit-stride streams, so every warp access is
//! coalesced.

use crunchy::unroll;
use gpu::*;

use crate::ix;

pub const BX: u32 = 32;
pub const BY: u32 = 8;
pub const ROWS: u32 = 4;
/// Rows of `A` covered by one CTA.
pub const CTA_ROWS: u32 = BY * ROWS;

const C11: f32 = 0.2;
const C21: f32 = 0.5;
const C31: f32 = -0.8;
const C12: f32 = -0.3;
const C22: f32 = 0.6;
const C32: f32 = -0.9;
const C13: f32 = 0.4;
const C23: f32 = 0.7;
const C33: f32 = 0.10;

#[gpu::device]
#[inline(always)]
fn row3(a: &[f32], last: u32, ir: u32, nj: u32, jm: u32, j: u32, jp: u32) -> [f32; 3] {
    let base = ir * nj;
    [a[ix(base + jm, last)], a[ix(base + j, last)], a[ix(base + jp, last)]]
}

#[gpu::cuda_kernel]
pub fn conv2d_kernel(a: &[f32], b: &mut [f32], ni: u32, nj: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();

    let j = block_id::<DimX>() * BX + tx;
    let row0 = block_id::<DimY>() * CTA_ROWS + ty * ROWS;

    let mut out =
        chunk_mut(b, reshape_map!([4] | [32, gx, 8, gy] => layout: [t0, t1, i0, t2, t3]));

    // See `crate::ix`.
    let total = ni * nj;
    if total == 0 || a.len() < total as usize {
        return;
    }
    let a = &a[..total as usize];
    let last = total - 1;

    let jm = j.max(1) - 1;
    let jp = (j + 1).min(nj - 1);

    let mut w = [[0.0f32; 3]; 3];
    w[0] = row3(a, last, row0.max(1) - 1, nj, jm, j, jp);
    w[1] = row3(a, last, row0, nj, jm, j, jp);
    w[2] = row3(a, last, (row0 + 1).min(ni - 1), nj, jm, j, jp);

    unroll! {
        for r in 0..4 {
            let i = row0 + r as u32;
            let interior = i > 0 && i + 1 < ni && j > 0 && j + 1 < nj;
            let v = C11 * w[0][0] + C21 * w[0][1] + C31 * w[0][2]
                + C12 * w[1][0] + C22 * w[1][1] + C32 * w[1][2]
                + C13 * w[2][0] + C23 * w[2][1] + C33 * w[2][2];
            out[r as u32] = if interior { v } else { 0.0 };
            w[0] = w[1];
            w[1] = w[2];
            w[2] = row3(a, last, (i + 2).min(ni - 1), nj, jm, j, jp);
        }
    }
}

/// CPU reference; the border of `B` stays zero.
pub fn conv2d_cpu(a: &[f32], ni: usize, nj: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; ni * nj];
    for i in 1..ni - 1 {
        for j in 1..nj - 1 {
            b[i * nj + j] = C11 * a[(i - 1) * nj + (j - 1)]
                + C21 * a[(i - 1) * nj + j]
                + C31 * a[(i - 1) * nj + (j + 1)]
                + C12 * a[i * nj + (j - 1)]
                + C22 * a[i * nj + j]
                + C32 * a[i * nj + (j + 1)]
                + C13 * a[(i + 1) * nj + (j - 1)]
                + C23 * a[(i + 1) * nj + j]
                + C33 * a[(i + 1) * nj + (j + 1)];
        }
    }
    b
}

/// `ni` must be a multiple of [`CTA_ROWS`] and `nj` a multiple of [`BX`]:
/// zero padding would change where the stencil's border lies, so the grid is
/// required to map exactly onto the data instead.
pub fn conv2d_gpu(a: &[f32], ni: usize, nj: usize) -> Vec<f32> {
    assert!(ni % CTA_ROWS as usize == 0 && nj % BX as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a).unwrap();
        let zb = vec![0.0f32; ni * nj];
        let mut db = ctx.new_tensor_view(zb.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(
            (nj / BX as usize) as u32, (ni / CTA_ROWS as usize) as u32, 1,
            @const BX, @const BY, 1, 0);
        conv2d_kernel::launch(cfg, ctx, m, &da, &mut db, ni as u32, nj as u32).unwrap();
        let mut h = vec![0.0f32; ni * nj];
        db.copy_to_host(&mut h).unwrap();
        h
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn conv2d_matches_cpu() {
        let (ni, nj) = (1024usize, 512usize);
        let a = seq(ni * nj, 51);
        let want = conv2d_cpu(&a, ni, nj);
        let got = conv2d_gpu(&a, ni, nj);
        assert_close(&got, &want, 1e-5, "conv2d");
    }
}
