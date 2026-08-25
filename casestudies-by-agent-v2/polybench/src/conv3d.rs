//! 3D convolution: a 3x3x3 stencil `B = conv(A)` over the interior of an
//! `NI x NJ x NK` array. The stencil weights follow the PolyBench/GPU source
//! exactly, including its repeated terms.
//!
//! The grid is three-dimensional: `x` walks `k` (the unit-stride axis, so all
//! loads are coalesced), `y` walks `j`, and the grid's `z` dimension walks `i`.
//! Because `i` comes from the grid rather than a loop, the output map needs no
//! local dimension and the kernel has no tail predicate.

use gpu::*;

use crate::ix;

pub const BX: u32 = 32;
pub const BY: u32 = 8;

const C11: f32 = 2.0;
const C21: f32 = 5.0;
const C31: f32 = -8.0;
const C12: f32 = -3.0;
const C22: f32 = 6.0;
const C32: f32 = -9.0;
const C13: f32 = 4.0;
const C23: f32 = 7.0;
const C33: f32 = 10.0;

#[gpu::device]
#[inline(always)]
fn at(a: &[f32], last: u32, plane: u32, nk: u32, i: u32, j: u32, k: u32) -> f32 {
    a[ix(i * plane + j * nk + k, last)]
}

#[gpu::cuda_kernel]
pub fn conv3d_kernel(a: &[f32], b: &mut [f32], ni: u32, nj: u32, nk: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    assert!(Config::BDIM_Z == 1);

    let k = block_id::<DimX>() * BX + thread_id::<DimX>();
    let j = block_id::<DimY>() * BY + thread_id::<DimY>();
    let i = block_id::<DimZ>();

    // The grid maps exactly onto the array, so the plain linear map already
    // addresses `b[(i * nj + j) * nk + k]`: `MapContinuousLinear` computes
    // `gid_x + (gid_z * gdim_y + gid_y) * gdim_x`, and `gdim_x == nk`,
    // `gdim_y == nj`. The equivalent `reshape_map!` un-flattens the linear
    // thread id and emits two runtime `div.u32` (see README, experiment B).
    let mut out = chunk_mut(b, MapContinuousLinear::new(1));

    // See `crate::ix`: narrowing `a` to its exact extent and clamping every
    // index against `total - 1` is what lets LLVM discharge the slice bounds
    // check, which is 31% of this kernel's runtime.
    let total = ni * nj * nk;
    if total == 0 || a.len() < total as usize {
        return;
    }
    let a = &a[..total as usize];
    let last = total - 1;

    let interior = i > 0 && i + 1 < ni && j > 0 && j + 1 < nj && k > 0 && k + 1 < nk;

    let im = i.max(1) - 1;
    let ip = (i + 1).min(ni - 1);
    let jm = j.max(1) - 1;
    let jp = (j + 1).min(nj - 1);
    let km = k.max(1) - 1;
    let kp = (k + 1).min(nk - 1);

    let plane = nj * nk;

    let v = C11 * at(a, last, plane, nk, im, jm, km)
        + C13 * at(a, last, plane, nk, ip, jm, km)
        + C21 * at(a, last, plane, nk, im, jm, km)
        + C23 * at(a, last, plane, nk, ip, jm, km)
        + C31 * at(a, last, plane, nk, im, jm, km)
        + C33 * at(a, last, plane, nk, ip, jm, km)
        + C12 * at(a, last, plane, nk, i, jm, k)
        + C22 * at(a, last, plane, nk, i, j, k)
        + C32 * at(a, last, plane, nk, i, jp, k)
        + C11 * at(a, last, plane, nk, im, jm, kp)
        + C13 * at(a, last, plane, nk, ip, jm, kp)
        + C21 * at(a, last, plane, nk, im, j, kp)
        + C23 * at(a, last, plane, nk, ip, j, kp)
        + C31 * at(a, last, plane, nk, im, jp, kp)
        + C33 * at(a, last, plane, nk, ip, jp, kp);

    out[0] = if interior { v } else { 0.0 };
}

fn at_cpu(a: &[f32], plane: usize, nk: usize, i: usize, j: usize, k: usize) -> f32 {
    a[i * plane + j * nk + k]
}

/// CPU reference; the border of `B` stays zero.
pub fn conv3d_cpu(a: &[f32], ni: usize, nj: usize, nk: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; ni * nj * nk];
    let plane = nj * nk;
    for i in 1..ni - 1 {
        for j in 1..nj - 1 {
            for k in 1..nk - 1 {
                b[i * plane + j * nk + k] = C11 * at_cpu(a, plane, nk, i - 1, j - 1, k - 1)
                    + C13 * at_cpu(a, plane, nk, i + 1, j - 1, k - 1)
                    + C21 * at_cpu(a, plane, nk, i - 1, j - 1, k - 1)
                    + C23 * at_cpu(a, plane, nk, i + 1, j - 1, k - 1)
                    + C31 * at_cpu(a, plane, nk, i - 1, j - 1, k - 1)
                    + C33 * at_cpu(a, plane, nk, i + 1, j - 1, k - 1)
                    + C12 * at_cpu(a, plane, nk, i, j - 1, k)
                    + C22 * at_cpu(a, plane, nk, i, j, k)
                    + C32 * at_cpu(a, plane, nk, i, j + 1, k)
                    + C11 * at_cpu(a, plane, nk, i - 1, j - 1, k + 1)
                    + C13 * at_cpu(a, plane, nk, i + 1, j - 1, k + 1)
                    + C21 * at_cpu(a, plane, nk, i - 1, j, k + 1)
                    + C23 * at_cpu(a, plane, nk, i + 1, j, k + 1)
                    + C31 * at_cpu(a, plane, nk, i - 1, j + 1, k + 1)
                    + C33 * at_cpu(a, plane, nk, i + 1, j + 1, k + 1);
            }
        }
    }
    b
}

/// `nk` must be a multiple of [`BX`] and `nj` a multiple of [`BY`].
pub fn conv3d_gpu(a: &[f32], ni: usize, nj: usize, nk: usize) -> Vec<f32> {
    assert!(nk % BX as usize == 0 && nj % BY as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a).unwrap();
        let zb = vec![0.0f32; ni * nj * nk];
        let mut db = ctx.new_tensor_view(zb.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(
            (nk / BX as usize) as u32, (nj / BY as usize) as u32, ni as u32,
            @const BX, @const BY, 1, 0);
        conv3d_kernel::launch(cfg, ctx, m, &da, &mut db, ni as u32, nj as u32, nk as u32).unwrap();
        let mut h = vec![0.0f32; ni * nj * nk];
        db.copy_to_host(&mut h).unwrap();
        h
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn conv3d_matches_cpu() {
        let (ni, nj, nk) = (128usize, 128usize, 128usize);
        let a = seq(ni * nj * nk, 61);
        let want = conv3d_cpu(&a, ni, nj, nk);
        let got = conv3d_gpu(&a, ni, nj, nk);
        assert_close(&got, &want, 1e-5, "conv3d");
    }
}
