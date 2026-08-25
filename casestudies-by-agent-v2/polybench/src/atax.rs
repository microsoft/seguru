//! ATAX: `y = A^T (A x)` with `A` of shape `NX x NY`.
//!
//! The two passes have opposite access patterns, so they get different
//! parallel decompositions:
//!
//! * `tmp = A x` reduces along a row. One warp owns one row, reads `A` as
//!   `Float4` (128 B per warp per step, perfectly coalesced) and finishes with
//!   a shuffle reduction; lane 0 stores the result.
//! * `y = A^T tmp` reduces along a column, so one thread per output column
//!   already gives coalesced loads and needs no cross-thread reduction.

use crunchy::unroll;
use gpu::*;

use crate::{ix, warp_sum};

/// Lanes cooperating on one row in [`atax_tmp`].
pub const MV_BX: u32 = 32;
/// Rows per CTA in [`atax_tmp`].
pub const MV_BY: u32 = 8;
/// Threads per CTA in [`atax_y`].
pub const COL_BDIM: u32 = 256;

/// `tmp[i] = sum_j A[i][j] * x[j]`, one warp per row.
#[gpu::cuda_kernel]
pub fn atax_tmp(a: &[Float4], x: &[Float4], tmp: &mut [f32], ny4: u32) {
    assert!(Config::BDIM_X == MV_BX);
    assert!(Config::BDIM_Y == MV_BY);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gy = grid_dim::<DimY>();
    let row = block_id::<DimY>() * MV_BY + ty;

    let base = row * ny4;
    // See `crate::ix`.
    let rows = grid_dim::<DimY>() * MV_BY;
    let total = rows * ny4;
    if total == 0 || a.len() < total as usize || x.len() < ny4 as usize {
        return;
    }
    let a = &a[..total as usize];
    let x = &x[..ny4 as usize];
    let (la, lx) = (total - 1, ny4 - 1);

    let mut acc = 0.0f32;
    let mut j = tx;
    while j < ny4 {
        let av = a[ix(base + j, la)];
        let xv = x[ix(j, lx)];
        acc += av[0] * xv[0] + av[1] * xv[1] + av[2] * xv[2] + av[3] * xv[3];
        j += MV_BX;
    }
    let s = warp_sum(acc);

    // Target size 1 on the `tid_x` dimension makes lane 0 the sole owner of
    // the row's slot; the chunk itself is built by every thread, only the
    // store is predicated.
    let mut out = chunk_mut(
        tmp,
        reshape_map!([1] | [(32, 1), 1, 8, gy] => layout: [i0, t0, t1, t2, t3]),
    );
    if tx == 0 {
        out[0] = s;
    }
}

/// `y[j] = sum_i A[i][j] * tmp[i]`, one thread per column.
#[gpu::cuda_kernel]
pub fn atax_y(a: &[f32], tmp: &[f32], y: &mut [f32], nx: u32, ny: u32) {
    assert!(Config::BDIM_X == COL_BDIM);
    let j = block_id::<DimX>() * COL_BDIM + thread_id::<DimX>();

    // See `crate::ix`.
    let total = nx * ny;
    if total == 0 || a.len() < total as usize || tmp.len() < nx as usize {
        return;
    }
    let a = &a[..total as usize];
    let tmp = &tmp[..nx as usize];
    let (la, lt) = (total - 1, nx - 1);

    let mut acc = [0.0f32; 4];
    let mut i = 0u32;
    while i < nx {
        unroll! {
            for u in 0..4 {
                let ii = i + u as u32;
                acc[u] += a[ix(ii * ny + j, la)] * tmp[ix(ii, lt)];
            }
        }
        i += 4;
    }

    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    out[0] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

/// CPU reference.
pub fn atax_cpu(a: &[f32], x: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; nx];
    for i in 0..nx {
        let mut s = 0.0f32;
        for j in 0..ny {
            s += a[i * ny + j] * x[j];
        }
        tmp[i] = s;
    }
    let mut y = vec![0.0f32; ny];
    for j in 0..ny {
        let mut s = 0.0f32;
        for i in 0..nx {
            s += a[i * ny + j] * tmp[i];
        }
        y[j] = s;
    }
    y
}

pub fn atax_gpu(a: &[f32], x: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let pnx = crate::common::round_up(nx, MV_BY as usize);
    let pny = crate::common::round_up(ny, COL_BDIM as usize);
    let ha = crate::common::pad2(a, nx, ny, pnx, pny);
    let ha4 = crate::common::to_float4(&ha);
    let hx4 = crate::common::to_float4(&crate::common::pad1(x, pny));

    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(ha4.as_slice()).unwrap();
        let dx4 = ctx.new_tensor_view(hx4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let ztmp = vec![0.0f32; pnx];
        let mut dtmp = ctx.new_tensor_view(ztmp.as_slice()).unwrap();
        let zy = vec![0.0f32; pny];
        let mut dy = ctx.new_tensor_view(zy.as_slice()).unwrap();

        let cfg1 = gpu_host::gpu_config!(
            1, (pnx / MV_BY as usize) as u32, 1, @const MV_BX, @const MV_BY, 1, 0);
        atax_tmp::launch(cfg1, ctx, m, &da4, &dx4, &mut dtmp, (pny / 4) as u32).unwrap();

        let cfg2 = gpu_host::gpu_config!(
            (pny / COL_BDIM as usize) as u32, 1, 1, @const COL_BDIM, 1, 1, 0);
        atax_y::launch(cfg2, ctx, m, &da, &dtmp, &mut dy, pnx as u32, pny as u32).unwrap();

        let mut h = vec![0.0f32; pny];
        dy.copy_to_host(&mut h).unwrap();
        h
    });
    out[..ny].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn atax_matches_cpu() {
        let (nx, ny) = (1024usize, 768usize);
        let a = seq(nx * ny, 11);
        let x = seq(ny, 12);
        let want = atax_cpu(&a, &x, nx, ny);
        let got = atax_gpu(&a, &x, nx, ny);
        assert_close(&got, &want, 1e-4, "atax");
    }
}
