//! MVT: `x1 += A y1` and `x2 += A^T y2`, with `A` of shape `N x N`.
//!
//! `x1` reduces along rows (warp-parallel with `Float4` loads), `x2` reduces
//! along columns (one thread per output, coalesced, no reduction).

use crunchy::unroll;
use gpu::*;

use crate::{ix, warp_sum};

pub const MV_BX: u32 = 32;
pub const MV_BY: u32 = 8;
pub const COL_BDIM: u32 = 256;

/// `x1[i] += sum_j A[i][j] * y1[j]`.
#[gpu::cuda_kernel]
pub fn mvt_x1(a: &[Float4], y1: &[Float4], x1: &mut [f32], n4: u32) {
    assert!(Config::BDIM_X == MV_BX);
    assert!(Config::BDIM_Y == MV_BY);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gy = grid_dim::<DimY>();
    let base = (block_id::<DimY>() * MV_BY + ty) * n4;

    // See `crate::ix`.
    let rows = grid_dim::<DimY>() * MV_BY;
    let total = rows * n4;
    if total == 0 || a.len() < total as usize || y1.len() < n4 as usize {
        return;
    }
    let a = &a[..total as usize];
    let y1 = &y1[..n4 as usize];
    let (la, ly) = (total - 1, n4 - 1);

    let mut acc = 0.0f32;
    let mut j = tx;
    while j < n4 {
        let av = a[ix(base + j, la)];
        let yv = y1[ix(j, ly)];
        acc += av[0] * yv[0] + av[1] * yv[1] + av[2] * yv[2] + av[3] * yv[3];
        j += MV_BX;
    }
    let s = warp_sum(acc);

    let mut out = chunk_mut(
        x1,
        reshape_map!([1] | [(32, 1), 1, 8, gy] => layout: [i0, t0, t1, t2, t3]),
    );
    if tx == 0 {
        out[0] = out[0] + s;
    }
}

/// `x2[i] += sum_j A[j][i] * y2[j]`.
#[gpu::cuda_kernel]
pub fn mvt_x2(a: &[f32], y2: &[f32], x2: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == COL_BDIM);
    let i = block_id::<DimX>() * COL_BDIM + thread_id::<DimX>();

    // See `crate::ix`.
    let total = n * n;
    if total == 0 || a.len() < total as usize || y2.len() < n as usize {
        return;
    }
    let a = &a[..total as usize];
    let y2 = &y2[..n as usize];
    let (la, ly) = (total - 1, n - 1);

    let mut acc = [0.0f32; 4];
    let mut j = 0u32;
    while j < n {
        unroll! {
            for u in 0..4 {
                let jj = j + u as u32;
                acc[u] += a[ix(jj * n + i, la)] * y2[ix(jj, ly)];
            }
        }
        j += 4;
    }

    let mut out = chunk_mut(x2, MapContinuousLinear::new(1));
    out[0] = out[0] + (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

/// CPU reference returning `(x1, x2)`.
pub fn mvt_cpu(
    a: &[f32],
    x1: &[f32],
    x2: &[f32],
    y1: &[f32],
    y2: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut x1 = x1.to_vec();
    let mut x2 = x2.to_vec();
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..n {
            s += a[i * n + j] * y1[j];
        }
        x1[i] += s;
    }
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..n {
            s += a[j * n + i] * y2[j];
        }
        x2[i] += s;
    }
    (x1, x2)
}

pub fn mvt_gpu(
    a: &[f32],
    x1: &[f32],
    x2: &[f32],
    y1: &[f32],
    y2: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pn = crate::common::round_up(n, COL_BDIM as usize);
    let ha = crate::common::pad2(a, n, n, pn, pn);
    let ha4 = crate::common::to_float4(&ha);
    let hy1 = crate::common::to_float4(&crate::common::pad1(y1, pn));
    let hy2 = crate::common::pad1(y2, pn);
    let hx1 = crate::common::pad1(x1, pn);
    let hx2 = crate::common::pad1(x2, pn);

    let (o1, o2) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(ha4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let dy1 = ctx.new_tensor_view(hy1.as_slice()).unwrap();
        let dy2 = ctx.new_tensor_view(hy2.as_slice()).unwrap();
        let mut dx1 = ctx.new_tensor_view(hx1.as_slice()).unwrap();
        let mut dx2 = ctx.new_tensor_view(hx2.as_slice()).unwrap();

        let cfg1 = gpu_host::gpu_config!(
            1, (pn / MV_BY as usize) as u32, 1, @const MV_BX, @const MV_BY, 1, 0);
        mvt_x1::launch(cfg1, ctx, m, &da4, &dy1, &mut dx1, (pn / 4) as u32).unwrap();

        let cfg2 = gpu_host::gpu_config!(
            (pn / COL_BDIM as usize) as u32, 1, 1, @const COL_BDIM, 1, 1, 0);
        mvt_x2::launch(cfg2, ctx, m, &da, &dy2, &mut dx2, pn as u32).unwrap();

        let mut o1 = vec![0.0f32; pn];
        let mut o2 = vec![0.0f32; pn];
        dx1.copy_to_host(&mut o1).unwrap();
        dx2.copy_to_host(&mut o2).unwrap();
        (o1, o2)
    });
    (o1[..n].to_vec(), o2[..n].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn mvt_matches_cpu() {
        let n = 1024usize;
        let a = seq(n * n, 31);
        let x1 = seq(n, 32);
        let x2 = seq(n, 33);
        let y1 = seq(n, 34);
        let y2 = seq(n, 35);
        let (w1, w2) = mvt_cpu(&a, &x1, &x2, &y1, &y2, n);
        let (g1, g2) = mvt_gpu(&a, &x1, &x2, &y1, &y2, n);
        assert_close(&g1, &w1, 1e-4, "mvt x1");
        assert_close(&g2, &w2, 1e-4, "mvt x2");
    }
}
