//! BiCG sub-kernel: `s = A^T r` and `q = A p`, with `A` of shape `NX x NY`.
//!
//! Same decomposition as [`crate::atax`]: the row reduction is warp-parallel
//! with `Float4` loads, the column reduction is thread-per-column and needs no
//! reduction at all. The two are independent, so they are launched together.

use crunchy::unroll;
use gpu::*;

use crate::{ix, warp_sum};

pub const MV_BX: u32 = 32;
pub const MV_BY: u32 = 8;
pub const COL_BDIM: u32 = 256;

/// `q[i] = sum_j A[i][j] * p[j]`.
#[gpu::cuda_kernel]
pub fn bicg_q(a: &[Float4], p: &[Float4], q: &mut [f32], ny4: u32) {
    assert!(Config::BDIM_X == MV_BX);
    assert!(Config::BDIM_Y == MV_BY);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gy = grid_dim::<DimY>();
    let base = (block_id::<DimY>() * MV_BY + ty) * ny4;

    // See `crate::ix`.
    let rows = grid_dim::<DimY>() * MV_BY;
    let total = rows * ny4;
    if total == 0 || a.len() < total as usize || p.len() < ny4 as usize {
        return;
    }
    let a = &a[..total as usize];
    let p = &p[..ny4 as usize];
    let (la, lx) = (total - 1, ny4 - 1);

    let mut acc = 0.0f32;
    let mut j = tx;
    while j < ny4 {
        let av = a[ix(base + j, la)];
        let pv = p[ix(j, lx)];
        acc += av[0] * pv[0] + av[1] * pv[1] + av[2] * pv[2] + av[3] * pv[3];
        j += MV_BX;
    }
    let s = warp_sum(acc);

    let mut out = chunk_mut(
        q,
        reshape_map!([1] | [(32, 1), 1, 8, gy] => layout: [i0, t0, t1, t2, t3]),
    );
    if tx == 0 {
        out[0] = s;
    }
}

/// `s[j] = sum_i A[i][j] * r[i]`.
#[gpu::cuda_kernel]
pub fn bicg_s(a: &[f32], r: &[f32], s: &mut [f32], nx: u32, ny: u32) {
    assert!(Config::BDIM_X == COL_BDIM);
    let j = block_id::<DimX>() * COL_BDIM + thread_id::<DimX>();

    // See `crate::ix`.
    let total = nx * ny;
    if total == 0 || a.len() < total as usize || r.len() < nx as usize {
        return;
    }
    let a = &a[..total as usize];
    let r = &r[..nx as usize];
    let (la, lt) = (total - 1, nx - 1);

    let mut acc = [0.0f32; 4];
    let mut i = 0u32;
    while i < nx {
        unroll! {
            for u in 0..4 {
                let ii = i + u as u32;
                acc[u] += a[ix(ii * ny + j, la)] * r[ix(ii, lt)];
            }
        }
        i += 4;
    }

    let mut out = chunk_mut(s, MapContinuousLinear::new(1));
    out[0] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

/// CPU reference returning `(s, q)`.
pub fn bicg_cpu(
    a: &[f32],
    p: &[f32],
    r: &[f32],
    nx: usize,
    ny: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = vec![0.0f32; ny];
    let mut q = vec![0.0f32; nx];
    for i in 0..nx {
        let mut acc = 0.0f32;
        for j in 0..ny {
            s[j] += r[i] * a[i * ny + j];
            acc += a[i * ny + j] * p[j];
        }
        q[i] = acc;
    }
    (s, q)
}

pub fn bicg_gpu(
    a: &[f32],
    p: &[f32],
    r: &[f32],
    nx: usize,
    ny: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pnx = crate::common::round_up(nx, MV_BY as usize);
    let pny = crate::common::round_up(ny, COL_BDIM as usize);
    let ha = crate::common::pad2(a, nx, ny, pnx, pny);
    let ha4 = crate::common::to_float4(&ha);
    let hp4 = crate::common::to_float4(&crate::common::pad1(p, pny));
    let hr = crate::common::pad1(r, pnx);

    let (hs, hq) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(ha4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let dp4 = ctx.new_tensor_view(hp4.as_slice()).unwrap();
        let dr = ctx.new_tensor_view(hr.as_slice()).unwrap();
        let zq = vec![0.0f32; pnx];
        let zs = vec![0.0f32; pny];
        let mut dq = ctx.new_tensor_view(zq.as_slice()).unwrap();
        let mut ds = ctx.new_tensor_view(zs.as_slice()).unwrap();

        let cfg1 = gpu_host::gpu_config!(
            1, (pnx / MV_BY as usize) as u32, 1, @const MV_BX, @const MV_BY, 1, 0);
        bicg_q::launch(cfg1, ctx, m, &da4, &dp4, &mut dq, (pny / 4) as u32).unwrap();

        let cfg2 = gpu_host::gpu_config!(
            (pny / COL_BDIM as usize) as u32, 1, 1, @const COL_BDIM, 1, 1, 0);
        bicg_s::launch(cfg2, ctx, m, &da, &dr, &mut ds, pnx as u32, pny as u32).unwrap();

        let mut hs = vec![0.0f32; pny];
        let mut hq = vec![0.0f32; pnx];
        ds.copy_to_host(&mut hs).unwrap();
        dq.copy_to_host(&mut hq).unwrap();
        (hs, hq)
    });
    (hs[..ny].to_vec(), hq[..nx].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn bicg_matches_cpu() {
        let (nx, ny) = (1024usize, 768usize);
        let a = seq(nx * ny, 21);
        let p = seq(ny, 22);
        let r = seq(nx, 23);
        let (ws, wq) = bicg_cpu(&a, &p, &r, nx, ny);
        let (gs, gq) = bicg_gpu(&a, &p, &r, nx, ny);
        assert_close(&gs, &ws, 1e-4, "bicg s");
        assert_close(&gq, &wq, 1e-4, "bicg q");
    }
}
