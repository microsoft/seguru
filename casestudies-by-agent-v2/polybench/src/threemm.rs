//! 3MM: `G = (A * B) * (C * D)`.
//!
//! Three chained GEMMs; like [`crate::twomm`] this reuses
//! [`crate::gemm::gemm_kernel`] and keeps both intermediates (`E` and `F`) on
//! the device.

use crate::common::{pad2, round_up, unpad2};
use crate::gemm::{BDIM, KTILE, TILE, gemm_kernel};

/// CPU reference.
#[allow(clippy::too_many_arguments)]
pub fn threemm_cpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    nm: usize,
) -> Vec<f32> {
    let mut e = vec![0.0f32; ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            let mut s = 0.0f32;
            for k in 0..nk {
                s += a[i * nk + k] * b[k * nj + j];
            }
            e[i * nj + j] = s;
        }
    }
    let mut f = vec![0.0f32; nj * nl];
    for i in 0..nj {
        for j in 0..nl {
            let mut s = 0.0f32;
            for k in 0..nm {
                s += c[i * nm + k] * d[k * nl + j];
            }
            f[i * nl + j] = s;
        }
    }
    let mut g = vec![0.0f32; ni * nl];
    for i in 0..ni {
        for j in 0..nl {
            let mut s = 0.0f32;
            for k in 0..nj {
                s += e[i * nj + k] * f[k * nl + j];
            }
            g[i * nl + j] = s;
        }
    }
    g
}

#[allow(clippy::too_many_arguments)]
pub fn threemm_gpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    nm: usize,
) -> Vec<f32> {
    let t = TILE as usize;
    let (pi, pj, pl) = (round_up(ni, t), round_up(nj, t), round_up(nl, t));
    let pk = round_up(nk, KTILE as usize);
    let pm = round_up(nm, KTILE as usize);
    let ha = pad2(a, ni, nk, pi, pk);
    let hb = pad2(b, nk, nj, pk, pj);
    let hc = pad2(c, nj, nm, pj, pm);
    let hd = pad2(d, nm, nl, pm, pl);

    let out = gpu_host::cuda_ctx(0, |ctx, mo| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let db = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let dd = ctx.new_tensor_view(hd.as_slice()).unwrap();
        let ze = vec![0.0f32; pi * pj];
        let zf = vec![0.0f32; pj * pl];
        let zg = vec![0.0f32; pi * pl];
        let mut de = ctx.new_tensor_view(ze.as_slice()).unwrap();
        let mut df = ctx.new_tensor_view(zf.as_slice()).unwrap();
        let mut dg = ctx.new_tensor_view(zg.as_slice()).unwrap();

        let cfg = gpu_host::gpu_config!(
            (pj / t) as u32, (pi / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, mo, &da, &db, &mut de, pk as u32, 1.0, 0.0).unwrap();

        let cfg = gpu_host::gpu_config!(
            (pl / t) as u32, (pj / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, mo, &dc, &dd, &mut df, pm as u32, 1.0, 0.0).unwrap();

        let cfg = gpu_host::gpu_config!(
            (pl / t) as u32, (pi / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, mo, &de, &df, &mut dg, pj as u32, 1.0, 0.0).unwrap();

        let mut h = vec![0.0f32; pi * pl];
        dg.copy_to_host(&mut h).unwrap();
        h
    });
    unpad2(&out, ni, nl, pl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn threemm_matches_cpu() {
        let (ni, nj, nk, nl, nm) = (192usize, 192usize, 128usize, 256usize, 128usize);
        let a = seq(ni * nk, 131);
        let b = seq(nk * nj, 132);
        let c = seq(nj * nm, 133);
        let d = seq(nm * nl, 134);
        let want = threemm_cpu(&a, &b, &c, &d, ni, nj, nk, nl, nm);
        let got = threemm_gpu(&a, &b, &c, &d, ni, nj, nk, nl, nm);
        // Three chained f32 GEMMs with fast-math enabled; the reference sums in a
        // different order, so allow a slightly wider relative error.
        assert_close(&got, &want, 5e-4, "threemm");
    }
}
