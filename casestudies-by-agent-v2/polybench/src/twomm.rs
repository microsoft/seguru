//! 2MM: `D = alpha * A * B * C + beta * D`, evaluated as
//! `tmp = alpha * A * B` followed by `D = tmp * C + beta * D`.
//!
//! Both products are plain GEMMs, so this module reuses
//! [`crate::gemm::gemm_kernel`] rather than duplicating the tiled kernel. The
//! intermediate `tmp` never leaves the device: both launches happen inside a
//! single CUDA context and only `D` is copied back.

use crate::common::{pad2, round_up, unpad2};
use crate::gemm::{BDIM, KTILE, TILE, gemm_kernel};

/// CPU reference.
#[allow(clippy::too_many_arguments)]
pub fn twomm_cpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &mut [f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: f32,
    beta: f32,
) {
    let mut tmp = vec![0.0f32; ni * nj];
    for i in 0..ni {
        for j in 0..nj {
            let mut s = 0.0f32;
            for k in 0..nk {
                s += a[i * nk + k] * b[k * nj + j];
            }
            tmp[i * nj + j] = alpha * s;
        }
    }
    for i in 0..ni {
        for j in 0..nl {
            let mut s = 0.0f32;
            for k in 0..nj {
                s += tmp[i * nj + k] * c[k * nl + j];
            }
            d[i * nl + j] = s + beta * d[i * nl + j];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn twomm_gpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let t = TILE as usize;
    let (pi, pj, pl) = (round_up(ni, t), round_up(nj, t), round_up(nl, t));
    let pk = round_up(nk, KTILE as usize);
    let ha = pad2(a, ni, nk, pi, pk);
    let hb = pad2(b, nk, nj, pk, pj);
    let hc = pad2(c, nj, nl, pj, pl);
    let hd = pad2(d, ni, nl, pi, pl);

    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let db = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let mut dd = ctx.new_tensor_view(hd.as_slice()).unwrap();
        let ztmp = vec![0.0f32; pi * pj];
        let mut dtmp = ctx.new_tensor_view(ztmp.as_slice()).unwrap();

        let cfg = gpu_host::gpu_config!(
            (pj / t) as u32, (pi / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, m, &da, &db, &mut dtmp, pk as u32, alpha, 0.0).unwrap();

        let cfg = gpu_host::gpu_config!(
            (pl / t) as u32, (pi / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, m, &dtmp, &dc, &mut dd, pj as u32, 1.0, beta).unwrap();

        let mut h = vec![0.0f32; pi * pl];
        dd.copy_to_host(&mut h).unwrap();
        h
    });
    unpad2(&out, ni, nl, pl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn twomm_matches_cpu() {
        let (ni, nj, nk, nl) = (192usize, 256usize, 128usize, 192usize);
        let a = seq(ni * nk, 121);
        let b = seq(nk * nj, 122);
        let c = seq(nj * nl, 123);
        let d = seq(ni * nl, 124);
        let (alpha, beta) = (0.5f32, 1.5f32);
        let mut want = d.clone();
        twomm_cpu(&a, &b, &c, &mut want, ni, nj, nk, nl, alpha, beta);
        let got = twomm_gpu(&a, &b, &c, &d, ni, nj, nk, nl, alpha, beta);
        assert_close(&got, &want, 1e-4, "twomm");
    }
}
