//! DOITGEN: `out[r][q][p] = sum_s A[r][q][s] * C4[s][p]`.
//!
//! Flattening the leading two indices turns the kernel into a single
//! `(NR*NQ) x NP x NP` matrix product, so it reuses the tiled
//! [`crate::gemm::gemm_kernel`] directly. That is both simpler and much faster
//! than the PolyBench/GPU formulation, which gives each thread its own
//! `sum` vector and re-reads `C4` from global memory for every `(r, q)` pair.
//!
//! The result is written to a separate buffer rather than in place, which
//! removes the `sum`/`A` copy-back kernel of the original.

use crate::common::{pad2, round_up, unpad2};
use crate::gemm::{BDIM, KTILE, TILE, gemm_kernel};

/// CPU reference.
pub fn doitgen_cpu(a: &[f32], c4: &[f32], nr: usize, nq: usize, np: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nr * nq * np];
    for rq in 0..nr * nq {
        for p in 0..np {
            let mut s = 0.0f32;
            for k in 0..np {
                s += a[rq * np + k] * c4[k * np + p];
            }
            out[rq * np + p] = s;
        }
    }
    out
}

pub fn doitgen_gpu(a: &[f32], c4: &[f32], nr: usize, nq: usize, np: usize) -> Vec<f32> {
    let t = TILE as usize;
    let rows = nr * nq;
    let prows = round_up(rows, t);
    let pp = round_up(np, t.max(KTILE as usize));
    let ha = pad2(a, rows, np, prows, pp);
    let hc = pad2(c4, np, np, pp, pp);

    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let zo = vec![0.0f32; prows * pp];
        let mut dout = ctx.new_tensor_view(zo.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(
            (pp / t) as u32, (prows / t) as u32, 1, @const BDIM, @const BDIM, 1, 0);
        gemm_kernel::launch(cfg, ctx, m, &da, &dc, &mut dout, pp as u32, 1.0, 0.0).unwrap();
        let mut h = vec![0.0f32; prows * pp];
        dout.copy_to_host(&mut h).unwrap();
        h
    });
    unpad2(&out, rows, np, pp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn doitgen_matches_cpu() {
        let (nr, nq, np) = (32usize, 32usize, 128usize);
        let a = seq(nr * nq * np, 141);
        let c4 = seq(np * np, 142);
        let want = doitgen_cpu(&a, &c4, nr, nq, np);
        let got = doitgen_gpu(&a, &c4, nr, nq, np);
        assert_close(&got, &want, 1e-4, "doitgen");
    }
}
