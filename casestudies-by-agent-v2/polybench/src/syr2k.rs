//! SYR2K: `C = alpha * A * B^T + alpha * B * A^T + beta * C`, with `A`, `B` of
//! shape `N x M` and `C` of shape `N x N`.
//!
//! Same 64x64 CTA tile / 4x4 register tile geometry as [`crate::syrk`], but
//! four shared-memory tiles are staged per `K` slab (the row and column blocks
//! of both `A` and `B`), so the two rank-`K` updates are accumulated in one
//! pass over the data instead of two.

use crunchy::unroll;
use gpu::*;

pub const BDIM: u32 = 16;
pub const TILE: u32 = 64;
pub const KTILE: u32 = 16;
pub const TT: u32 = TILE / BDIM;

const SMEM: usize = (KTILE * TILE) as usize;

#[gpu::cuda_kernel]
pub fn syr2k_kernel(a: &[f32], b: &[f32], c: &mut [f32], mm: u32, alpha: f32, beta: f32) {
    assert!(Config::BDIM_X == BDIM);
    assert!(Config::BDIM_Y == BDIM);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();

    let mut cc = chunk_mut(
        c,
        reshape_map!([4, 4] | [16, gx, 16, gy] => layout: [t0, i1, t1, t2, i0, t3]),
    );

    let row0 = block_id::<DimY>() * TILE;
    let col0 = block_id::<DimX>() * TILE;

    let mut ar_s = GpuShared::<[f32; SMEM]>::zero();
    let mut ac_s = GpuShared::<[f32; SMEM]>::zero();
    let mut br_s = GpuShared::<[f32; SMEM]>::zero();
    let mut bc_s = GpuShared::<[f32; SMEM]>::zero();
    let row_map = reshape_map!([4] | [16, 16] => layout: [t1, i0, t0]);

    let mut acc = [[0.0f32; TT as usize]; TT as usize];

    for slab in 0..(mm / KTILE) {
        let kt = slab * KTILE;
        sync_threads();
        {
            let mut arc = ar_s.chunk_mut(row_map);
            let mut acc_ = ac_s.chunk_mut(row_map);
            let mut brc = br_s.chunk_mut(row_map);
            let mut bcc = bc_s.chunk_mut(row_map);
            unroll! {
                for j in 0..4 {
                    let jj = j as u32;
                    let ri = ((row0 + ty + BDIM * jj) * mm + kt + tx) as usize;
                    let ci = ((col0 + ty + BDIM * jj) * mm + kt + tx) as usize;
                    arc[jj] = a[ri];
                    acc_[jj] = a[ci];
                    brc[jj] = b[ri];
                    bcc[jj] = b[ci];
                }
            }
        }
        sync_threads();

        let arv = &*ar_s;
        let acv = &*ac_s;
        let brv = &*br_s;
        let bcv = &*bc_s;
        for kh in 0..(KTILE / 4) {
            unroll! {
                for kl in 0..4 {
                    let base = (kh * 4 + kl as u32) * TILE;
                    let mut ar = [0.0f32; TT as usize];
                    let mut br = [0.0f32; TT as usize];
                    let mut acol = [0.0f32; TT as usize];
                    let mut bcol = [0.0f32; TT as usize];
                    unroll! {
                        for i in 0..4 {
                            let ro = (base + ty + BDIM * (i as u32)) as usize;
                            let co = (base + tx + BDIM * (i as u32)) as usize;
                            ar[i] = arv[ro];
                            br[i] = brv[ro];
                            acol[i] = acv[co];
                            bcol[i] = bcv[co];
                        }
                    }
                    unroll! {
                        for i in 0..4 {
                            unroll! {
                                for j in 0..4 {
                                    acc[i][j] += ar[i] * bcol[j] + br[i] * acol[j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    unroll! {
        for i in 0..4 {
            unroll! {
                for j in 0..4 {
                    let idx = (i as u32, j as u32);
                    cc[idx] = alpha * acc[i][j] + beta * cc[idx];
                }
            }
        }
    }
}

/// CPU reference.
pub fn syr2k_cpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..m {
                s += a[i * m + k] * b[j * m + k] + b[i * m + k] * a[j * m + k];
            }
            c[i * n + j] = alpha * s + beta * c[i * n + j];
        }
    }
}

pub fn syr2k_gpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let pn = crate::common::round_up(n, TILE as usize);
    let pm = crate::common::round_up(m, KTILE as usize);
    let ha = crate::common::pad2(a, n, m, pn, pm);
    let hb = crate::common::pad2(b, n, m, pn, pm);
    let hc = crate::common::pad2(c, n, n, pn, pn);

    let out = gpu_host::cuda_ctx(0, |ctx, mo| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let db = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let g = (pn / TILE as usize) as u32;
        let cfg = gpu_host::gpu_config!(g, g, 1, @const BDIM, @const BDIM, 1, 0);
        syr2k_kernel::launch(cfg, ctx, mo, &da, &db, &mut dc, pm as u32, alpha, beta).unwrap();
        let mut h = vec![0.0f32; pn * pn];
        dc.copy_to_host(&mut h).unwrap();
        h
    });
    crate::common::unpad2(&out, n, n, pn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn syr2k_matches_cpu() {
        let (n, m) = (256usize, 256usize);
        let a = seq(n * m, 111);
        let b = seq(n * m, 112);
        let c = seq(n * n, 113);
        let (alpha, beta) = (0.5f32, 2.0f32);
        let mut want = c.clone();
        syr2k_cpu(&a, &b, &mut want, n, m, alpha, beta);
        let got = syr2k_gpu(&a, &b, &c, n, m, alpha, beta);
        assert_close(&got, &want, 1e-4, "syr2k");
    }
}
