//! SYRK: `C = alpha * A * A^T + beta * C`, with `A` of shape `N x M` and `C`
//! of shape `N x N`.
//!
//! Structurally the same register-blocked tiled kernel as [`crate::gemm`]:
//! a 16x16 CTA owns a 64x64 tile of `C` and each thread keeps a 4x4
//! micro-tile in registers. The only difference is that the second operand is
//! `A` again, read by *row* rather than by column — which is precisely what
//! makes the `A^T` product coalesced: both shared-memory tiles are filled with
//! the same access pattern.

use crunchy::unroll;
use gpu::*;

pub const BDIM: u32 = 16;
pub const TILE: u32 = 64;
pub const KTILE: u32 = 16;
pub const TT: u32 = TILE / BDIM;

const SMEM: usize = (KTILE * TILE) as usize;

#[gpu::cuda_kernel]
pub fn syrk_kernel(a: &[f32], c: &mut [f32], mm: u32, alpha: f32, beta: f32) {
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

    let mut as_s = unsafe { GpuShared::<[f32; SMEM]>::uninit() };
    let mut bs_s = unsafe { GpuShared::<[f32; SMEM]>::uninit() };
    let row_map = reshape_map!([4] | [16, 16] => layout: [t1, i0, t0]);

    let mut acc = [[0.0f32; TT as usize]; TT as usize];

    for slab in 0..(mm / KTILE) {
        let kt = slab * KTILE;
        sync_threads();
        {
            let mut ac = as_s.chunk_mut(row_map);
            let mut bc = bs_s.chunk_mut(row_map);
            unroll! {
                for j in 0..4 {
                    let jj = j as u32;
                    ac[jj] = a[((row0 + ty + BDIM * jj) * mm + kt + tx) as usize];
                    bc[jj] = a[((col0 + ty + BDIM * jj) * mm + kt + tx) as usize];
                }
            }
        }
        sync_threads();

        let av = &*as_s;
        let bv = &*bs_s;
        for kh in 0..(KTILE / 4) {
            unroll! {
                for kl in 0..4 {
                    let base = (kh * 4 + kl as u32) * TILE;
                    let mut af = [0.0f32; TT as usize];
                    let mut bf = [0.0f32; TT as usize];
                    unroll! {
                        for i in 0..4 {
                            af[i] = av[(base + ty + BDIM * (i as u32)) as usize];
                            bf[i] = bv[(base + tx + BDIM * (i as u32)) as usize];
                        }
                    }
                    unroll! {
                        for i in 0..4 {
                            unroll! {
                                for j in 0..4 {
                                    acc[i][j] += af[i] * bf[j];
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
pub fn syrk_cpu(a: &[f32], c: &mut [f32], n: usize, m: usize, alpha: f32, beta: f32) {
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..m {
                s += a[i * m + k] * a[j * m + k];
            }
            c[i * n + j] = alpha * s + beta * c[i * n + j];
        }
    }
}

pub fn syrk_gpu(
    a: &[f32],
    c: &[f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let pn = crate::common::round_up(n, TILE as usize);
    let pm = crate::common::round_up(m, KTILE as usize);
    let ha = crate::common::pad2(a, n, m, pn, pm);
    let hc = crate::common::pad2(c, n, n, pn, pn);

    let out = gpu_host::cuda_ctx(0, |ctx, mo| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let g = (pn / TILE as usize) as u32;
        let cfg = gpu_host::gpu_config!(g, g, 1, @const BDIM, @const BDIM, 1, 0);
        syrk_kernel::launch(cfg, ctx, mo, &da, &mut dc, pm as u32, alpha, beta).unwrap();
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
    fn syrk_matches_cpu() {
        let (n, m) = (256usize, 320usize);
        let a = seq(n * m, 101);
        let c = seq(n * n, 102);
        let (alpha, beta) = (0.75f32, -1.5f32);
        let mut want = c.clone();
        syrk_cpu(&a, &mut want, n, m, alpha, beta);
        let got = syrk_gpu(&a, &c, n, m, alpha, beta);
        assert_close(&got, &want, 1e-4, "syrk");
    }
}
