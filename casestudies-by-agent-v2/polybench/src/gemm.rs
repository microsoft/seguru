//! GEMM: `C = alpha * A * B + beta * C`.
//!
//! `A` is `NI x NK`, `B` is `NK x NJ`, `C` is `NI x NJ`, all row-major.
//!
//! The kernel is a register-blocked, shared-memory tiled matrix multiply:
//! a 16x16 CTA owns a 64x64 tile of `C`, each thread accumulates a 4x4
//! micro-tile in registers, and the `K` dimension is walked in 16-wide slabs
//! staged through shared memory. Host buffers are padded to the tile geometry
//! so the kernel contains no tail predicate at all.

use crunchy::unroll;
use gpu::*;

/// Threads per CTA in each dimension.
pub const BDIM: u32 = 16;
/// Rows/columns of `C` owned by one CTA.
pub const TILE: u32 = 64;
/// Depth of one `K` slab staged in shared memory.
pub const KTILE: u32 = 16;
/// Outputs per thread in each dimension (`TILE / BDIM`).
pub const TT: u32 = TILE / BDIM;

const SMEM: usize = (KTILE * TILE) as usize;

#[gpu::cuda_kernel]
pub fn gemm_kernel(a: &[f32], b: &[f32], c: &mut [f32], nk: u32, alpha: f32, beta: f32) {
    assert!(Config::BDIM_X == BDIM);
    assert!(Config::BDIM_Y == BDIM);

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let nj = gx * TILE;

    // global = tx + j*16 + bx*64 + ty*nj + i*16*nj + by*64*nj
    let mut cc = chunk_mut(
        c,
        reshape_map!([4, 4] | [16, gx, 16, gy] => layout: [t0, i1, t1, t2, i0, t3]),
    );

    let row0 = block_id::<DimY>() * TILE;
    let col0 = block_id::<DimX>() * TILE;

    let mut as_s = GpuShared::<[f32; SMEM]>::zero();
    let mut bs_s = GpuShared::<[f32; SMEM]>::zero();

    // As is stored k-major: As[k][r]; Bs likewise: Bs[k][c].
    let a_map = reshape_map!([4] | [16, 16] => layout: [t1, i0, t0]);
    let b_map = reshape_map!([4] | [16, 16] => layout: [t0, i0, t1]);

    let mut acc = [[0.0f32; TT as usize]; TT as usize];

    for slab in 0..(nk / KTILE) {
        let kt = slab * KTILE;
        sync_threads();
        {
            let mut ac = as_s.chunk_mut(a_map);
            let mut bc = bs_s.chunk_mut(b_map);
            unroll! {
                for j in 0..4 {
                    let jj = j as u32;
                    ac[jj] = a[((row0 + ty + BDIM * jj) * nk + kt + tx) as usize];
                    bc[jj] = b[((kt + ty) * nj + col0 + tx + BDIM * jj) as usize];
                }
            }
        }
        sync_threads();

        let av = &*as_s;
        let bv = &*bs_s;
        // The K slab is walked 4 at a time rather than as one 16-wide unroll:
        // a fully unrolled 16x(4x4) body overflows the codegen backend's
        // stack, and 4 is already enough for `mem2reg` to keep `af`/`bf` and
        // the accumulator in registers.
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

/// Padded dimensions required by [`gemm_kernel`] for a logical `ni x nj x nk`
/// problem.
pub fn padded_dims(ni: usize, nj: usize, nk: usize) -> (usize, usize, usize) {
    (
        crate::common::round_up(ni, TILE as usize),
        crate::common::round_up(nj, TILE as usize),
        crate::common::round_up(nk, KTILE as usize),
    )
}

/// CPU reference for `C = alpha * A * B + beta * C`.
pub fn gemm_cpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..ni {
        for j in 0..nj {
            let mut acc = 0.0f32;
            for k in 0..nk {
                acc += a[i * nk + k] * b[k * nj + j];
            }
            c[i * nj + j] = alpha * acc + beta * c[i * nj + j];
        }
    }
}

/// Run GEMM on the GPU for a logical `ni x nj x nk` problem, returning the
/// unpadded `ni x nj` result.
pub fn gemm_gpu(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let (pi, pj, pk) = padded_dims(ni, nj, nk);
    let ha = crate::common::pad2(a, ni, nk, pi, pk);
    let hb = crate::common::pad2(b, nk, nj, pk, pj);
    let hc = crate::common::pad2(c, ni, nj, pi, pj);

    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let db = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(
            (pj / TILE as usize) as u32, (pi / TILE as usize) as u32, 1,
            @const BDIM, @const BDIM, 1, 0
        );
        gemm_kernel::launch(cfg, ctx, m, &da, &db, &mut dc, pk as u32, alpha, beta).unwrap();
        let mut h = vec![0.0f32; pi * pj];
        dc.copy_to_host(&mut h).unwrap();
        h
    });
    crate::common::unpad2(&out, ni, nj, pj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn gemm_matches_cpu() {
        let (ni, nj, nk) = (256usize, 192usize, 320usize);
        let a = seq(ni * nk, 1);
        let b = seq(nk * nj, 2);
        let c = seq(ni * nj, 3);
        let (alpha, beta) = (0.5f32, 1.25f32);

        let mut want = c.clone();
        gemm_cpu(&a, &b, &mut want, ni, nj, nk, alpha, beta);
        let got = gemm_gpu(&a, &b, &c, ni, nj, nk, alpha, beta);
        assert_close(&got, &want, 1e-4, "gemm");
    }
}
