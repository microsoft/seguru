//! Modified Gram-Schmidt QR factorisation of an `N x M` matrix `A`.
//!
//! Like LU this is a sequential outer loop over columns `k`, with four
//! launches per step:
//!
//! 1. `gs_norm`   — `R[k][k] = ||A[:,k]||`, a single-CTA reduction.
//! 2. `gs_q`      — `Q[:,k] = A[:,k] / R[k][k]`.
//! 3. `gs_r`      — `R[k][j] = Q[:,k] . A[:,j]` for `j > k`, one thread per `j`.
//! 4. `gs_update` — `A[:,j] -= Q[:,k] * R[k][j]` for `j > k`.
//!
//! Steps 1 and 2 could be fused with a grid-wide barrier, but keeping them
//! separate means every kernel is a plain data-parallel map and no kernel
//! aliases a buffer as both read and write.

use crunchy::unroll;
use gpu::*;

use crate::warp_sum;

pub const RED_BDIM: u32 = 256;
pub const BX: u32 = 32;
pub const BY: u32 = 8;

/// `R[k][k] = sqrt(sum_i A[i][k]^2)`, computed by a single CTA of
/// [`RED_BDIM`] threads.
#[gpu::cuda_kernel]
pub fn gs_norm(a: &[f32], r: &mut [f32], n: u32, m: u32, k: u32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let t = thread_id::<DimX>();

    let mut acc = 0.0f32;
    let mut i = t;
    while i < n {
        let v = a[(i * m + k) as usize];
        acc += v * v;
        i += RED_BDIM;
    }
    let ws = warp_sum(acc);

    let mut smem = GpuShared::<[f32; RED_BDIM as usize]>::zero();
    {
        let mut c = smem.chunk_mut(MapContinuousLinear::new(1));
        c[0] = ws;
    }
    sync_threads();
    let s = &*smem;
    let mut total = 0.0f32;
    unroll! {
        for w in 0..8 {
            total += s[w * 32];
        }
    }

    // Only lane 0 has a slot in this chunk; the target dim of `t0` is 1.
    let mut out =
        chunk_mut(r, reshape_map!([1] | [(256, 1), 1] => layout: [i0, t0, t1], offset: k * m + k));
    if t == 0 {
        out[0] = total.sqrt();
    }
}

/// `Q[i][k] = A[i][k] / R[k][k]`.
#[gpu::cuda_kernel]
pub fn gs_q(a: &[f32], r: &[f32], q: &mut [f32], m: u32, k: u32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let gx = grid_dim::<DimX>();
    let i = block_id::<DimX>() * RED_BDIM + thread_id::<DimX>();
    let nrm = r[(k * m + k) as usize];

    let mut out = chunk_mut(
        q,
        reshape_map!([(1, m)] | [256, gx] => layout: [i0, t0, t1], offset: k),
    );
    out[0] = a[(i * m + k) as usize] / nrm;
}

/// `R[k][j] = sum_i Q[i][k] * A[i][j]` for `j > k`.
#[gpu::cuda_kernel]
pub fn gs_r(a: &[f32], q: &[f32], r: &mut [f32], n: u32, m: u32, k: u32) {
    assert!(Config::BDIM_X == RED_BDIM);
    let gx = grid_dim::<DimX>();
    let j = block_id::<DimX>() * RED_BDIM + thread_id::<DimX>();

    let mut acc = [0.0f32; 4];
    let mut i = 0u32;
    while i < n {
        unroll! {
            for u in 0..4 {
                let row = (i + u as u32) * m;
                acc[u] += q[(row + k) as usize] * a[(row + j) as usize];
            }
        }
        i += 4;
    }
    let dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);

    let mut out = chunk_mut(
        r,
        reshape_map!([1] | [256, gx] => layout: [i0, t0, t1], offset: k * m),
    );
    out[0] = if j > k { dot } else { out[0] };
}

/// `A[i][j] -= Q[i][k] * R[k][j]` for `j > k`.
#[gpu::cuda_kernel]
pub fn gs_update(a: &mut [f32], q: &[f32], r: &[f32], m: u32, k: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    let gx = grid_dim::<DimX>();
    let gy = grid_dim::<DimY>();
    let j = block_id::<DimX>() * BX + thread_id::<DimX>();
    let i = block_id::<DimY>() * BY + thread_id::<DimY>();

    let mut out =
        chunk_mut(a, reshape_map!([1] | [32, gx, 8, gy] => layout: [i0, t0, t1, t2, t3]));
    let d = q[(i * m + k) as usize] * r[(k * m + j) as usize];
    out[0] = if j > k { out[0] - d } else { out[0] };
}

/// CPU reference. Returns `(A, R, Q)` in PolyBench's final state.
pub fn gramschm_cpu(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut a = a.to_vec();
    let mut r = vec![0.0f32; m * m];
    let mut q = vec![0.0f32; n * m];
    for k in 0..m {
        let mut nrm = 0.0f32;
        for i in 0..n {
            nrm += a[i * m + k] * a[i * m + k];
        }
        r[k * m + k] = nrm.sqrt();
        for i in 0..n {
            q[i * m + k] = a[i * m + k] / r[k * m + k];
        }
        for j in k + 1..m {
            let mut dot = 0.0f32;
            for i in 0..n {
                dot += q[i * m + k] * a[i * m + j];
            }
            r[k * m + j] = dot;
            for i in 0..n {
                a[i * m + j] -= q[i * m + k] * r[k * m + j];
            }
        }
    }
    (a, r, q)
}

/// `n` and `m` must both be multiples of [`RED_BDIM`].
pub fn gramschm_gpu(a: &[f32], n: usize, m: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(n % RED_BDIM as usize == 0);
    assert!(m % RED_BDIM as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, mo| {
        let mut da = ctx.new_tensor_view(a).unwrap();
        let zq = vec![0.0f32; n * m];
        let mut dq = ctx.new_tensor_view(zq.as_slice()).unwrap();
        let zr = vec![0.0f32; m * m];
        let mut dr = ctx.new_tensor_view(zr.as_slice()).unwrap();

        let qgrid = (n / RED_BDIM as usize) as u32;
        let rgrid = (m / RED_BDIM as usize) as u32;
        let gx = (m / BX as usize) as u32;
        let gy = (n / BY as usize) as u32;

        for k in 0..m as u32 {
            let cfg = gpu_host::gpu_config!(1, 1, 1, @const RED_BDIM, 1, 1, 0);
            gs_norm::launch(cfg, ctx, mo, &da, &mut dr, n as u32, m as u32, k).unwrap();
            let cfg = gpu_host::gpu_config!(qgrid, 1, 1, @const RED_BDIM, 1, 1, 0);
            gs_q::launch(cfg, ctx, mo, &da, &dr, &mut dq, m as u32, k).unwrap();
            let cfg = gpu_host::gpu_config!(rgrid, 1, 1, @const RED_BDIM, 1, 1, 0);
            gs_r::launch(cfg, ctx, mo, &da, &dq, &mut dr, n as u32, m as u32, k).unwrap();
            let cfg = gpu_host::gpu_config!(gx, gy, 1, @const BX, @const BY, 1, 0);
            gs_update::launch(cfg, ctx, mo, &mut da, &dq, &dr, m as u32, k).unwrap();
        }

        let mut ha = vec![0.0f32; n * m];
        da.copy_to_host(&mut ha).unwrap();
        let mut hr = vec![0.0f32; m * m];
        dr.copy_to_host(&mut hr).unwrap();
        let mut hq = vec![0.0f32; n * m];
        dq.copy_to_host(&mut hq).unwrap();
        (ha, hr, hq)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn gramschm_matches_cpu() {
        let (n, m) = (512usize, 256usize);
        let a = seq(n * m, 181);
        let (_wa, wr, wq) = gramschm_cpu(&a, n, m);
        let (_ga, gr, gq) = gramschm_gpu(&a, n, m);
        // `A` itself converges to ~0 in its leading columns, so only the
        // meaningful outputs Q and R are compared.
        assert_close(&gq, &wq, 1e-3, "gramschm Q");
        assert_close(&gr, &wr, 1e-3, "gramschm R");
    }
}
