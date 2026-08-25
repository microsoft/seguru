//! Jacobi 1D: `TSTEPS` sweeps of `B[i] = (A[i-1] + A[i] + A[i+1]) / 3`
//! followed by a copy back into `A`, both over the interior only.
//!
//! Two kernels per time step, exactly as in the PolyBench/GPU original. Both
//! read their destination through the chunk and write it back unchanged
//! outside the interior, which keeps every store unconditional and the border
//! semantics intact.

use gpu::*;

use crate::ix;

pub const BDIM: u32 = 256;

const THIRD: f32 = 0.33333;

#[gpu::cuda_kernel]
pub fn jacobi1d_step(a: &[f32], b: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == BDIM);
    let i = block_id::<DimX>() * BDIM + thread_id::<DimX>();
    let mut out = chunk_mut(b, MapContinuousLinear::new(1));

    // See `crate::ix`.
    if n == 0 || a.len() < n as usize {
        return;
    }
    let a = &a[..n as usize];
    let last = n - 1;

    let im = i.max(1) - 1;
    let ip = (i + 1).min(n - 1);
    let v = THIRD * (a[ix(im, last)] + a[ix(i, last)] + a[ix(ip, last)]);
    let interior = i > 0 && i + 1 < n;
    out[0] = if interior { v } else { out[0] };
}

#[gpu::cuda_kernel]
pub fn jacobi1d_copy(b: &[f32], a: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == BDIM);
    let i = block_id::<DimX>() * BDIM + thread_id::<DimX>();
    let mut out = chunk_mut(a, MapContinuousLinear::new(1));
    // No `crate::ix` here: this kernel has a single load, already guarded by
    // `interior`, and the sub-slice guard costs more PTX than it saves
    // (32 -> 51 instructions). See README, experiment A.
    let interior = i > 0 && i + 1 < n;
    out[0] = if interior { b[i as usize] } else { out[0] };
}

/// CPU reference returning `(A, B)` after `tsteps` sweeps.
pub fn jacobi1d_cpu(a: &[f32], b: &[f32], n: usize, tsteps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for _ in 0..tsteps {
        for i in 1..n - 1 {
            b[i] = THIRD * (a[i - 1] + a[i] + a[i + 1]);
        }
        for i in 1..n - 1 {
            a[i] = b[i];
        }
    }
    (a, b)
}

/// `n` must be a multiple of [`BDIM`].
pub fn jacobi1d_gpu(a: &[f32], b: &[f32], n: usize, tsteps: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(n % BDIM as usize == 0);
    gpu_host::cuda_ctx(0, |ctx, m| {
        let mut da = ctx.new_tensor_view(a).unwrap();
        let mut db = ctx.new_tensor_view(b).unwrap();
        for _ in 0..tsteps {
            let cfg = gpu_host::gpu_config!((n / BDIM as usize) as u32, 1, 1, @const BDIM, 1, 1, 0);
            jacobi1d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
            let cfg = gpu_host::gpu_config!((n / BDIM as usize) as u32, 1, 1, @const BDIM, 1, 1, 0);
            jacobi1d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
        }
        let mut ha = vec![0.0f32; n];
        let mut hb = vec![0.0f32; n];
        da.copy_to_host(&mut ha).unwrap();
        db.copy_to_host(&mut hb).unwrap();
        (ha, hb)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{assert_close, seq};

    #[test]
    fn jacobi1d_matches_cpu() {
        let (n, tsteps) = (8192usize, 20usize);
        let a = seq(n, 71);
        let b = seq(n, 72);
        let (wa, wb) = jacobi1d_cpu(&a, &b, n, tsteps);
        let (ga, gb) = jacobi1d_gpu(&a, &b, n, tsteps);
        assert_close(&ga, &wa, 1e-4, "jacobi1d a");
        assert_close(&gb, &wb, 1e-4, "jacobi1d b");
    }
}
