//! Losses.
//!
//! `mse_loss` is a full-tensor reduction, done in two launches: every CTA
//! reduces its tile to one partial with the same warp-shuffle block reduction
//! the row kernels use, then a single CTA reduces the partials and applies the
//! `1/n` scale. Two launches keep the result deterministic (no atomics) and
//! keep the kernel free of grid-wide synchronisation, which SeGuRu does not
//! model.

use crunchy::unroll;
use gpu::*;

use crate::device::block_reduce_sum;
use crate::util::{ELEMS_PER_CTA, EW_BLOCK, ROW_BLOCK, to_float4_padded};

/// Per-CTA partial sums of `(a - b)^2`.
#[gpu::cuda_kernel]
pub fn mse_partial_kernel(a: &[Float4], b: &[Float4], partial: &mut [f32]) {
    assert!(Config::BDIM_X == EW_BLOCK);
    let tid = thread_id::<DimX>();
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + tid;
    let mut out = chunk_mut(
        partial,
        reshape_map!([1] | [(EW_BLOCK, 1), grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );

    let mut acc = 0.0f32;
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            let va = a[i];
            let vb = b[i];
            let d0 = va[0] - vb[0];
            let d1 = va[1] - vb[1];
            let d2 = va[2] - vb[2];
            let d3 = va[3] - vb[3];
            acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
    }
    let total = block_reduce_sum(acc);
    if tid == 0 {
        out[0] = total;
    }
}

/// Final stage: one CTA reduces the per-CTA partials and applies `scale`.
#[gpu::cuda_kernel]
pub fn sum_partials_kernel(partial: &[f32], out: &mut [f32], n: u32, scale: f32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    assert!(Config::GDIM_X == 1);
    let tid = thread_id::<DimX>();
    let mut out = chunk_mut(
        out,
        reshape_map!([1] | [(ROW_BLOCK, 1), grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < n {
        acc += partial[i as usize];
        i += Config::BDIM_X;
    }
    let total = block_reduce_sum(acc);
    if tid == 0 {
        out[0] = total * scale;
    }
}

/// `mse_loss(a, b) = mean((a - b)^2)`.
pub fn mse_loss(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let grid = n.div_ceil(ELEMS_PER_CTA).max(1) as u32;
    let padded = grid as usize * ELEMS_PER_CTA;
    // The padded tail contributes (0 - 0)^2 = 0 to the sum, so the kernel needs
    // no tail predicate.
    let ha = to_float4_padded(a, padded);
    let hb = to_float4_padded(b, padded);
    let inv_n = 1.0 / n as f32;

    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_a = ctx.new_tensor_view::<[Float4]>(&ha).unwrap();
        let d_b = ctx.new_tensor_view::<[Float4]>(&hb).unwrap();
        let zeros = vec![0.0f32; grid as usize];
        let mut d_p = ctx.new_tensor_view::<[f32]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
        mse_partial_kernel::launch(cfg, ctx, m, &d_a, &d_b, &mut d_p).unwrap();

        let out0 = vec![0.0f32; 1];
        let mut d_out = ctx.new_tensor_view::<[f32]>(&out0).unwrap();
        let cfg = gpu_host::gpu_config!(@const 1, 1, 1, @const ROW_BLOCK, 1, 1, 0);
        sum_partials_kernel::launch(cfg, ctx, m, &d_p, &mut d_out, grid, inv_n).unwrap();
        let mut h_out = vec![0.0f32; 1];
        d_out.copy_to_host(&mut h_out).unwrap();
        h_out[0]
    })
}

/// CPU reference, accumulated in `f64`.
pub fn mse_loss_cpu(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    (s / a.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::sample;
    use crate::util::VEC_PER_THREAD;

    #[test]
    fn mse_matches_cpu() {
        for &n in &[1024usize, ELEMS_PER_CTA, 4 * ELEMS_PER_CTA + 3, 1 << 20] {
            let a = sample(n, 51);
            let b = sample(n, 53);
            let g = mse_loss(&a, &b);
            let c = mse_loss_cpu(&a, &b);
            assert!((g - c).abs() <= 1e-5 * c.max(1.0), "mse n={n}: gpu {g} vs cpu {c}");
        }
    }

    #[test]
    fn mse_of_equal_tensors_is_zero() {
        let a = sample(4096, 57);
        assert_eq!(mse_loss(&a, &a), 0.0);
    }

    #[test]
    fn vec_per_thread_is_four() {
        // The kernel body is unrolled by hand for four Float4s per thread.
        assert_eq!(VEC_PER_THREAD, 4);
    }
}
