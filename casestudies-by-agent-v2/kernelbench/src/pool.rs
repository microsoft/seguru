//! Pooling.
//!
//! `max_pool1d` over `[batch, channels, length]` with an arbitrary kernel size
//! and stride. Each thread produces [`OUT_PER_THREAD`] outputs strided by the
//! grid, so consecutive lanes write consecutive addresses and the pooling
//! windows they read overlap in L1. The output buffer is padded to a whole
//! number of CTA tiles, and the *input* index is clamped instead of predicated,
//! so the write mapping stays exact and divergence-free.

use crunchy::unroll;
use gpu::*;

pub const POOL_BLOCK: u32 = 256;
pub const OUT_PER_THREAD: u32 = 4;
pub const OUT_PER_CTA: usize = (POOL_BLOCK * OUT_PER_THREAD) as usize;

#[gpu::cuda_kernel]
pub fn max_pool1d_kernel(
    x: &[f32],
    y: &mut [f32],
    kernel_size: u32,
    stride: u32,
    l_in: u32,
    l_out: u32,
    n_out: u32,
) {
    assert!(Config::BDIM_X == POOL_BLOCK);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut out = chunk_mut(y, reshape_map!([OUT_PER_THREAD] | [nthreads] => layout: [t0, i0]));

    unroll! {
        for j in 0..4 {
            // Threads in the padded tail recompute output 0; their slot exists
            // in the padded buffer and is dropped by the host.
            let o = (gid + (j as u32) * nthreads).min(n_out - 1);
            let row = o / l_out;
            let pos = o % l_out;
            let base = (row * l_in + pos * stride) as usize;
            let mut acc = x[base];
            let mut t = 1u32;
            while t < kernel_size {
                acc = acc.max(x[base + t as usize]);
                t += 1;
            }
            out[j as u32] = acc;
        }
    }
}

/// Output length of a 1-D pooling window.
pub fn out_len(l_in: usize, kernel_size: usize, stride: usize) -> usize {
    assert!(l_in >= kernel_size && stride > 0);
    (l_in - kernel_size) / stride + 1
}

/// `max_pool1d` over `[batch, channels, l_in]`.
pub fn max_pool1d(
    x: &[f32],
    batch: usize,
    channels: usize,
    l_in: usize,
    kernel_size: usize,
    stride: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), batch * channels * l_in);
    let l_out = out_len(l_in, kernel_size, stride);
    let n_out = batch * channels * l_out;
    let grid = n_out.div_ceil(OUT_PER_CTA).max(1) as u32;
    let padded = grid as usize * OUT_PER_CTA;

    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[f32]>(x).unwrap();
        let zeros = vec![0.0f32; padded];
        let mut d_y = ctx.new_tensor_view::<[f32]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const POOL_BLOCK, 1, 1, 0);
        max_pool1d_kernel::launch(
            cfg,
            ctx,
            m,
            &d_x,
            &mut d_y,
            kernel_size as u32,
            stride as u32,
            l_in as u32,
            l_out as u32,
            n_out as u32,
        )
        .unwrap();
        let mut h_y = vec![0.0f32; padded];
        d_y.copy_to_host(&mut h_y).unwrap();
        h_y.truncate(n_out);
        h_y
    })
}

pub fn max_pool1d_cpu(
    x: &[f32],
    batch: usize,
    channels: usize,
    l_in: usize,
    kernel_size: usize,
    stride: usize,
) -> Vec<f32> {
    let l_out = out_len(l_in, kernel_size, stride);
    let mut out = Vec::with_capacity(batch * channels * l_out);
    for row in 0..batch * channels {
        for p in 0..l_out {
            let base = row * l_in + p * stride;
            let mut acc = x[base];
            for t in 1..kernel_size {
                acc = acc.max(x[base + t]);
            }
            out.push(acc);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::sample;

    #[test]
    fn max_pool1d_matches_cpu() {
        let cases: &[(usize, usize, usize, usize, usize)] = &[
            (1, 1, 16, 4, 4),
            (16, 64, 128, 4, 4),
            (4, 32, 1000, 3, 2),
            (2, 8, 4096, 5, 1),
        ];
        for &(b, c, l, k, s) in cases {
            let x = sample(b * c * l, 61);
            let g = max_pool1d(&x, b, c, l, k, s);
            let r = max_pool1d_cpu(&x, b, c, l, k, s);
            assert_eq!(g.len(), r.len());
            assert_eq!(g, r, "max_pool1d b={b} c={c} l={l} k={k} s={s}");
        }
    }
}
