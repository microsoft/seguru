#![no_std]
#![allow(clippy::too_many_arguments)]

use gpu::*;

type ThreadChunkMatrix2D<'a> = GlobalThreadChunk<'a, f32, Map2D>;

#[gpu_macros::cuda_kernel]
pub fn inner_product_kernel2(a: &[f32], b: &[f32], c: ThreadChunkMatrix2D<'_>, n: usize) {
    let mut row = block_id::<gpu::DimY>() * block_dim::<gpu::DimY>() + thread_id::<gpu::DimY>();
    let mut c = c;
    for i in 0..((n - 1) / gpu::dim::<gpu::DimY>() + 1) {
        let mut col = block_id::<gpu::DimX>() * block_dim::<gpu::DimX>() + thread_id::<gpu::DimX>();
        for j in 0..((n - 1) / gpu::dim::<gpu::DimX>() + 1) {
            if row < n && col < n {
                let mut sum = 0.0;
                let aa = &a[row * n..row * n + n];
                let mut b_idx = col;
                #[expect(clippy::needless_range_loop)]
                for k in 0..aa.len() {
                    sum += aa[k] * b[b_idx];
                    b_idx += n;
                }
                c[(j, i)] = sum;
            }
            col += gpu::dim::<gpu::DimX>();
        }
        row += gpu::dim::<gpu::DimY>();
    }
}

#[cfg(feature = "v1")]
#[gpu_macros::cuda_kernel]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let mut row = block_id::<gpu::DimY>() * block_dim::<gpu::DimY>() + thread_id::<gpu::DimY>();
    for i in 0..((n - 1) / gpu::dim::<gpu::DimY>() + 1) {
        let mut col = block_id::<gpu::DimX>() * block_dim::<gpu::DimX>() + thread_id::<gpu::DimX>();
        for j in 0..((n - 1) / gpu::dim::<gpu::DimX>() + 1) {
            if row < n && col < n {
                let mut sum = 0.0;
                let aa = &a[row * n..row * n + n];
                let mut b_idx = col;
                #[expect(clippy::needless_range_loop)]
                for k in 0..aa.len() {
                    sum += aa[k] * b[b_idx];
                    b_idx += n;
                }
                c[(j, i)] = sum;
            }
            col += gpu::dim::<gpu::DimX>();
        }
        row += gpu::dim::<gpu::DimY>();
    }
}

/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[cfg(feature = "v3")]
#[gpu_macros::cuda_kernel]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: ThreadChunkMatrix2D<'_>, n: usize) {
    let mut row = gpu::block_id::<gpu::DimY>() * gpu::block_dim::<gpu::DimY>()
        + gpu::thread_id::<gpu::DimY>();
    let mut c = c;
    for i in 0..((n - 1) / gpu::dim::<gpu::DimY>() + 1) {
        let mut col = gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
            + gpu::thread_id::<gpu::DimX>();
        for j in 0..((n - 1) / gpu::dim::<gpu::DimX>() + 1) {
            if row < n && col < n {
                let mut sum = 0.0;
                let aa = gpu::iter::GpuIter::new(&a[row * n..row * n + n]);
                let mut b_idx = col;
                for a in aa {
                    sum += *a * b[b_idx];
                    b_idx += n;
                }
                // Here's the cream: updating c
                c[(j, i)] = sum;
            }
            col += gpu::dim::<gpu::DimX>();
        }
        row += gpu::dim::<gpu::DimY>();
    }
}
