#![no_std]
#![allow(clippy::too_many_arguments)]

use gpu::*;

type ThreadChunkMatrix2D<'a> = GlobalThreadChunk<'a, f32, 2, Map2D>;

#[cfg(feature = "v1")]
#[gpu_macros::kernel_v2]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: ThreadChunkMatrix2D<'_>, n: usize) {
    let mut row = block_id(DimType::Y) * block_dim(DimType::Y) + thread_id(DimType::Y);
    let mut c = c;
    for i in 0..((n - 1) / gpu::dim(gpu::DimType::Y) + 1) {
        let mut col = block_id(DimType::X) * block_dim(DimType::X) + thread_id(DimType::X);
        for j in 0..((n - 1) / gpu::dim(gpu::DimType::X) + 1) {
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
            col += gpu::dim(gpu::DimType::X);
        }
        row += gpu::dim(gpu::DimType::Y);
    }
}

#[gpu_macros::kernel_v2]
pub fn inner_product_kernel2(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = GlobalThreadChunk::new(c, Map2D::new(n));
    let mut row = block_id(DimType::Y) * block_dim(DimType::Y) + thread_id(DimType::Y);
    for i in 0..((n - 1) / gpu::dim(gpu::DimType::Y) + 1) {
        let mut col = block_id(DimType::X) * block_dim(DimType::X) + thread_id(DimType::X);
        for j in 0..((n - 1) / gpu::dim(gpu::DimType::X) + 1) {
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
            col += gpu::dim(gpu::DimType::X);
        }
        row += gpu::dim(gpu::DimType::Y);
    }
}

/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[cfg(feature = "v3")]
#[gpu_macros::kernel_v2]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: ThreadChunkMatrix2D<'_>, n: usize) {
    let mut row = gpu::block_id(gpu::DimType::Y) * gpu::block_dim(gpu::DimType::Y)
        + gpu::thread_id(gpu::DimType::Y);
    let mut c = c;
    for i in 0..((n - 1) / gpu::dim(gpu::DimType::Y) + 1) {
        let mut col = gpu::block_id(gpu::DimType::X) * gpu::block_dim(gpu::DimType::X)
            + gpu::thread_id(gpu::DimType::X);
        for j in 0..((n - 1) / gpu::dim(gpu::DimType::X) + 1) {
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
            col += gpu::dim(gpu::DimType::X);
        }
        row += gpu::dim(gpu::DimType::Y);
    }
}
