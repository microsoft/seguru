#![no_std]
#![allow(clippy::too_many_arguments)]

#[allow(non_upper_case_globals)]
#[gpu_macros::shared_size]
pub static shared_size_inner_product_kernel: usize = 0;

/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: gpu::GpuChunkableMut2D<f32>, n: usize) {
    let mut row = gpu::block_id(gpu::DimType::Y) * gpu::block_dim(gpu::DimType::Y)
        + gpu::thread_id(gpu::DimType::Y);
    let mut c = c;
    let c_ref = &mut c;

    let mut i = 0;

    while i < 2 {
        let mut col = gpu::block_id(gpu::DimType::X) * gpu::block_dim(gpu::DimType::X)
            + gpu::thread_id(gpu::DimType::X);
        let mut j = 0;
        while j < 2 {
            if row < n && col < n {
                let mut sum = 0.0;
                let mut k = 0;
                while k < n {
                    sum += a[row * n + k] * b[k * n + col];
                    k += 1;
                }

                // Here's the cream: updating c
                let c_local = gpu::get_local_mut_2d::<f32>(c_ref, j as usize, i as usize);
                *c_local = sum;
            }
            col += gpu::block_dim(gpu::DimType::X);
            j += 1;
        }
        row += gpu::block_dim(gpu::DimType::Y);
        i += 1;
    }
}
