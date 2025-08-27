#![allow(clippy::too_many_arguments)]

#[gpu_macros::host(matmul_gpu::inner_product_kernel)]
pub fn inner_product_kernel(
    a: &gpu_host::CudaMemBox<[f32]>,
    b: &gpu_host::CudaMemBox<[f32]>,
    c: gpu::GpuChunkableMut2D<f32>,
    n: usize,
) {
}
