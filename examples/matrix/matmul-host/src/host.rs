#![allow(clippy::too_many_arguments)]

#[gpu::host(matmul_gpu::inner_product_kernel)]
pub fn inner_product_kernel(
    a: &gpu_host::CudaMemBox<[f32]>,
    b: &gpu_host::CudaMemBox<[f32]>,
    c: &mut gpu_host::CudaMemBox<[f32]>,
    n: usize,
) {
}

#[gpu::host(matmul_gpu::inner_product_kernel2)]
pub fn inner_product_kernel2(
    a: &gpu_host::CudaMemBox<[f32]>,
    b: &gpu_host::CudaMemBox<[f32]>,
    c: gpu::GlobalThreadChunk<'_, f32, gpu::Map2D>,
    n: usize,
) {
}
