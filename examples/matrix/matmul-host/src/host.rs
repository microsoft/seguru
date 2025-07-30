#![allow(clippy::too_many_arguments)]

mod internal {
    /// Manually inserted function to test the host side API matches the GPU side API.
    /// This should be generated automatically by the macro.
    /// This is needed in order to force the compiler to link the GPU code.
    #[allow(dead_code)]
    fn dummy_api_checker_kernel_launch_wrapper(
        a: &gpu_host::CudaMemBox<[f32]>,
        b: &gpu_host::CudaMemBox<[f32]>,
        c: gpu::GpuChunkableMut2D<f32>,
        n: usize,
    ) {
        matmul_gpu::inner_product_kernel(a, b, c, n);
    }
}

#[gpu_macros::host(matmul_gpu::inner_product_kernel)]
pub fn inner_product_kernel(
    a: &gpu_host::CudaMemBox<[f32]>,
    b: &gpu_host::CudaMemBox<[f32]>,
    c: gpu::GpuChunkableMut2D<f32>,
    n: usize,
) {
}
