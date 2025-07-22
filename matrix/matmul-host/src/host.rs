#![allow(clippy::too_many_arguments)]

mod internal {
    /// Manually inserted function to test the host side API matches the GPU side API.
    /// This should be generated automatically by the macro.
    /// This is needed in order to force the compiler to link the GPU code.
    #[allow(dead_code)]
    fn dummy_api_checker_kernel_launch_wrapper(
        a: &cuda_bindings::CudaMemBox<[f32]>,
        b: &cuda_bindings::CudaMemBox<[f32]>,
        c: gpu::GpuChunkableMut2D<f32>,
        n: usize,
    ) {
        matmul_gpu::inner_product_kernel(a, b, c, n);
    }
}

#[gpu_macros::host(matmul_gpu::inner_product_kernel)]
pub fn inner_product_kernel(
    a: &cuda_bindings::CudaMemBox<[f32]>,
    b: &cuda_bindings::CudaMemBox<[f32]>,
    c: gpu::GpuChunkableMut2D<f32>,
    n: usize,
) {
    let config = cuda_bindings::GPUConfig {
        grid_dim_x: 1,
        grid_dim_y: 1,
        grid_dim_z: 1,
        block_dim_x: 2,
        block_dim_y: 2,
        block_dim_z: 1,
    };
}
