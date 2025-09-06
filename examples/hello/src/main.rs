#[gpu_macros::kernel]
pub fn kernel(input: &[f32; 1]) {
    gpu::println!("Hello world... input = {}", input[0]);
}

#[gpu_macros::host(kernel)]
pub fn host(input: &gpu_host::CudaMemBox<[f32; 1]>) {}

#[gpu_macros::kernel]
pub fn kernel2(input: &mut [f32]) {
    let chunk = gpu::GlobalThreadChunk::new(input, gpu::MapLinearWithDim::<1>::new(1));
    gpu::println!("Hello world... input = {}", chunk[10]);
}

#[gpu_macros::host(kernel2)]
pub fn host2(input: &mut gpu_host::CudaMemBox<[f32]>) {}

fn main() {
    gpu_host::cuda_ctx(0, |ctx, m| {
        let input = ctx.new_gmem([1.01; 1]).expect("Failed to allocate input");
        let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        host(config, ctx, m, input).expect("Failed to run host arithmetic");
    });

    gpu_host::cuda_ctx(0, |ctx, m| {
        let input = ctx
            .new_gmem_with_len(1, &[1.01])
            .expect("Failed to allocate input");
        let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        host2(config, ctx, m, input).expect("Failed to run host arithmetic");
    });
}
