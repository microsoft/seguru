#[gpu_macros::kernel]
pub fn kernel(input: &[f32; 1]) {
    gpu::println!("Hello world... input = %f", input[0]);
}

#[gpu_macros::host(kernel)]
pub fn host(input: &gpu_host::CudaMemBox<[f32; 1]>) {}

fn main() {
    gpu_host::cuda_ctx(0, |ctx, m| {
        let input = ctx.new_gmem([1.01; 1]).expect("Failed to allocate input");
        let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        host(config, ctx, m, input).expect("Failed to run host arithmetic");
    });
}
