#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use gpu::SafeGpuConfig;

#[gpu_macros::cuda_kernel]
pub fn kernel(input: &[f32; Config::BDIM_X as _]) {
    gpu::println!(
        "Hello world... input = {}",
        input[gpu::thread_id::<gpu::DimX>()]
    );
}

fn main() {
    gpu_host::cuda_ctx(0, |ctx, m| {
        let input = ctx
            .new_tensor_view(&[1.01; 1])
            .expect("Failed to allocate input");
        let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
        kernel::launch(config, ctx, m, &input).expect("Failed to run host arithmetic");
    });
}
