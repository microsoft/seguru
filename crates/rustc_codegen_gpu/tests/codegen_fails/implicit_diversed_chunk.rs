// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_diversed_implicit(mut a: &mut [f32]) {
    let local = if gpu::thread_id(gpu::DimType::X) == 0 {
        gpu::chunk_mut(a, 1, gpu::GpuChunkIdx::new()) //~ ERROR Invalid use of diversed data in GPU code
    } else {
        gpu::chunk_mut(a, 2, gpu::GpuChunkIdx::new()) //~ ERROR Invalid use of diversed data in GPU code
    };
    local[0] = 1.0;
}