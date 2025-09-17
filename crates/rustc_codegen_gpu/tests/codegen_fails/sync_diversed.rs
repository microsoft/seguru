// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel]
#[no_mangle]
pub fn test_diversed_sync_threads(mut a: &mut [f32]) {
    if gpu::thread_id::<gpu::DimX>() == 0 {
        gpu::sync::sync_threads(); //~ ERROR Invalid use of diversed data in GPU code
    }
}
