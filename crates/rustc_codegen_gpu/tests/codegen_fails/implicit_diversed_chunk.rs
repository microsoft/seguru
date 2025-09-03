// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_diversed_implicit(mut a: &mut [f32]) {
    let mut local = if gpu::thread_id(gpu::DimType::X) == 0 {
        gpu::GlobalThreadChunk::new(a, gpu::MapLinear::new(1)) //~ ERROR Invalid use of diversed data in GPU code
    } else {
        gpu::GlobalThreadChunk::new(a, gpu::MapLinear::new(2)) //~ ERROR Invalid use of diversed data in GPU code
    };
    local[0] = 1.0;
}