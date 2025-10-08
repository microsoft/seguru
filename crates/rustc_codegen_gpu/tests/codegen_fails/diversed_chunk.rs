// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel]
#[no_mangle]
pub fn test_diversed_chunk(mut a: &mut [f32]) {
    let mut local = gpu::GlobalThreadChunk::new(a, gpu::MapLinear::new(gpu::thread_id::<gpu::DimX>() as usize)); //~ ERROR Invalid use of diversed data in GPU code
    //~| ERROR Invalid use of diversed data in GPU code
    local[0] = 1.0;
}
