// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_codegen::device]
pub fn test_diversed_chunk(a: &mut [f32]) {
    // `a` must be a unique slice per threads, so we do not need to use chunk_mut here.
    let local = gpu::chunk_mut(a, 1, gpu::GpuChunkIdx::new()); //~ ERROR Invalid use of diversed data in GPU code
    local[0] = 1.0;
}
