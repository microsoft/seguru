// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::device]
#[no_mangle]
#[inline(never)]
fn dev_call(a: usize, b: usize) -> usize {
    (a + b) + 1
}

#[gpu_macros::device]
#[inline(always)]
fn dev_call_inline(a: usize, b: usize) -> usize {
    (a + b) + 1
}


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_diversed_chunk(a: &mut [f32], b: &mut [f32]) {
    let local = gpu::chunk_mut(b, dev_call_inline(1, 2), gpu::GpuChunkIdx::new()); // No error here due to inlining
    local[0] = 1.0;
    let local = gpu::chunk_mut(a, dev_call(1, 2), gpu::GpuChunkIdx::new()); //~ ERROR Invalid use of diversed data in GPU code
    local[0] = 1.0;
}
