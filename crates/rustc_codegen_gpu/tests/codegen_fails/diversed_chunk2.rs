// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::device]
#[no_mangle]
#[inline(never)]
fn dev_call(a: u32, b: u32) -> u32 {
    (a + b) + 1
}

#[gpu_macros::device]
#[inline(always)]
fn dev_call_inline(a: u32, b: u32) -> u32 {
    (a + b) + 1
}


#[gpu_macros::kernel]
pub fn test_diversed_chunk(a: &mut [f32], b: &mut [f32], v: u32) {
    let mut local = gpu::GlobalThreadChunk::new(b, gpu::MapLinear::new(dev_call_inline(1, 2))); // No error here due to inlining
    local[0] = 1.0;
    let mut local = gpu::GlobalThreadChunk::new(a, gpu::MapLinear::new(dev_call(v, 2))); //~ ERROR Invalid use of diversed data in GPU code
    //~| ERROR Invalid use of diversed data in GPU code
    local[0] = 1.0;
}
