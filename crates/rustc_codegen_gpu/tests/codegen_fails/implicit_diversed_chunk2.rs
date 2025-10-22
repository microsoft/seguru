// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu::device]
#[inline(never)]
fn device_call() {
    unimplemented!();
}

#[gpu::kernel]
#[no_mangle]
pub fn test_diversed_implicit(mut a: &mut [f32]) {
    if gpu::thread_id::<gpu::DimX>() == 0 {
        let mut local =  gpu::GlobalThreadChunk::new(a, gpu::MapLinear::new(1)); //~ ERROR Invalid use of diversed data in GPU code
        local[0] = 1.0;
    } else {
        device_call();
    }  
}