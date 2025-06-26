#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    let c = gpu::thread_id(gpu::DimType::X);
    b[c] = a[c];
}
