#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[gpu_codegen::device]
#[no_mangle]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
}
