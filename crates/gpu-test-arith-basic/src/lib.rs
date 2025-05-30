#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_print(a: &[u8], b: &mut [u8]) {
    let i = 0;
    b[i] = a[i];
}
