#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn assign(a: u32, b: &mut u32) -> u32 { //~ ERROR GPU kernel entry function must not return a value
    *b = a;
    a
}
