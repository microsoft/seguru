// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[repr(packed, C)]
pub struct A {
    a: i32,
    c: u64,
    b: u32,
    d: f32,
}

#[no_mangle]
#[gpu_codegen::kernel]
pub fn assign_struct(a: A, b: &mut A) { //~ ERROR Please reorder fields from largest alignment to smallest
    *b = a;
}
