// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[repr(packed, C)]
pub struct A {
    a: i32,
    b: u32,
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign_struct(a: A, b: &mut A) { //~ ERROR Does not support fn abi cast
    *b = a;
}


// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry struct_