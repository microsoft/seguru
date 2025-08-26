// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[inline(never)]
#[gpu_codegen::device]
fn foo(a: i32, b: &mut i32) {
    *b = a;
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign(a: i32, b: &mut i32) {
    foo(a, b);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry fncall
