// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

#[repr(packed, C)]
#[derive(Clone, Copy)]
pub struct A {
    a: i32,
    b: u32,
    d: u64,
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign_with_struct(b: &mut u32, c: &mut u32, aa: &A) {
    let a = A { a: 1, b: 2, d: 3 };
    *b = a.b;
    *c = aa.b;
}

#[no_mangle]
#[gpu_codegen::device]
pub unsafe fn _assign_with_struct(b: &mut u32, a: A) {
    *b = a.b;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry assign_with_struct
// PTX_CHECK: st.global.u32
