// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

#[repr(packed, C)]
pub struct A {
    a: i32,
    b: u32,
}

#[no_mangle]
#[gpu_codegen::kernel]
pub fn assign_with_struct(b: &mut u32) {
    let a = A { a: 1, b: 2 };
    *b = a.b;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry assign_with_struct
// PTX_CHECK: st.global.u32