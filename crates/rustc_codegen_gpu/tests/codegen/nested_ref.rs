// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

pub struct X<'a> {
    a: &'a [u8]
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign(a: &X<'_>, b: &mut i32) {
    *b = a.a[0] as _;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry nested_ref_3A__3A_assign
