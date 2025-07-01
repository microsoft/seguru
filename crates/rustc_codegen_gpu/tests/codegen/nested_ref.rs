// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

pub struct X<'a> {
    a: &'a [u8]
}

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn assign(a: &X<'_>, b: &mut i32) {
    *b = a.a[0] as _;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry assign
