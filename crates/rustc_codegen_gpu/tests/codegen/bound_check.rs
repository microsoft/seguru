// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn bound_check(a: &mut [u8], i: usize) {
    a[i] = 0;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry bound_check
// PTX_CHECK: setp.lt
// PTX_CHECK: selp.b64