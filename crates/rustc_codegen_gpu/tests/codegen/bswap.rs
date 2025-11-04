// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(core_intrinsics)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn bswap(a: u64, b: &mut u64) {
    *b = core::intrinsics::bswap(a);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry bswap
// PTX_CHECK: prmt