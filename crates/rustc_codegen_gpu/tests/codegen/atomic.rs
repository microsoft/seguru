// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_atomic(b: &mut [u8]) {
    let b0 = &mut b[0];
    gpu::atomic_add::<u8>(b0, 1);
    gpu::atomic_add::<u8>(&mut b[1], 1);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry test_atomic
// PTX_CHECK: atom.
