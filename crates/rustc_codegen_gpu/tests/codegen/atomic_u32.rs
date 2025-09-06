// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_atomic(b: &mut [u32]) {
    let b0 = &mut b[0];
    gpu::sync::atomic_addi(b0, 1);
    gpu::sync::atomic_addi(&mut b[1], 1);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry atomic_
// PTX_CHECK: atom.global.add.u32
