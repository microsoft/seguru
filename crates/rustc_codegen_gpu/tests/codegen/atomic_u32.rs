// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_atomic(b: &mut u32) {
    let mut atomic_b = gpu::sync::Atomic::new(b);
    atomic_b.atomic_addi(1);
    atomic_b.atomic_addi(1);
    atomic_b.atomic_ori(1);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry atomic_
// PTX_CHECK: atom.global.add.u32
