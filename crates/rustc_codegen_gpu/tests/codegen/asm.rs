// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]
#![feature(asm_experimental_arch)]

use core::arch::asm;
#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn test_asm(a: u32, b: &mut u32) {
   gpu::asm!(
        "mov.u32 {0:reg32}, {1:reg32};",
        out(reg) *b,
        in(reg) a,
    );
    gpu::asm!("membar.gl;");
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry asm_3A__3A_test_asm
// PTX_CHECK: mov.u32
// PTX_CHECK: membar.gl
