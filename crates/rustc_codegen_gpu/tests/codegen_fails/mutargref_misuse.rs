// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu::device]
#[inline(never)]
pub fn f(x: &mut u8) {
    *x = 10;
}

#[gpu_codegen::kernel]
#[no_mangle]
pub fn test_mut_arg(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize) {
    let mut tmp = 0; // valid
    let z = b[2]; //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
    let x = &mut b[0]; // valid
    *x = 42; //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
    tmp = 42; // valid
    f(x); //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
    f(&mut tmp); // valid
}
