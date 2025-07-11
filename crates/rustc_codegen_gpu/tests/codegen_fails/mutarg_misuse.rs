// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::device]
#[inline(never)]
pub fn f(x: &mut u8) {
    *x = 10;
}

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_mut_arg2(mut a: u8) {
    let b = a; //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
    let c = &mut a;
    a = 0; //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
    f(&mut a); //~ ERROR Mutable argument must be used in Valid chunking or atomic functions
}
