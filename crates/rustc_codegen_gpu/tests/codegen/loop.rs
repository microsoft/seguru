// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn assign(a: i32, b: &mut [i32; 10]) {
    for i in 0..10 {
        b[i] = a;
    }
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry assign
