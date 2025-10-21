// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

#[gpu::kernel]
#[no_mangle]
pub fn check_const(in2: u8) {
    const X: [u8; 10] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; 

    for i in 0..10 {
        if in2 == X[i] {
            gpu::println!("found {}", in2);
        }
    }

    const X2: [u8; 2] = [11, 22]; 

    for i in 0..2 {
        if in2 == X2[i] {
            gpu::println!("found {}", in2);
        }
    }
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry const
// PTX_CHECK: {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
// PTX_CHECK: {11, 22}

