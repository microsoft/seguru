// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub unsafe fn test(a: &[i16], b: &mut [i16]) {
    let id = gpu::thread_id(gpu::DimType::X);
    let c = #[gpu_codegen::device]|v| {
        v + a[id]
    };
    b[id] = c(b[id]);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry closure
// PTX_CHECK: add.s16
