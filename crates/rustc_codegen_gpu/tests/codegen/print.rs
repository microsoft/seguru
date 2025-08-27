// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

extern crate gpu;

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_arith(a: &[u8], b: &[u8]) {
    let id = gpu::thread_id(gpu::DimType::X);
    gpu::println!("Hello from GPU! Value: %u %d at %d", a[0], b[0] as u32, id);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry print_3A__3A_kernel_arith
// PTX_CHECK: vprintf,