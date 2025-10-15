// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu::kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
pub fn bounded_kernel(a: i32, b: &mut i32) {
    gpu::println!("Hello from GPU!{}", a);
}
// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .maxntid 16, 16, 1
// PTX_CHECK: .minnctapersm 2