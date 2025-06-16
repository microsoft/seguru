#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let c = gpu::thread_id();
    b[c] = a[c];
}
