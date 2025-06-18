#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_codegen::device]
#[inline(always)]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
}

#[no_mangle]
#[gpu_codegen::kernel]
pub fn kernel_arith_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let c = gpu::thread_id();

    let a_local: &[u8] = gpu::subslice(a, c * a_window, a_window);
    let b_local: &mut [u8] = gpu::subslice_mut(b, c * b_window, b_window);

    kernel_arith(a_local, b_local);
}
