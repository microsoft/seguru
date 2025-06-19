#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

mod gpu {
#[gpu_codegen::builtin(gpu.thread_id)]
pub fn thread_id() -> usize {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice)]
pub fn subslice<T>(_original: &[T], _offset: usize, _window: usize) -> &[T] {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice_mut)]
pub fn subslice_mut<T>(_original: &mut [T], _offset: usize, _window: usize) -> &mut [T] {
    unimplemented!()
}

/// Add a string attribute to the MLIR module.
#[gpu_codegen::builtin(add_mlir_string_attr)]
pub fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}
}

#[gpu_codegen::device]
#[inline(always)]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
}

#[gpu_codegen::kernel]
pub fn kernel_arith_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) {
    gpu::add_mlir_string_attr("#gpu<dim x>");
    let c = gpu::thread_id();

    let a_local: &[u8] = gpu::subslice(a, c * a_window, a_window);
    let b_local: &mut [u8] = gpu::subslice_mut(b, c * b_window, b_window);

    kernel_arith(a_local, b_local);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: mul.lo.s64
// PTX_CHECK: st.global.u8