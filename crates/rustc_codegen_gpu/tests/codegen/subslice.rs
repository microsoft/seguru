#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_codegen::device]
#[inline(always)]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
}

#[gpu_codegen::kernel]
pub fn kernel_arith_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) {
    let c = gpu::GpuChunkIdx::new().as_usize();

    let b_local: &mut [u8] = gpu::chunk_mut(b, b_window, gpu::GpuChunkIdx::new());

    kernel_arith(a, b_local);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: mul.lo.s64
// PTX_CHECK: st.global.u8
// PTX_CHECK: %tid.x
// PTX_CHECK: %tid.y
// PTX_CHECK: %tid.z
// PTX_CHECK: %ntid.x
// PTX_CHECK: %ntid.y
// PTX_CHECK: %ntid.z
// PTX_CHECK: %nctaid.x
// PTX_CHECK: %nctaid.y
// PTX_CHECK: %nctaid.z

