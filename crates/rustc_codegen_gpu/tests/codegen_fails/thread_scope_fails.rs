#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_codegen::device]
#[no_mangle]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8], b: &mut u8) {
    let thread_id_x = gpu::thread_id(gpu::DimType::X);
    *b = a[thread_id_x];
}

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn kernel_arith(a: &[u8], b: &mut [u8], window: usize) {
    let chunks = gpu::GpuChunksMut::<'_, u8>::new(b, window, gpu::GpuChunkIdx::new());
    let scope = gpu::scope(|s| {
        s //~ ERROR lifetime may not live long enough
    });
    let c = chunks.unique_chunk(scope);
    kernel(a, &mut c[0]);
}
