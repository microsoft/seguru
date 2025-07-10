#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_codegen::device]
#[no_mangle]
/// assume BK * BK == number of threads in a block x axis.
fn kernel(a: &[u8], b: &mut u8) -> u8 {
    let thread_id_x = gpu::thread_id(gpu::DimType::X);
    *b = a[thread_id_x];
    *b
}

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn kernel_arith(a: &[u8], b: &mut [u8], window: usize) {
    let chunks = gpu::GpuChunksMut::<'_, u8>::new(b, window, gpu::GpuChunkIdx::new());
    let val = gpu::scope(|s| {
        let c = chunks.unique_chunk(s);
        kernel(a, &mut c[0])
    });
    gpu::println!("setting b[?] = %u", val);

    // TODO(datarace): The following code should be disallowed by MIR checker for global memory.
    // Changing chunks should only be possible for shared memory since we can use _thread_sync() to ensure
    // all threads in a block have finished using the prior chunk.
    /* let chunks: gpu::GpuChunksMut<'_, u8> = gpu::gpu_chunk_mut(b, 2, gpu::GpuChunkIdx::new());
    gpu::scope(|s| {
        let c = chunks.next(s);
        kernel(a, &mut c[0]);
    });*/
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry kernel_arith
// PTX_CHECK: [kernel_arith_param_0];
// PTX_CHECK: [kernel_arith_param_1];
// PTX_CHECK: [kernel_arith_param_2];
// PTX_CHECK: [kernel_arith_param_4];
// PTX_CHECK: tid.x;
// PTX_CHECK: tid.y;
// PTX_CHECK: tid.z;
