// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_valid_conditional_chunk(a: &mut [f32]) {
    // This is allowed since block_dim() returns a consistent value across all threads.
    let local = if gpu::block_dim(gpu::DimType::X) == 1 {
        gpu::chunk_mut(a, 1, gpu::GpuChunkIdx::new())
    } else {
        gpu::chunk_mut(a, 2, gpu::GpuChunkIdx::new())
    };
    local[0] = 1.0;
}

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn test_valid_chunk_size(a: &mut [f32]) {
    // This is allowed since block_dim() returns a consistent value across all threads.
    let local = gpu::chunk_mut(a, gpu::block_dim(gpu::DimType::X), gpu::GpuChunkIdx::new());
    local[0] = 1.0;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry conditional_5F_chunk_3A__3A_test_5F_valid_5F_conditional_5F_chunk
// PTX_CHECK: .visible .entry conditional_5F_chunk_3A__3A_test_5F_valid_5F_chunk_5F_size
