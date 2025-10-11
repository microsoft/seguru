// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel]
#[no_mangle]
pub fn test_valid_conditional_chunk(a: &mut [f32]) {
    // This is allowed since block_dim() returns a consistent value across all threads.
    let mut local = if gpu::block_dim::<gpu::DimX>() == 1 {
        gpu::chunk_mut(a, gpu::MapLinear::new(1))
    } else {
        gpu::chunk_mut(a, gpu::MapLinear::new(2))
    };
    local[0] = 1.0;
}

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_valid_chunk_size(a: &mut [f32]) {
    // This is allowed since block_dim() returns a consistent value across all threads.
    let mut local = gpu::chunk_mut(a, gpu::MapLinear::new(gpu::block_dim::<gpu::DimX>()));
    local[0] = 1.0;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry conditional_chunk_3A__3A_test_valid_conditional_chunk
// PTX_CHECK: .visible .entry conditional_chunk_3A__3A_test_valid_chunk_size
