// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu::kernel]
#[no_mangle]
pub fn shared_chunk_read_after_write_no_sync_needed(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize, f: &mut [f32], salloc: gpu::DynamicSharedAlloc) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>() as usize];
    chunk_shared[0] += 1;
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    chunked_b[0] = chunk_shared[0]; // No sync_threads needed, as no read of chunk owned by other threads.
}

// CHECK: @gpu_bin_cst = internal constant