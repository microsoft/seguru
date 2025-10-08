// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_shared(a: &[u8], b: &mut [u8]) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared1 = shared.chunk_mut(gpu::MapLinear::new(1));

    // Below if the Write-Read test
    chunk_shared1[0] = a[gpu::thread_id::<gpu::DimX>() as usize];
    gpu::sync::sync_threads(); // This is needed
    let mut chunk_shared2 = shared.chunk_mut(gpu::MapLinear::new(2));
}

// CHECK: @gpu_bin_cst = internal constant