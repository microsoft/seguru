// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_shared(a: &[u8], b: &mut [u8]) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));

    // Below if the Write-Read test
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>()];
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = chunk_shared[0];
    chunked_b[0] = val;

    gpu::sync::sync_threads(); // This is needed

    let z = shared[0];
    for i in 0..10 {
        chunked_b[0] += shared[i];
    }
    chunked_b[0] += shared[0]; 
}

// CHECK: @gpu_bin_cst = internal constant