// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_shared_read_write(a: &[u8], b: &mut [u8]) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>() as usize];
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = chunk_shared[0]; // This is safe, as no read/write to other chunk of the shared mem.
    chunked_b[0] = val;

    gpu::sync::sync_threads();
    let z = shared[0]; //~ Error The read needs a `sync for write_scope` called before other write
    for i in 0..10 {
        chunked_b[0] += shared[i]; //~ Error The read needs a `sync for write_scope` called before other write
    }
    chunked_b[0] += shared[0]; //~ Error The read needs a `sync for write_scope` called before other write
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1)); //~ NOTE need `sync for write_scope` before this write
    //~| NOTE need `sync for write_scope` before this write
    //~| NOTE need `sync for write_scope` before this write
    chunk_shared[0] = 1; 
}
