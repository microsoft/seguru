// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn test_shared_write_read(a: &[u8], b: &mut [u8]) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));//~ ERROR The write needs a `sync_threads` called before other read/write
    //~| ERROR The write needs a `sync_threads` called before other read/write
    //~| ERROR The write needs a `sync_threads` called before other read/write
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>() as usize]; 
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = chunk_shared[0]; // This is safe, as no read/write to other chunk of the shared mem.
    chunked_b[0] = val;

    // gpu::sync::sync_threads();
    let z = shared[0]; //~ NOTE need `sync_threads` before this read/write
    for i in 0..10 {
        chunked_b[0] += shared[i]; //~ NOTE need `sync_threads` before this read/write
    }
    chunked_b[0] += shared[0];  //~ NOTE need `sync_threads` before this read/write
}
