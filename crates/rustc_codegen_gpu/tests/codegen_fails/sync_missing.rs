// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize, f: &mut [f32], salloc: gpu::DynamicSharedAlloc) {
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>()]; //~ ERROR The write needs a `sync_threads` called before other read/write
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    chunked_b[0] = chunk_shared[0]; // This is safe, as no read/write to other chunk of the shared mem.

    // gpu::sync::sync_threads();
    let z = shared[0]; //~ NOTE may need `sync_threads` before this read/write
    for i in 0..10 {
        chunked_b[0] += shared[i]; //~ NOTE may need `sync_threads` before this read/write
        //~| NOTE may need `sync_threads` before this read/write
    }
    chunked_b[0] += shared[0];  //~ NOTE may need `sync_threads` before this read/write
}
