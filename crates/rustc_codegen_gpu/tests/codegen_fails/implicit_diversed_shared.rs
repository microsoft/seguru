// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize, f: &mut [f32], salloc: gpu::DynamicSharedAlloc) {
    let mut salloc = salloc;
    let mut dy_shared = salloc.alloc::<f32>(gpu::thread_id(gpu::DimType::X));
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let chunk_dy_shared = dy_shared.chunk_mut(1, gpu::GpuSharedChunkIdx::new()); //~ ERROR Invalid use of diversed data in GPU code
    let chunk_shared = shared.chunk_mut(1, gpu::GpuSharedChunkIdx::new());
    let c = chunk_shared;
    c[0] = a[gpu::thread_id(gpu::DimType::X)];
    chunk_dy_shared[0] = 1.1;
    gpu::sync_threads();

    let mut chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    for i in 0..10 {
        chunked_b[0] += shared[i];
    }

    let mut chunked_f = gpu::chunk_mut(f, 1, gpu::GpuChunkIdx::new());
    for i in 0..10 {
        chunked_f[0] += chunk_dy_shared[i];
    }
}
