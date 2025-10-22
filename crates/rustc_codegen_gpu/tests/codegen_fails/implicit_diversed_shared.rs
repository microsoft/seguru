// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]


#[gpu::kernel]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize, f: &mut [f32], salloc: gpu::DynamicSharedAlloc) {
    let mut salloc = salloc;
    let mut dy_shared = salloc.alloc::<f32>(gpu::thread_id::<gpu::DimX>() as usize); //~ ERROR Invalid use of diversed data in GPU code
    //~| ERROR Invalid use of diversed data in GPU code
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_dy_shared = dy_shared.chunk_mut(gpu::MapLinear::new(1)); //~ ERROR Invalid use of diversed data in GPU code
    //~| ERROR Invalid use of diversed data in GPU code
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(b_window)); // No error.
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>() as usize];
    chunk_dy_shared[0] = 1.1;
    gpu::sync_threads();

    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    for i in 0..10 {
        chunked_b[0] += shared[i];
    }

    let mut chunked_f = gpu::chunk_mut(f, gpu::MapLinear::new(1));
    for i in 0..10 {
        chunked_f[0] += chunk_dy_shared[i];
    }
}
