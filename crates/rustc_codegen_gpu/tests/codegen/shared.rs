// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize, f: &mut [f32], salloc: gpu::DynamicSharedAlloc) {
    let mut salloc = salloc;
    let mut dy_shared = salloc.alloc::<f32>(32);
    let mut shared = gpu::GpuShared::<[u8; 10]>::zero();
    let mut chunk_dy_shared = dy_shared.chunk_mut(gpu::MapLinear::new(1));
    let mut chunk_shared = shared.chunk_mut(gpu::MapLinear::new(1));
    chunk_shared[0] = a[gpu::thread_id::<gpu::DimX>()];
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


// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry shared
// PTX_CHECK: bar.sync
// PTX_CHECK: st.shared
// PTX_CHECK: ld.shared
// PTX_CHECK: static_shared_0
// PTX_CHECK: __dynamic_shmem__0 
