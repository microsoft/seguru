// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn alloc_shared(a: &[u8], _a_window: usize, b: &mut [u8], b_window: usize) {
    let mut shared: gpu::GpuShared::<[u8; 10]> = gpu::GpuShared::uninit();
    let chunk_shared = gpu::chunk_mut(&mut *shared, 1, gpu::GpuChunkIdx::new());
    let c = chunk_shared;
    c[0] = a[gpu::thread_id(gpu::DimType::X)];
    gpu::sync_threads();

    let chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    for i in 0..10 {
        chunked_b[0] += shared[i];
    }
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry alloc_shared
// PTX_CHECK: bar.sync
// PTX_CHECK: st.shared
// PTX_CHECK: ld.shared
