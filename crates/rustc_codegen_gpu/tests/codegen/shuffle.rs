// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn shuffle_reduce(a: &[f32], _a_window: usize, b: &mut [f32], _b_window: usize) {
    let mut chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    let val = a[gpu::thread_id::<gpu::DimX>()];
    let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    chunked_b[0] = gpu::cg::reduce_add_f32(warp, val);
}

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn shuffle_reduce_max(a: &[f32], _a_window: usize, b: &mut [f32], _b_window: usize) {
    let mut chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    let val = a[gpu::thread_id::<gpu::DimX>()];
    let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    chunked_b[0] = gpu::cg::reduce_max_f32(warp, val);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry shuffle
// PTX_CHECK: shfl.sync.bfly.b32
// PTX_CHECK: add.rn.f32
