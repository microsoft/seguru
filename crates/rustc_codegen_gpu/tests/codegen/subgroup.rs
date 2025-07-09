// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn subgroup_reduce(a: &[u32], _a_window: usize, b: &mut [u32], _b_window: usize) {
    let chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    let val = a[gpu::thread_id(gpu::DimType::X)];
    gpu::add_mlir_string_attr("#gpu<all_reduce_op add>");
    chunked_b[0] = warp._subgroup_reduce::<_>(val);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry subgroup_reduce
// PTX_CHECK: redux.sync.add.s32