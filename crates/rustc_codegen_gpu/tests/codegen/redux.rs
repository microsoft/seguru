// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn reduce_max(a: &[u32], _a_window: usize, b: &mut [u32], _b_window: usize) {
    let mut chunked_b = gpu::chunk_mut(b, 1, gpu::GpuChunkIdx::new());
    let val = a[gpu::thread_id(gpu::DimType::X)];
    gpu::add_mlir_string_attr("#nvvm<redux_kind max>");
    chunked_b[0] = gpu::cg::_redux_sync::<_>(val, u32::MAX);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry redux_3A__3A_reduce_max
// PTX_CHECK: redux.sync.max.s32
