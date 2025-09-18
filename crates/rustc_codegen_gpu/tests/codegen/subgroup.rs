// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

#[gpu_macros::kernel]
#[no_mangle]
pub fn subgroup_reduce(a: &[u32], _a_window: usize, b: &mut [u32], b_window: usize) {
    let mut chunked_b = gpu::GlobalThreadChunk::new(b, gpu::MapLinear::new(b_window));
    let warp = gpu::cg::ThreadWarpTile::<32>;
    let val = a[gpu::thread_id::<gpu::DimX>()];
    chunked_b[0] = warp.subgroup_reduce(gpu::cg::ReduxAdd, val);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry subgroup_
// PTX_CHECK: redux.sync.add.s32