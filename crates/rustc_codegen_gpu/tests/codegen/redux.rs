// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

#[gpu::kernel]
#[no_mangle]
pub fn redux_max(a: &[u32], _a_window: usize, b: &mut [u32], _b_window: usize) {
    use gpu::cg::WarpReduceOp;
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = a[gpu::thread_id::<gpu::DimX>() as usize];
    let warp = gpu::cg::ThreadWarpTile::<8>;
    chunked_b[0] = warp.redux(gpu::cg::ReduxMax, val);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry redux_3A__3A_redux_max
// PTX_CHECK: redux.sync.max.u32
