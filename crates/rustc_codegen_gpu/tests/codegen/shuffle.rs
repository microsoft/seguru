// compile-flags: -C llvm-args=--fp-contract=contract -C llvm-args=--denormal-fp-math=ieee --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

#[gpu_macros::kernel]
#[no_mangle]
pub fn shuffle_redux(a: &[f32], _a_window: usize, b: &mut [f32], _b_window: usize) {
    use gpu::cg::WarpReduceOp;
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = a[gpu::thread_id::<gpu::DimX>() as usize];
    let warp = gpu::cg::ThreadWarpTile::<32>;
    chunked_b[0] = warp.redux(gpu::cg::ReduxAdd, val);
}

#[gpu_macros::kernel]
#[no_mangle]
pub fn shuffle_redux_max(a: &[f32], _a_window: usize, b: &mut [f32], _b_window: usize) {
    use gpu::cg::WarpReduceOp;
    let mut chunked_b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    let val = a[gpu::thread_id::<gpu::DimX>() as usize];
    let warp = gpu::cg::ThreadWarpTile::<32>;
    chunked_b[0] = warp.redux(gpu::cg::ReduxMax, val);
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry shuffle
// PTX_CHECK: shfl.sync.bfly.b32
// PTX_CHECK: add.rn.f32
