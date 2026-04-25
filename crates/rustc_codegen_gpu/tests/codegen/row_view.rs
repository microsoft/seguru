// compile-flags: --emit=llvm-ir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

use gpu::{Float4, chunk_mut, reshape_map};

#[gpu::kernel]
#[no_mangle]
pub fn row_view(out: &mut [Float4], input: &[Float4]) {
    let map = reshape_map!(
        [2, 8] | [16, gpu::grid_dim::<gpu::DimX>(), 16, gpu::grid_dim::<gpu::DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let a = input[0];
    let b = input[1];
    let mut tile = chunk_mut(out, map).open_tile();
    let mut row = tile.row_mut(0u32);
    row[0u32] = a;
    row[1u32] = b;
}

// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry row_view
// PTX_CHECK: st.v2.u64
