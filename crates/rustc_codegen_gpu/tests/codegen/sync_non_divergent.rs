// compile-flags: --emit=llvm-ir --emit=mir
// compile-pass
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

extern crate gpu;
use gpu::prelude::*;

#[gpu::device]
#[gpu::attr(sync_data)]
#[inline(never)]
fn device_call() {
    gpu::sync_threads();
}

#[no_mangle]
#[gpu::kernel]
pub fn reduce_per_grid(
    inputs: &[u32],
    partial_sums: &mut [u32],
    mut smem_alloc: gpu::DynamicSharedAlloc,
) {
    let tid = thread_id::<DimX>();
    device_call();
    let block_dim = block_dim::<DimX>();
    let id = tid + block_dim * block_id::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let grid_size = block_dim * grid_dim * 2;
    let smem = smem_alloc.alloc::<u32>(block_dim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut partial_sums_chunk = chunk_mut(
        partial_sums,
        reshape_map!([1] | [(block_dim, 1), grid_dim] => layout: [i0, t1, t0]),
    );

    let mut local_sum = u32::default();
    for i in (id..(inputs.len() as u32)).step_by(grid_size as usize) {
        let left = inputs[i as usize];
        let right = inputs[(i + grid_size / 2) as usize];
        local_sum = local_sum + left + right;
    }
    smem_chunk[0] = local_sum;
    gpu::sync_threads();

    for i in 0..10 {
        if i >= block_dim {
            continue;
        }
        let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [i] => layout: [t0, i0]));
        if tid < i {
            smem_chunk[0] = tid;
        }
        sync_threads();
    }
    if tid == 0 {
        partial_sums_chunk[0] = *smem[0];
    }
}
// CHECK: @gpu_bin_cst = internal constant
// PTX_CHECK: .visible .entry sync_