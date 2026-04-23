// KernelBench Level 1 — argreduce kernels
use gpu::prelude::*;

// KB#51: Argmax reduction — index of max element per row
#[gpu::cuda_kernel(dynamic_shared)]
pub fn argmax_reduce(input: &[f32], output: &mut [u32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();

    let smem_vals = smem_alloc.alloc::<f32>(bdim as usize);
    let smem_idxs = smem_alloc.alloc::<u32>(bdim as usize);

    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];
    let mut local_max = -3.4028235e38_f32;
    let mut local_idx = 0u32;
    let mut i = tid;
    while i < dim {
        let v = row[i as usize];
        if v > local_max {
            local_max = v;
            local_idx = i;
        }
        i += bdim;
    }

    let mut val_chunk = smem_vals.chunk_mut(MapLinear::new(1));
    val_chunk[0] = local_max;
    let mut idx_chunk = smem_idxs.chunk_mut(MapLinear::new(1));
    idx_chunk[0] = local_idx;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut vc = smem_vals.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let mut ic = smem_idxs.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left_val = vc[0];
            let right_val = vc[1];
            if right_val > left_val {
                vc[0] = right_val;
                ic[0] = ic[1];
            }
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), gdim] => layout: [i0, t1, t0]),
        );
        out[0] = *smem_idxs[0];
    }
}

// KB#52: Argmin reduction — index of min element per row
#[gpu::cuda_kernel(dynamic_shared)]
pub fn argmin_reduce(input: &[f32], output: &mut [u32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();

    let smem_vals = smem_alloc.alloc::<f32>(bdim as usize);
    let smem_idxs = smem_alloc.alloc::<u32>(bdim as usize);

    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];
    let mut local_min = 3.4028235e38_f32;
    let mut local_idx = 0u32;
    let mut i = tid;
    while i < dim {
        let v = row[i as usize];
        if v < local_min {
            local_min = v;
            local_idx = i;
        }
        i += bdim;
    }

    let mut val_chunk = smem_vals.chunk_mut(MapLinear::new(1));
    val_chunk[0] = local_min;
    let mut idx_chunk = smem_idxs.chunk_mut(MapLinear::new(1));
    idx_chunk[0] = local_idx;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut vc = smem_vals.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let mut ic = smem_idxs.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left_val = vc[0];
            let right_val = vc[1];
            if right_val < left_val {
                vc[0] = right_val;
                ic[0] = ic[1];
            }
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), gdim] => layout: [i0, t1, t0]),
        );
        out[0] = *smem_idxs[0];
    }
}
