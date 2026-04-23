// KernelBench Level 1 — reduction kernels
use gpu::prelude::*;

// KB#47: Sum reduction along dim=1 — one block per row
#[gpu::cuda_kernel(dynamic_shared)]
pub fn sum_reduce(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid * dim;
    let mut local_sum = 0.0f32;
    let mut idx = tid;
    while idx < dim {
        local_sum += input[(row_start + idx) as usize];
        idx += bdim;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), gdim] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0];
    }
}

// KB#48: Mean reduction along dim=1 — sum / dim
#[gpu::cuda_kernel(dynamic_shared)]
pub fn mean_reduce(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid * dim;
    let mut local_sum = 0.0f32;
    let mut idx = tid;
    while idx < dim {
        local_sum += input[(row_start + idx) as usize];
        idx += bdim;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(
            output,
            reshape_map!([1] | [(bdim, 1), gdim] => layout: [i0, t1, t0]),
        );
        out[0] = *smem[0] / dim as f32;
    }
}

// KB#49: Max reduction along dim=1
#[gpu::cuda_kernel(dynamic_shared)]
pub fn max_reduce(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid * dim;
    let mut local_max = -3.4028235e38_f32;
    let mut idx = tid;
    while idx < dim {
        let val = input[(row_start + idx) as usize];
        if val > local_max {
            local_max = val;
        }
        idx += bdim;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_max;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            if right > left {
                sc[0] = right;
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
        out[0] = *smem[0];
    }
}

// KB#53: Min reduction along dim=1
#[gpu::cuda_kernel(dynamic_shared)]
pub fn min_reduce(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid * dim;
    let mut local_min = 3.4028235e38_f32;
    let mut idx = tid;
    while idx < dim {
        let val = input[(row_start + idx) as usize];
        if val < local_min {
            local_min = val;
        }
        idx += bdim;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_min;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            if right < left {
                sc[0] = right;
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
        out[0] = *smem[0];
    }
}
