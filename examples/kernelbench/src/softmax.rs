// KernelBench Level 1 — softmax kernels
use gpu::prelude::*;

// KB#23: softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
// One block per row, 3-pass algorithm with shared memory tree reduction.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn softmax_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    // Pass 1: find row max
    let mut local_max = -3.4028235e38_f32;
    let mut i = tid;
    while i < dim {
        let v = row[i as usize];
        if v > local_max {
            local_max = v;
        }
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_max;
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
    let row_max = *smem[0];
    sync_threads();

    // Pass 2: sum of exp(x - max)
    let mut local_sum = 0.0f32;
    i = tid;
    while i < dim {
        local_sum += (row[i as usize] - row_max).exp();
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();
    stride = bdim / 2;
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
    let row_sum = *smem[0];
    sync_threads();

    // Pass 3: write softmax output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = (row[i as usize] - row_max).exp() / row_sum;
        i += bdim;
        iter_idx += 1;
    }
}

// KB#24: log_softmax(x)_i = (x_i - max(x)) - log(Σ exp(x_j - max(x)))
// Same 3-pass structure as softmax, different final transform.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn log_softmax_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    // Pass 1: find row max
    let mut local_max = -3.4028235e38_f32;
    let mut i = tid;
    while i < dim {
        let v = row[i as usize];
        if v > local_max {
            local_max = v;
        }
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_max;
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
    let row_max = *smem[0];
    sync_threads();

    // Pass 2: sum of exp(x - max)
    let mut local_sum = 0.0f32;
    i = tid;
    while i < dim {
        local_sum += (row[i as usize] - row_max).exp();
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
    sync_threads();
    stride = bdim / 2;
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
    let log_sum = GPUDeviceFloatIntrinsics::log(*smem[0]);
    sync_threads();

    // Pass 3: write log-softmax output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = (row[i as usize] - row_max) - log_sum;
        i += bdim;
        iter_idx += 1;
    }
}
