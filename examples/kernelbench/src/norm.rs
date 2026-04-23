// KernelBench Level 1 — norm kernels
use gpu::prelude::*;

// KB#36: rms_norm(x)_i = x_i / sqrt(mean(x²) + eps), per row
// 2-pass: sum-of-squares → normalize
#[gpu::cuda_kernel(dynamic_shared)]
pub fn rms_norm_forward(input: &[f32], output: &mut [f32], dim: u32, eps: f32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;

    // Pass 1: reduce x*x per row
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < dim {
        let v = input[(row_start + i) as usize];
        local_sum += v * v;
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
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
    let rms = (*smem[0] / dim as f32 + eps).sqrt();
    sync_threads();

    // Pass 2: write normalized output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = input[(row_start + i) as usize] / rms;
        i += bdim;
        iter_idx += 1;
    }
}

// KB#37: frobenius_norm = sqrt(Σx²) over entire tensor (scalar output)
// Single block, global reduction
#[gpu::cuda_kernel(dynamic_shared)]
pub fn frobenius_norm_forward(input: &[f32], output: &mut [f32], n: u32) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < n {
        let v = input[i as usize];
        local_sum += v * v;
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
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
            reshape_map!([1] | [(bdim, 1), (1, 1)] => layout: [i0, t1, t0]),
        );
        out[0] = (*smem[0]).sqrt();
    }
}

// KB#38: l1_norm(x)_i = x_i / Σ|x_j|, per row
// 2-pass: sum-of-abs → normalize
#[gpu::cuda_kernel(dynamic_shared)]
pub fn l1_norm_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;

    // Pass 1: reduce |x| per row
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < dim {
        let v = input[(row_start + i) as usize];
        if v < 0.0 {
            local_sum += -v;
        } else {
            local_sum += v;
        }
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
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
    let sum_abs = *smem[0];
    sync_threads();

    // Pass 2: write normalized output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = input[(row_start + i) as usize] / sum_abs;
        i += bdim;
        iter_idx += 1;
    }
}

// KB#39: l2_norm(x)_i = x_i / sqrt(Σx_j²), per row
// 2-pass: sum-of-squares → normalize
#[gpu::cuda_kernel(dynamic_shared)]
pub fn l2_norm_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;

    // Pass 1: reduce x*x per row
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < dim {
        let v = input[(row_start + i) as usize];
        local_sum += v * v;
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
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
    let l2 = (*smem[0]).sqrt();
    sync_threads();

    // Pass 2: write normalized output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = input[(row_start + i) as usize] / l2;
        i += bdim;
        iter_idx += 1;
    }
}

// KB#40: layer_norm(x)_i = (x_i - mean) / sqrt(var + eps), per row
// 3-pass: mean → variance → normalize
#[gpu::cuda_kernel(dynamic_shared)]
pub fn layer_norm_forward(input: &[f32], output: &mut [f32], dim: u32, eps: f32) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gdim = grid_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid * dim;

    // Pass 1: compute mean
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < dim {
        local_sum += input[(row_start + i) as usize];
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_sum;
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
    let mean = *smem[0] / dim as f32;
    sync_threads();

    // Pass 2: compute variance = Σ(x - mean)² / dim
    let mut local_var = 0.0f32;
    i = tid;
    while i < dim {
        let diff = input[(row_start + i) as usize] - mean;
        local_var += diff * diff;
        i += bdim;
    }
    let mut sc = smem.chunk_mut(MapLinear::new(1));
    sc[0] = local_var;
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
    let inv_std = 1.0 / (*smem[0] / dim as f32 + eps).sqrt();
    sync_threads();

    // Pass 3: write normalized output
    let iters = (dim + bdim - 1) / bdim;
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );
    i = tid;
    let mut iter_idx = 0u32;
    while i < dim {
        out[iter_idx] = (input[(row_start + i) as usize] - mean) * inv_std;
        i += bdim;
        iter_idx += 1;
    }
}
