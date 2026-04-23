// KernelBench Level 1 — cumulative kernels
use gpu::prelude::*;

// KB#89: Inclusive cumulative sum along dim=1
// One thread per row processes sequentially
#[gpu::cuda_kernel]
pub fn cumsum_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    let iters = dim;
    let gdim = grid_dim::<DimX>();
    let bdim = block_dim::<DimX>();
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );

    let mut acc = 0.0f32;
    let mut i = 0u32;
    while i < dim {
        acc += row[i as usize];
        out[i] = acc;
        i += 1;
    }
}

// KB#90: Inclusive cumulative product along dim=1
#[gpu::cuda_kernel]
pub fn cumprod_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    let iters = dim;
    let gdim = grid_dim::<DimX>();
    let bdim = block_dim::<DimX>();
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );

    let mut acc = 1.0f32;
    let mut i = 0u32;
    while i < dim {
        acc *= row[i as usize];
        out[i] = acc;
        i += 1;
    }
}

// KB#91: Reverse inclusive cumulative sum (right-to-left)
#[gpu::cuda_kernel]
pub fn cumsum_reverse_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    let iters = dim;
    let gdim = grid_dim::<DimX>();
    let bdim = block_dim::<DimX>();
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );

    let mut acc = 0.0f32;
    let mut i = dim;
    while i > 0 {
        i -= 1;
        acc += row[i as usize];
        out[i] = acc;
    }
}

// KB#92: Exclusive cumulative sum (first element = 0, shifted right)
#[gpu::cuda_kernel]
pub fn cumsum_exclusive_forward(input: &[f32], output: &mut [f32], dim: u32) {
    let bid = block_id::<DimX>();
    let row_start = bid * dim;
    let row = &input[row_start as usize..(row_start + dim) as usize];

    let iters = dim;
    let gdim = grid_dim::<DimX>();
    let bdim = block_dim::<DimX>();
    let mut out = chunk_mut(
        output,
        reshape_map!([iters] | [bdim, gdim] => layout: [t0, i0, t1]),
    );

    let mut acc = 0.0f32;
    let mut i = 0u32;
    while i < dim {
        out[i] = acc;
        acc += row[i as usize];
        i += 1;
    }
}
