// KernelBench Level 1 — mat-vec, scalar multiply, 3D tensor matmul kernels
use gpu::prelude::*;

// KB#4: y = A(M×N) * x(N) → y(M), one thread per row
#[gpu::cuda_kernel]
pub fn matvec_forward(a: &[f32], x: &[f32], y: &mut [f32], m: u32, n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(y, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < m {
        let n_us = n as usize;
        let a_row = &a[tid as usize * n_us..(tid as usize + 1) * n_us];
        let mut sum = 0.0f32;
        let mut idx = 0u32;
        while idx < n {
            sum += a_row[idx as usize] * x[idx as usize];
            idx += 1;
        }
        out[0] = sum;
    }
}

// KB#5: C = A * s, element-wise scalar multiply
#[gpu::cuda_kernel]
pub fn scalar_multiply(input: &[f32], output: &mut [f32], s: f32, n: u32) {
    let tid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let mut out = chunk_mut(output, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    if tid < n {
        out[0] = input[tid as usize] * s;
    }
}

// KB#10: Batched 3D tensor matmul — A(batch×M×K) * B(batch×K×N) → C(batch×M×N)
// 1D grid with batch/row/col decomposition from linear tid.
#[gpu::cuda_kernel]
pub fn tensor3d_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
) {
    let mut c = chunk_mut(c, reshape_map!([1] | [block_dim::<DimX>(), grid_dim::<DimX>()] => layout: [i0, t0, t1]));
    let tid = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let mn = m * n;
    let total = batch * mn;
    if tid < total {
        let b_idx = tid / mn;
        let rem = tid % mn;
        let row = rem / n;
        let col = rem % n;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let n_us = n as usize;
        let b_idx_us = b_idx as usize;
        let a_off = b_idx_us * (m as usize) * k_us;
        let b_off = b_idx_us * k_us * n_us;
        let a_row = &a[a_off + row_us * k_us..a_off + (row_us + 1) * k_us];
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < k_us {
            sum += a_row[i] * b[b_off + i * n_us + col_us];
            i += 1;
        }
        c[0] = sum;
    }
}
