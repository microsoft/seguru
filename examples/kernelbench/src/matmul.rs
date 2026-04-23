// KernelBench Level 1 — matmul kernels
use gpu::prelude::*;

// KB#1,2,6,7,8,9: C(M×N) = A(M×K) * B(K×N)
#[gpu::cuda_kernel]
pub fn matmul_forward(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, reshape_map!([1] | [(block_dim::<DimX>() * grid_dim::<DimX>(), n), (block_dim::<DimY>() * grid_dim::<DimY>(), m)] => layout: [i0, t0, t1]));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let n_us = n as usize;
        let a_row = &a[row_us * k_us..(row_us + 1) * k_us];
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        let k_us_4 = k_us & !3;
        while idx < k_us_4 {
            sum += a_row[idx] * b[idx * n_us + col_us]
                + a_row[idx + 1] * b[(idx + 1) * n_us + col_us]
                + a_row[idx + 2] * b[(idx + 2) * n_us + col_us]
                + a_row[idx + 3] * b[(idx + 3) * n_us + col_us];
            idx += 4;
        }
        while idx < k_us {
            sum += a_row[idx] * b[idx * n_us + col_us];
            idx += 1;
        }
        c[0] = sum;
    }
}

// KB#16: C(M×N) = Aᵀ * B, where A is stored as K×M
#[gpu::cuda_kernel]
pub fn matmul_transposed_a(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, reshape_map!([1] | [(block_dim::<DimX>() * grid_dim::<DimX>(), n), (block_dim::<DimY>() * grid_dim::<DimY>(), m)] => layout: [i0, t0, t1]));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let m_us = m as usize;
        let n_us = n as usize;
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        let k_us_4 = k_us & !3;
        while idx < k_us_4 {
            sum += a[idx * m_us + row_us] * b[idx * n_us + col_us]
                + a[(idx + 1) * m_us + row_us] * b[(idx + 1) * n_us + col_us]
                + a[(idx + 2) * m_us + row_us] * b[(idx + 2) * n_us + col_us]
                + a[(idx + 3) * m_us + row_us] * b[(idx + 3) * n_us + col_us];
            idx += 4;
        }
        while idx < k_us {
            sum += a[idx * m_us + row_us] * b[idx * n_us + col_us];
            idx += 1;
        }
        c[0] = sum;
    }
}

// KB#17: C(M×N) = A * Bᵀ, where B is stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_b(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, reshape_map!([1] | [(block_dim::<DimX>() * grid_dim::<DimX>(), n), (block_dim::<DimY>() * grid_dim::<DimY>(), m)] => layout: [i0, t0, t1]));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let a_row = &a[row_us * k_us..(row_us + 1) * k_us];
        let b_row = &b[col_us * k_us..(col_us + 1) * k_us];
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        let k_us_4 = k_us & !3;
        while idx < k_us_4 {
            sum += a_row[idx] * b_row[idx]
                + a_row[idx + 1] * b_row[idx + 1]
                + a_row[idx + 2] * b_row[idx + 2]
                + a_row[idx + 3] * b_row[idx + 3];
            idx += 4;
        }
        while idx < k_us {
            sum += a_row[idx] * b_row[idx];
            idx += 1;
        }
        c[0] = sum;
    }
}

// KB#18: C(M×N) = Aᵀ * Bᵀ, A stored as K×M, B stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_both(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, reshape_map!([1] | [(block_dim::<DimX>() * grid_dim::<DimX>(), n), (block_dim::<DimY>() * grid_dim::<DimY>(), m)] => layout: [i0, t0, t1]));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let m_us = m as usize;
        let b_row = &b[col_us * k_us..(col_us + 1) * k_us];
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        let k_us_4 = k_us & !3;
        while idx < k_us_4 {
            sum += a[idx * m_us + row_us] * b_row[idx]
                + a[(idx + 1) * m_us + row_us] * b_row[idx + 1]
                + a[(idx + 2) * m_us + row_us] * b_row[idx + 2]
                + a[(idx + 3) * m_us + row_us] * b_row[idx + 3];
            idx += 4;
        }
        while idx < k_us {
            sum += a[idx * m_us + row_us] * b_row[idx];
            idx += 1;
        }
        c[0] = sum;
    }
}

// KB#3,10: Batched matmul — A(batch×M×K) * B(batch×K×N) → C(batch×M×N)
// Uses 1D grid since the batch dimension requires DimZ.
#[gpu::cuda_kernel]
pub fn matmul_batched(
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
        let k_us_4 = k_us & !3;
        while i < k_us_4 {
            sum += a_row[i] * b[b_off + i * n_us + col_us]
                + a_row[i + 1] * b[b_off + (i + 1) * n_us + col_us]
                + a_row[i + 2] * b[b_off + (i + 2) * n_us + col_us]
                + a_row[i + 3] * b[b_off + (i + 3) * n_us + col_us];
            i += 4;
        }
        while i < k_us {
            sum += a_row[i] * b[b_off + i * n_us + col_us];
            i += 1;
        }
        c[0] = sum;
    }
}
