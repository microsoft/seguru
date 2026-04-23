// KernelBench Level 1 — matmul kernels
use gpu::prelude::*;

// KB#1,2,6,7,8,9: C(M×N) = A(M×K) * B(K×N)
#[gpu::cuda_kernel]
pub fn matmul_forward(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let mut sum = 0.0f32;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let n_us = n as usize;
        let mut idx = 0usize;
        while idx < k_us {
            sum += a[row_us * k_us + idx] * b[idx * n_us + col_us];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#16: C(M×N) = Aᵀ * B, where A is stored as K×M
#[gpu::cuda_kernel]
pub fn matmul_transposed_a(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let mut sum = 0.0f32;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let m_us = m as usize;
        let n_us = n as usize;
        let mut idx = 0usize;
        while idx < k_us {
            sum += a[idx * m_us + row_us] * b[idx * n_us + col_us];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#17: C(M×N) = A * Bᵀ, where B is stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_b(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let mut sum = 0.0f32;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let mut idx = 0usize;
        while idx < k_us {
            sum += a[row_us * k_us + idx] * b[col_us * k_us + idx];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#18: C(M×N) = Aᵀ * Bᵀ, A stored as K×M, B stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_both(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let col = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if row < m && col < n {
        let mut sum = 0.0f32;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let m_us = m as usize;
        let mut idx = 0usize;
        while idx < k_us {
            sum += a[idx * m_us + row_us] * b[col_us * k_us + idx];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#3,10: Batched matmul — A(batch×M×K) * B(batch×K×N) → C(batch×M×N)
// Uses 1D grid with MapContinuousLinear since Map2D requires DimZ==1.
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
    let mut c = chunk_mut(c, MapContinuousLinear::new(1));
    let tid = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let mn = m * n;
    let total = batch * mn;
    if tid < total {
        let b_idx = tid / mn;
        let rem = tid % mn;
        let row = rem / n;
        let col = rem % n;
        let mut sum = 0.0f32;
        let row_us = row as usize;
        let col_us = col as usize;
        let k_us = k as usize;
        let m_us = m as usize;
        let n_us = n as usize;
        let b_idx_us = b_idx as usize;
        let a_off = b_idx_us * m_us * k_us;
        let b_off = b_idx_us * k_us * n_us;
        let mut i = 0usize;
        while i < k_us {
            sum += a[a_off + row_us * k_us + i] * b[b_off + i * n_us + col_us];
            i += 1;
        }
        c[0] = sum;
    }
}
