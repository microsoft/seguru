use gpu::prelude::*;
use gpu::CacheStreamLoadStore;

// kernel1: r[k*nj+k] = sqrt(sum_i(a[i*nj+k]^2))
#[gpu::cuda_kernel]
pub fn gramschm_kernel1(a: &[f32], r_kk: &mut [f32], ni: u32, nj: u32, k: u32) {
    let mut r_kk = chunk_mut(r_kk, MapContinuousLinear::new(1));
    let tid = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if tid == 0 {
        let mut nrm = 0.0f32;
        let mut i: u32 = 0;
        while i < ni {
            let value = a[(i * nj + k) as usize].ldcs();
            nrm += value * value;
            i += 1;
        }
        r_kk[0] = nrm.sqrt();
    }
}

// kernel2: q[i*nj+k] = a[i*nj+k] / r[k*nj+k]
// Launch full ni*nj grid, only column k threads write
#[gpu::cuda_kernel]
pub fn gramschm_kernel2(
    a: &[f32],
    r: &[f32],
    q: &mut [f32],
    nj: u32,
    ni: u32,
    k: u32,
) {
    let mut q = chunk_mut(q, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < ni && j == k {
        q[(0, 0)] = a[(i * nj + k) as usize] / r[(k * nj + k) as usize];
    }
}

// kernel3a: r[k*nj+j] = dot(q[:,k], a[:,j]) for j > k
// Launch nj*nj grid, only row k threads write
#[gpu::cuda_kernel]
pub fn gramschm_kernel3a(
    q: &[f32],
    a: &[f32],
    r: &mut [f32],
    ni: u32,
    nj: u32,
    k: u32,
) {
    let mut r = chunk_mut(r, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let row = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if row == k && j > k && j < nj {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < ni {
            sum += q[(i * nj + k) as usize].ldcs() * a[(i * nj + j) as usize].ldcs();
            i += 1;
        }
        r[(0, 0)] = sum;
    }
}

// kernel3b: a[i*nj+j] -= q[i*nj+k] * r[k*nj+j] for j > k
#[gpu::cuda_kernel]
pub fn gramschm_kernel3b(
    q: &[f32],
    r: &[f32],
    a: &mut [f32],
    ni: u32,
    nj: u32,
    k: u32,
) {
    let mut a = chunk_mut(a, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j > k && j < nj && i < ni {
        a[0] = a[0] - q[(i * nj + k) as usize] * r[(k * nj + j) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_gramschm(ni: usize, nj: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h_a: Vec<f32> = vec![0.0; ni * nj];
        for i in 0..ni {
            for j in 0..nj {
                h_a[i * nj + j] = ((i * j + 1) as f32) / (ni * nj) as f32;
            }
        }

        // CPU reference
        let mut h_a_cpu = h_a.clone();
        let mut h_r_cpu: Vec<f32> = vec![0.0; nj * nj];
        let mut h_q_cpu: Vec<f32> = vec![0.0; ni * nj];
        for k in 0..nj {
            let mut nrm = 0.0f32;
            for i in 0..ni {
                nrm += h_a_cpu[i * nj + k] * h_a_cpu[i * nj + k];
            }
            h_r_cpu[k * nj + k] = nrm.sqrt();
            for i in 0..ni {
                h_q_cpu[i * nj + k] = h_a_cpu[i * nj + k] / h_r_cpu[k * nj + k];
            }
            for j in (k + 1)..nj {
                let mut dot = 0.0f32;
                for i in 0..ni {
                    dot += h_q_cpu[i * nj + k] * h_a_cpu[i * nj + j];
                }
                h_r_cpu[k * nj + j] = dot;
                for i in 0..ni {
                    h_a_cpu[i * nj + j] -= h_q_cpu[i * nj + k] * h_r_cpu[k * nj + j];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_r_gpu: Vec<f32> = vec![0.0; nj * nj];
        let mut h_q_gpu: Vec<f32> = vec![0.0; ni * nj];

        cuda_ctx(0, |ctx, m_module| {
            let mut d_a = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a");
            let mut d_r = ctx
                .new_tensor_view(h_r_gpu.as_mut_slice())
                .expect("alloc r");
            let mut d_q = ctx
                .new_tensor_view(h_q_gpu.as_mut_slice())
                .expect("alloc q");

            let block_size: u32 = 16;

            for k in 0..nj {
                // kernel1: r[k][k] = sqrt(sum_i(a[i*nj+k]^2)) on the GPU
                {
                    let diag = k * nj + k;
                    let (_, mut diag_and_after) = d_r.split_at_mut(diag);
                    let (mut r_kk, _) = diag_and_after.split_at_mut(1);
                    let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
                    gramschm_kernel1::launch(
                        config,
                        ctx,
                        m_module,
                        &d_a,
                        &mut r_kk,
                        ni as u32,
                        nj as u32,
                        k as u32,
                    )
                    .expect("kernel1 failed");
                }

                // kernel2: q[:,k] = a[:,k] / r_kk — full ni*nj grid, only col k writes
                let grid_x = (nj as u32 + block_size - 1) / block_size;
                let grid_y = (ni as u32 + block_size - 1) / block_size;
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                gramschm_kernel2::launch(
                    config, ctx, m_module, &d_a, &d_r, &mut d_q, nj as u32, ni as u32, k as u32,
                )
                .expect("kernel2 failed");

                // kernel3a: r[k][j] = dot(q[:,k], a[:,j]) — full nj*nj grid, only row k writes
                let grid_x = (nj as u32 + block_size - 1) / block_size;
                let grid_y = (nj as u32 + block_size - 1) / block_size;
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                gramschm_kernel3a::launch(
                    config, ctx, m_module, &d_q, &d_a, &mut d_r, ni as u32, nj as u32, k as u32,
                )
                .expect("kernel3a failed");

                // kernel3b: a[i][j] -= q[i][k] * r[k][j] — full nj*ni grid
                let grid_x = (nj as u32 + block_size - 1) / block_size;
                let grid_y = (ni as u32 + block_size - 1) / block_size;
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                gramschm_kernel3b::launch(
                    config, ctx, m_module, &d_q, &d_r, &mut d_a, ni as u32, nj as u32, k as u32,
                )
                .expect("kernel3b failed");
            }

            d_a.copy_to_host(&mut h_a_gpu).expect("copy a");
            d_r.copy_to_host(&mut h_r_gpu).expect("copy r");
            d_q.copy_to_host(&mut h_q_gpu).expect("copy q");
        });

        (h_a_gpu, h_a_cpu, h_r_gpu, h_r_cpu)
    }

    #[test]
    fn test_gramschm() {
        let ni = 32;
        let nj = 32;
        let (_, _, r_gpu, r_cpu) = run_gramschm(ni, nj);

        // Compare R matrices
        for i in 0..r_gpu.len() {
            assert!(
                (r_gpu[i] - r_cpu[i]).abs() < 1.0,
                "R mismatch at {}: gpu={} cpu={}",
                i, r_gpu[i], r_cpu[i],
            );
        }

        // Verify R diagonal is positive
        for k in 0..nj {
            assert!(
                r_gpu[k * nj + k] > 0.0,
                "R diagonal at {} should be positive: {}",
                k, r_gpu[k * nj + k],
            );
        }
    }
}
