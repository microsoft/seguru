use gpu::prelude::*;

// A[k][j] /= A[k][k] for j > k (only row k is modified)
#[gpu::cuda_kernel]
pub fn lu_kernel1(a_read: &[f32], a_write: &mut [f32], n: u32, k: u32) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i == k && j > k && j < n {
        a_write[(0, 0)] = a_read[(k * n + j) as usize] / a_read[(k * n + k) as usize];
    }
}

// A[i][j] -= A[i][k] * A[k][j] for i,j > k
#[gpu::cuda_kernel]
pub fn lu_kernel2(a_read: &[f32], a_write: &mut [f32], n: u32, k: u32) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i > k && j > k && i < n && j < n {
        a_write[(0, 0)] = a_read[(i * n + j) as usize] - a_read[(i * n + k) as usize] * a_read[(k * n + j) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_lu(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_a: Vec<f32> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                // Diagonally dominant to avoid zero pivots
                h_a[i * n + j] = if i == j {
                    (n as f32) + 1.0
                } else {
                    (i as f32 + j as f32) / n as f32
                };
            }
        }

        // CPU reference
        let mut h_a_cpu = h_a.clone();
        for k in 0..n {
            for j in (k + 1)..n {
                h_a_cpu[k * n + j] /= h_a_cpu[k * n + k];
            }
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    h_a_cpu[i * n + j] -= h_a_cpu[i * n + k] * h_a_cpu[k * n + j];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_a_read = h_a.clone();

        cuda_ctx(0, |ctx, m_module| {
            let mut d_a_write = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a_write");
            let mut d_a_read = ctx
                .new_tensor_view(h_a_read.as_mut_slice())
                .expect("alloc a_read");

            let block_size: u32 = 16;
            let grid_x = (n as u32 + block_size - 1) / block_size;
            let grid_y = (n as u32 + block_size - 1) / block_size;

            for k in 0..n {
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                lu_kernel1::launch(
                    config, ctx, m_module, &d_a_read, &mut d_a_write, n as u32, k as u32,
                )
                .expect("kernel1 failed");

                // Sync: copy d_a_write → d_a_read for kernel2 to read updated values
                d_a_write
                    .copy_to_host(&mut h_a_gpu)
                    .expect("copy to host");
                d_a_read
                    .copy_from_host(&h_a_gpu)
                    .expect("copy to device");

                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                lu_kernel2::launch(
                    config, ctx, m_module, &d_a_read, &mut d_a_write, n as u32, k as u32,
                )
                .expect("kernel2 failed");

                // Sync for next iteration
                d_a_write
                    .copy_to_host(&mut h_a_gpu)
                    .expect("copy to host");
                d_a_read
                    .copy_from_host(&h_a_gpu)
                    .expect("copy to device");
            }

            d_a_write
                .copy_to_host(&mut h_a_gpu)
                .expect("copy failed");
        });

        (h_a_gpu, h_a_cpu)
    }

    #[test]
    fn test_lu() {
        let n = 32;
        let (gpu, cpu) = run_lu(n);

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "result is all zeros");
    }
}
