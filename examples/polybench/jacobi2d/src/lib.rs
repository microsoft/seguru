use gpu::prelude::*;

/// B[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
#[gpu::cuda_kernel]
pub fn jacobi2d_kernel1(a: &[f32], b: &mut [f32], n: u32) {
    let mut b = chunk_mut(b, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i >= 1 && i < n - 1 && j >= 1 && j < n - 1 {
        b[(0, 0)] = 0.2
            * (a[(i * n + j) as usize]
                + a[(i * n + (j - 1)) as usize]
                + a[(i * n + (j + 1)) as usize]
                + a[((i + 1) * n + j) as usize]
                + a[((i - 1) * n + j) as usize]);
    }
}

/// A[i][j] = B[i][j]
#[gpu::cuda_kernel]
pub fn jacobi2d_kernel2(a: &mut [f32], b: &[f32], n: u32) {
    let mut a = chunk_mut(a, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i >= 1 && i < n - 1 && j >= 1 && j < n - 1 {
        a[(0, 0)] = b[(i * n + j) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn jacobi2d_cpu(a: &mut [f32], b: &mut [f32], n: usize, tsteps: usize) {
        for _ in 0..tsteps {
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    b[i * n + j] = 0.2
                        * (a[i * n + j]
                            + a[i * n + (j - 1)]
                            + a[i * n + (j + 1)]
                            + a[(i + 1) * n + j]
                            + a[(i - 1) * n + j]);
                }
            }
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    a[i * n + j] = b[i * n + j];
                }
            }
        }
    }

    fn run_jacobi2d(h_a: &mut [f32], h_b: &mut [f32], n: usize, tsteps: usize) {
        cuda_ctx(0, |ctx, m| {
            let mut d_a = ctx.new_tensor_view(h_a.as_mut()).expect("alloc a");
            let mut d_b = ctx.new_tensor_view(h_b.as_mut()).expect("alloc b");

            let block_size: u32 = 16;
            let grid_x: u32 = (n as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (n as u32 + block_size - 1) / block_size;

            for _ in 0..tsteps {
                let c1 = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                jacobi2d_kernel1::launch(c1, ctx, m, &d_a, &mut d_b, n as u32).expect("k1");
                let c2 = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                jacobi2d_kernel2::launch(c2, ctx, m, &mut d_a, &d_b, n as u32).expect("k2");
            }

            d_a.copy_to_host(h_a).expect("copy a");
            d_b.copy_to_host(h_b).expect("copy b");
        });
    }

    #[test]
    fn test_jacobi2d() {
        let n = 33;
        let tsteps = 5;

        let mut h_a_gpu: Vec<f32> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                (i * (j + 2) + 2) as f32 / (n as f32)
            })
            .collect();
        let mut h_b_gpu: Vec<f32> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                (i * (j + 3) + 3) as f32 / (n as f32)
            })
            .collect();
        let mut h_a_cpu = h_a_gpu.clone();
        let mut h_b_cpu = h_b_gpu.clone();

        run_jacobi2d(&mut h_a_gpu, &mut h_b_gpu, n, tsteps);
        jacobi2d_cpu(&mut h_a_cpu, &mut h_b_cpu, n, tsteps);

        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                assert!(
                    (h_a_gpu[idx] - h_a_cpu[idx]).abs() < 1e-2,
                    "A mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    h_a_gpu[idx],
                    h_a_cpu[idx],
                );
                assert!(
                    (h_b_gpu[idx] - h_b_cpu[idx]).abs() < 1e-2,
                    "B mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    h_b_gpu[idx],
                    h_b_cpu[idx],
                );
            }
        }
    }
}
