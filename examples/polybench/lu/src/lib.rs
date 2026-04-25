use gpu::prelude::*;

// A[k][j] /= A[k][k] for j > k (only row k is modified)
#[gpu::cuda_kernel]
pub fn lu_kernel1(pivot: &[f32], row_tail: &mut [f32], rem: u32) {
    let mut row_tail = chunk_mut(row_tail, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < rem {
        row_tail[0] = row_tail[0] / pivot[0];
    }
}

#[gpu::cuda_kernel]
pub fn lu_copy_col(a: &[f32], col: &mut [f32], n: u32, k: u32) {
    let mut col = chunk_mut(col, MapContinuousLinear::new(1));
    let i_tail = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let rem = n - k - 1;
    if i_tail < rem {
        col[0] = a[((i_tail + k + 1) * n + k) as usize];
    }
}

// A[i][j] -= A[i][k] * A[k][j] for i,j > k
#[gpu::cuda_kernel]
pub fn lu_kernel2(row_tail: &[f32], col: &[f32], rows_below: &mut [f32], n: u32, k: u32) {
    let j_tail = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i_tail = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let rem = n - k - 1;
    let mut rows_below = chunk_mut(
        rows_below,
        reshape_map!([1] | [(rem, n), rem] => layout: [i0, t0, t1], offset: k + 1),
    );
    if i_tail < rem && j_tail < rem {
        rows_below[0] = rows_below[0] - col[i_tail as usize] * row_tail[j_tail as usize];
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
        let mut h_col = vec![0.0f32; n];

        cuda_ctx(0, |ctx, m_module| {
            let mut d_a = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a");
            let mut d_col = ctx
                .new_tensor_view(h_col.as_mut_slice())
                .expect("alloc col");

            let block_size: u32 = 16;
            let row_block_size: u32 = 256;

            for k in 0..n {
                let rem = n - k - 1;
                if rem > 0 {
                    {
                        let row_tail_start = k * n + k + 1;
                        let row_tail_end = (k + 1) * n;
                        let (prefix, mut tail_and_after) = d_a.split_at_mut(row_tail_start);
                        let pivot = prefix.index(row_tail_start - 1..row_tail_start);
                        let (mut row_tail, _) =
                            tail_and_after.split_at_mut(row_tail_end - row_tail_start);
                        let grid = (rem as u32 + row_block_size - 1) / row_block_size;
                        let config = gpu_host::gpu_config!(grid, 1, 1, row_block_size, 1, 1, 0);
                        lu_kernel1::launch(
                            config,
                            ctx,
                            m_module,
                            &pivot,
                            &mut row_tail,
                            rem as u32,
                        )
                        .expect("kernel1 failed");
                    }
                    ctx.sync().expect("kernel1 sync failed");
                    {
                        let grid = (rem as u32 + row_block_size - 1) / row_block_size;
                        let config = gpu_host::gpu_config!(grid, 1, 1, row_block_size, 1, 1, 0);
                        lu_copy_col::launch(
                            config, ctx, m_module, &d_a, &mut d_col, n as u32, k as u32,
                        )
                        .expect("copy col failed");
                    }
                    {
                        let row_tail_start = k * n + k + 1;
                        let row_tail_end = (k + 1) * n;
                        let split_at = (k + 1) * n;
                        let (prefix, mut rows_below) = d_a.split_at_mut(split_at);
                        let row_tail = prefix.index(row_tail_start..row_tail_end);
                        let col = d_col.index(..rem);
                        let grid = (rem as u32 + block_size - 1) / block_size;
                        let config =
                            gpu_host::gpu_config!(grid, grid, 1, block_size, block_size, 1, 0);
                        lu_kernel2::launch(
                            config,
                            ctx,
                            m_module,
                            &row_tail,
                            &col,
                            &mut rows_below,
                            n as u32,
                            k as u32,
                        )
                        .expect("kernel2 failed");
                    }
                    ctx.sync().expect("kernel2 sync failed");
                }
            }

            d_a.copy_to_host(&mut h_a_gpu).expect("copy failed");
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
