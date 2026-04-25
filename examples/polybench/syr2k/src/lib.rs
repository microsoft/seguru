use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn syr2k_kernel(a: &[f32], b: &[f32], c: &mut [f32], ni: u32, nj: u32, alpha: f32, beta: f32) {
    let mut c = chunk_mut(c, Map2D::new(ni as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j <= i {
        let mut val = c[(0, 0)] * beta;
        let a_row_i: &[f32] = &a[(i * nj) as usize..((i + 1) * nj) as usize];
        let b_row_j: &[f32] = &b[(j * nj) as usize..((j + 1) * nj) as usize];
        let b_row_i: &[f32] = &b[(i * nj) as usize..((i + 1) * nj) as usize];
        let a_row_j: &[f32] = &a[(j * nj) as usize..((j + 1) * nj) as usize];
        for k in 0..nj as usize {
            val += alpha * a_row_i[k] * b_row_j[k] + alpha * b_row_i[k] * a_row_j[k];
        }
        c[(0, 0)] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_syr2k(ni: usize, nj: usize, alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        // a[i][k] = ((i*nj + k) % 7) as f32
        let h_a: Vec<f32> = (0..ni * nj).map(|x| (x % 7) as f32).collect();
        let h_b: Vec<f32> = (0..ni * nj).map(|x| (x % 5) as f32).collect();
        let mut h_c_gpu: Vec<f32> = vec![0.0; ni * ni];
        let mut h_c_cpu: Vec<f32> = vec![0.0; ni * ni];

        // CPU reference
        for i in 0..ni {
            for j in 0..=i {
                h_c_cpu[i * ni + j] *= beta;
                for k in 0..nj {
                    h_c_cpu[i * ni + j] += alpha * h_a[i * nj + k] * h_b[j * nj + k]
                        + alpha * h_b[i * nj + k] * h_a[j * nj + k];
                }
            }
        }

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b");
            let mut d_c = ctx
                .new_tensor_view(h_c_gpu.as_mut_slice())
                .expect("alloc c");

            let block_size: u32 = 16;
            let grid_x: u32 = (ni as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            syr2k_kernel::launch(
                config, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, alpha, beta,
            )
            .expect("kernel launch failed");

            d_c.copy_to_host(&mut h_c_gpu).expect("copy failed");
        });

        (h_c_gpu, h_c_cpu)
    }

    #[test]
    fn test_syr2k() {
        let ni = 32;
        let nj = 32;
        let (gpu, cpu) = run_syr2k(ni, nj, 1.0, 0.0);
        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-1,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
    }
}
