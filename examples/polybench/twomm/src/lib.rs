use gpu::prelude::*;

// kernel1: tmp = alpha * A * B
#[gpu::cuda_kernel]
pub fn mm2_kernel1(
    a: &[f32],
    b: &[f32],
    tmp: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
    alpha: f32,
) {
    let mut tmp = chunk_mut(tmp, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j < nj {
        let mut val = 0.0f32;
        let mut k: u32 = 0;
        while k < nk {
            val += alpha * a[(i * nk + k) as usize] * b[(k * nj + j) as usize];
            k += 1;
        }
        tmp[(0, 0)] = val;
    }
}

// kernel2: D = D*beta + tmp*C
#[gpu::cuda_kernel]
pub fn mm2_kernel2(
    tmp: &[f32],
    c: &[f32],
    d: &mut [f32],
    ni: u32,
    nj: u32,
    nl: u32,
    beta: f32,
) {
    let mut d = chunk_mut(d, Map2D::new(nl as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j < nl {
        let mut val = d[(0, 0)] * beta;
        let mut k: u32 = 0;
        while k < nj {
            val += tmp[(i * nj + k) as usize] * c[(k * nl + j) as usize];
            k += 1;
        }
        d[(0, 0)] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_twomm(
        ni: usize,
        nj: usize,
        nk: usize,
        nl: usize,
        alpha: f32,
        beta: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = vec![1.0; ni * nk];
        let h_b: Vec<f32> = vec![1.0; nk * nj];
        let h_c: Vec<f32> = vec![1.0; nj * nl];
        let mut h_tmp_gpu: Vec<f32> = vec![0.0; ni * nj];
        let mut h_d_gpu: Vec<f32> = vec![0.0; ni * nl];
        let mut h_d_cpu: Vec<f32> = vec![0.0; ni * nl];

        // CPU reference: tmp = alpha * A * B
        let mut h_tmp_cpu: Vec<f32> = vec![0.0; ni * nj];
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    h_tmp_cpu[i * nj + j] += alpha * h_a[i * nk + k] * h_b[k * nj + j];
                }
            }
        }
        // CPU reference: D = D*beta + tmp*C
        for i in 0..ni {
            for j in 0..nl {
                h_d_cpu[i * nl + j] *= beta;
                for k in 0..nj {
                    h_d_cpu[i * nl + j] += h_tmp_cpu[i * nj + k] * h_c[k * nl + j];
                }
            }
        }

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b");
            let d_c = ctx.new_tensor_view(h_c.as_slice()).expect("alloc c");
            let mut d_tmp = ctx
                .new_tensor_view(h_tmp_gpu.as_mut_slice())
                .expect("alloc tmp");
            let mut d_d = ctx
                .new_tensor_view(h_d_gpu.as_mut_slice())
                .expect("alloc d");

            let block_size: u32 = 16;

            // Launch kernel1: tmp = alpha * A * B
            let grid_x: u32 = (nj as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            mm2_kernel1::launch(
                config, ctx, m, &d_a, &d_b, &mut d_tmp, ni as u32, nj as u32, nk as u32, alpha,
            )
            .expect("kernel1 launch failed");

            // Launch kernel2: D = D*beta + tmp*C
            let grid_x: u32 = (nl as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            mm2_kernel2::launch(
                config, ctx, m, &d_tmp, &d_c, &mut d_d, ni as u32, nj as u32, nl as u32, beta,
            )
            .expect("kernel2 launch failed");

            d_d.copy_to_host(&mut h_d_gpu).expect("copy failed");
        });

        (h_d_gpu, h_d_cpu)
    }

    #[test]
    fn test_twomm_ones() {
        let n = 32;
        let (gpu, cpu) = run_twomm(n, n, n, n, 1.0, 0.0);
        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-2,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i],
                cpu[i],
            );
        }
        // tmp[i][j] = nk = 32, D[i][j] = sum_k(32*1) = nj*nk = 1024
        assert!((gpu[0] - 1024.0).abs() < 1e-2);
    }
}
