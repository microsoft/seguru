use gpu::prelude::*;
use gpu::CacheStreamLoadStore;

#[gpu::cuda_kernel]
pub fn gemm_kernel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
    alpha: f32,
    beta: f32,
) {
    let mut c = chunk_mut(c, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j < nj {
        let mut val = c[(0, 0)] * beta;
        let a_row: &[f32] = &a[(i * nk) as usize..((i + 1) * nk) as usize];
        let mut b_idx = j as usize;
        for a_val in a_row {
            val += alpha * a_val * b[b_idx].ldcs();
            b_idx += nj as usize;
        }
        c[(0, 0)] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_gemm(ni: usize, nj: usize, nk: usize, alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = vec![1.0; ni * nk];
        let h_b: Vec<f32> = vec![1.0; nk * nj];
        let mut h_c_gpu: Vec<f32> = vec![0.0; ni * nj];
        let mut h_c_cpu: Vec<f32> = vec![0.0; ni * nj];

        // CPU reference
        for i in 0..ni {
            for j in 0..nj {
                h_c_cpu[i * nj + j] *= beta;
                for k in 0..nk {
                    h_c_cpu[i * nj + j] += alpha * h_a[i * nk + k] * h_b[k * nj + j];
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
            let grid_x: u32 = (nj as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            gemm_kernel::launch(
                config, ctx, m, &d_a, &d_b, &mut d_c, ni as u32, nj as u32, nk as u32, alpha, beta,
            )
            .expect("kernel launch failed");

            d_c.copy_to_host(&mut h_c_gpu).expect("copy failed");
        });

        (h_c_gpu, h_c_cpu)
    }

    #[test]
    fn test_gemm_ones() {
        let n = 32;
        let (gpu, cpu) = run_gemm(n, n, n, 1.0, 0.0);
        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
        // Each element should be nk = 32
        assert!((gpu[0] - 32.0).abs() < 1e-4);
    }
}
