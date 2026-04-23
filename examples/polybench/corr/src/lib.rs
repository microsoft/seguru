use gpu::prelude::*;
use gpu::CacheStreamLoadStore;

const FLOAT_N: f32 = 3214212.01;
const EPS: f32 = 0.005;

// mean[j] = sum_i(data[i*m+j]) / FLOAT_N
#[gpu::cuda_kernel]
pub fn corr_mean_kernel(data: &[f32], mean: &mut [f32], m: u32, n: u32) {
    let mut mean = chunk_mut(mean, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j) as usize].ldcs();
            i += 1;
        }
        mean[0] = sum / FLOAT_N;
    }
}

// stddev[j] = sqrt(sum_i((data[i*m+j]-mean[j])^2) / FLOAT_N); clamp to 1.0 if <= EPS
#[gpu::cuda_kernel]
pub fn corr_std_kernel(
    data: &[f32],
    mean: &[f32],
    stddev: &mut [f32],
    m: u32,
    n: u32,
) {
    let mut stddev = chunk_mut(stddev, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < m {
        let mut sum = 0.0f32;
        let mean_j = mean[j as usize];
        let mut i: u32 = 0;
        while i < n {
            let diff = data[(i * m + j) as usize].ldcs() - mean_j;
            sum += diff * diff;
            i += 1;
        }
        sum /= FLOAT_N;
        let s = sum.sqrt();
        stddev[0] = if s <= EPS { 1.0 } else { s };
    }
}

// data[i*m+j] = (data[i*m+j] - mean[j]) / (sqrt(FLOAT_N) * stddev[j])
#[gpu::cuda_kernel]
pub fn corr_reduce_kernel(
    mean: &[f32],
    stddev: &[f32],
    data: &mut [f32],
    m: u32,
    n: u32,
) {
    let mut data = chunk_mut(data, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < n && j < m {
        let val = data[0] - mean[j as usize];
        data[0] = val / (FLOAT_N.sqrt() * stddev[j as usize]);
    }
}

// symmat[j1*m+j2] = sum_i(data[i*m+j1] * data[i*m+j2])
#[gpu::cuda_kernel]
pub fn corr_corr_kernel(data: &[f32], symmat: &mut [f32], m: u32, n: u32) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m as usize));
    let j2 = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let j1 = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < n {
            sum += data[(i * m + j1) as usize].ldcs() * data[(i * m + j2) as usize].ldcs();
            i += 1;
        }
        symmat[(0, 0)] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_corr(m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_data: Vec<f32> = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                h_data[i * m + j] = (i * j) as f32 / m as f32;
            }
        }
        let h_data_orig = h_data.clone();
        let mut h_data_gpu = h_data.clone();
        let mut h_mean_gpu: Vec<f32> = vec![0.0; m];
        let mut h_stddev_gpu: Vec<f32> = vec![0.0; m];
        let mut h_symmat_gpu: Vec<f32> = vec![0.0; m * m];

        // CPU reference
        let mut h_mean_cpu: Vec<f32> = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += h_data_orig[i * m + j];
            }
            h_mean_cpu[j] = sum / FLOAT_N;
        }
        let mut h_stddev_cpu: Vec<f32> = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0f32;
            for i in 0..n {
                let diff = h_data_orig[i * m + j] - h_mean_cpu[j];
                sum += diff * diff;
            }
            sum /= FLOAT_N;
            let s = sum.sqrt();
            h_stddev_cpu[j] = if s <= EPS { 1.0 } else { s };
        }
        let mut h_data_cpu = h_data_orig.clone();
        let sqrt_fn = FLOAT_N.sqrt();
        for i in 0..n {
            for j in 0..m {
                h_data_cpu[i * m + j] =
                    (h_data_cpu[i * m + j] - h_mean_cpu[j]) / (sqrt_fn * h_stddev_cpu[j]);
            }
        }
        let mut h_symmat_cpu: Vec<f32> = vec![0.0; m * m];
        for j1 in 0..m {
            for j2 in 0..m {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += h_data_cpu[i * m + j1] * h_data_cpu[i * m + j2];
                }
                h_symmat_cpu[j1 * m + j2] = sum;
            }
        }

        cuda_ctx(0, |ctx, m_module| {
            let d_data_ro =
                ctx.new_tensor_view(h_data.as_slice()).expect("alloc data_ro");
            let mut d_data = ctx
                .new_tensor_view(h_data_gpu.as_mut_slice())
                .expect("alloc data");
            let mut d_mean = ctx
                .new_tensor_view(h_mean_gpu.as_mut_slice())
                .expect("alloc mean");
            let mut d_stddev = ctx
                .new_tensor_view(h_stddev_gpu.as_mut_slice())
                .expect("alloc stddev");
            let mut d_symmat = ctx
                .new_tensor_view(h_symmat_gpu.as_mut_slice())
                .expect("alloc symmat");

            let block_size: u32 = 16;

            // kernel1: mean
            let grid_x = (m as u32 + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
            corr_mean_kernel::launch(config, ctx, m_module, &d_data_ro, &mut d_mean, m as u32, n as u32)
                .expect("mean kernel failed");

            // kernel2: stddev
            let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
            corr_std_kernel::launch(
                config, ctx, m_module, &d_data_ro, &d_mean, &mut d_stddev, m as u32, n as u32,
            )
            .expect("std kernel failed");

            // kernel3: reduce
            let grid_y = (n as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            corr_reduce_kernel::launch(
                config, ctx, m_module, &d_mean, &d_stddev, &mut d_data, m as u32, n as u32,
            )
            .expect("reduce kernel failed");

            // kernel4: correlation
            let grid_y = (m as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            corr_corr_kernel::launch(config, ctx, m_module, &d_data, &mut d_symmat, m as u32, n as u32)
                .expect("corr kernel failed");

            d_symmat
                .copy_to_host(&mut h_symmat_gpu)
                .expect("copy failed");
        });

        (h_symmat_gpu, h_symmat_cpu)
    }

    #[test]
    fn test_corr() {
        let m = 32;
        let n = 64;
        let (gpu, cpu) = run_corr(m, n);

        // Verify symmetry
        for j1 in 0..m {
            for j2 in 0..m {
                assert!(
                    (gpu[j1 * m + j2] - gpu[j2 * m + j1]).abs() < 1e-1,
                    "Not symmetric at ({},{}): {} vs {}",
                    j1, j2, gpu[j1 * m + j2], gpu[j2 * m + j1],
                );
            }
        }

        // Verify GPU matches CPU reference
        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "symmat is all zeros");
    }
}
