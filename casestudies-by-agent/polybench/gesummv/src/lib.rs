use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn gesummv_kernel(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    y: &mut [f32],
    n: u32,
    alpha: f32,
    beta: f32,
) {
    let mut y = chunk_mut(y, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();

    if i < n {
        let mut tmp_val: f32 = 0.0;
        let mut y_val: f32 = 0.0;
        let a_row: &[f32] = &a[(i * n) as usize..((i + 1) * n) as usize];
        let b_row: &[f32] = &b[(i * n) as usize..((i + 1) * n) as usize];
        let mut j_idx: usize = 0;
        while j_idx < n as usize {
            tmp_val += a_row[j_idx] * x[j_idx];
            y_val += b_row[j_idx] * x[j_idx];
            j_idx += 1;
        }
        y[0] = alpha * tmp_val + beta * y_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_gesummv(n: usize, alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = vec![1.0; n * n];
        let h_b: Vec<f32> = vec![1.0; n * n];
        let h_x: Vec<f32> = vec![1.0; n];
        let mut h_y_gpu: Vec<f32> = vec![0.0; n];
        let mut h_y_cpu: Vec<f32> = vec![0.0; n];

        // CPU reference
        for i in 0..n {
            let mut tmp_val: f32 = 0.0;
            let mut y_val: f32 = 0.0;
            for j in 0..n {
                tmp_val += h_a[i * n + j] * h_x[j];
                y_val += h_b[i * n + j] * h_x[j];
            }
            h_y_cpu[i] = alpha * tmp_val + beta * y_val;
        }

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b");
            let d_x = ctx.new_tensor_view(h_x.as_slice()).expect("alloc x");
            let mut d_y = ctx
                .new_tensor_view(h_y_gpu.as_mut_slice())
                .expect("alloc y");

            let block_size: u32 = 256;
            let num_blocks: u32 = (n as u32 + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, 0);
            gesummv_kernel::launch(
                config, ctx, m, &d_a, &d_b, &d_x, &mut d_y, n as u32, alpha, beta,
            )
            .expect("kernel launch failed");

            d_y.copy_to_host(&mut h_y_gpu).expect("copy failed");
        });

        (h_y_gpu, h_y_cpu)
    }

    #[test]
    fn test_gesummv() {
        let n = 64;
        let alpha = 1.0;
        let beta = 1.0;
        let (gpu, cpu) = run_gesummv(n, alpha, beta);
        // Expected: alpha*n + beta*n = 2*n = 128.0
        for i in 0..n {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i],
                cpu[i]
            );
        }
        assert!((gpu[0] - 128.0).abs() < 1e-4);
    }
}
