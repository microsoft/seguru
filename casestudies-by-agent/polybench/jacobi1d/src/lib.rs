use gpu::prelude::*;

/// B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
#[gpu::cuda_kernel]
pub fn jacobi1d_kernel1(a: &[f32], b: &mut [f32], n: u32) {
    let mut b = chunk_mut(b, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i > 0 && i < n - 1 {
        b[0] = 0.33333 * (a[(i - 1) as usize] + a[i as usize] + a[(i + 1) as usize]);
    }
}

/// A[j] = B[j]
#[gpu::cuda_kernel]
pub fn jacobi1d_kernel2(a: &mut [f32], b: &[f32], n: u32) {
    let mut a = chunk_mut(a, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j > 0 && j < n - 1 {
        a[0] = b[j as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn jacobi1d_cpu(a: &mut [f32], b: &mut [f32], n: usize, tsteps: usize) {
        for _ in 0..tsteps {
            for i in 1..n - 1 {
                b[i] = 0.33333 * (a[i - 1] + a[i] + a[i + 1]);
            }
            for j in 1..n - 1 {
                a[j] = b[j];
            }
        }
    }

    fn run_jacobi1d(h_a: &mut [f32], h_b: &mut [f32], n: usize, tsteps: usize) {
        cuda_ctx(0, |ctx, m| {
            let mut d_a = ctx.new_tensor_view(h_a.as_mut()).expect("alloc a");
            let mut d_b = ctx.new_tensor_view(h_b.as_mut()).expect("alloc b");

            let bs: u32 = 256;
            let nb: u32 = (n as u32 + bs - 1) / bs;

            for _ in 0..tsteps {
                let c1 = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                jacobi1d_kernel1::launch(c1, ctx, m, &d_a, &mut d_b, n as u32).expect("k1");
                let c2 = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                jacobi1d_kernel2::launch(c2, ctx, m, &mut d_a, &d_b, n as u32).expect("k2");
            }

            d_a.copy_to_host(h_a).expect("copy a");
            d_b.copy_to_host(h_b).expect("copy b");
        });
    }

    #[test]
    fn test_jacobi1d() {
        let n = 64;
        let tsteps = 10;

        let mut h_a_gpu: Vec<f32> = (0..n).map(|i| (4 * i + 10) as f32 / n as f32).collect();
        let mut h_b_gpu: Vec<f32> = (0..n).map(|i| (7 * i + 11) as f32 / n as f32).collect();
        let mut h_a_cpu = h_a_gpu.clone();
        let mut h_b_cpu = h_b_gpu.clone();

        run_jacobi1d(&mut h_a_gpu, &mut h_b_gpu, n, tsteps);
        jacobi1d_cpu(&mut h_a_cpu, &mut h_b_cpu, n, tsteps);

        for i in 0..n {
            assert!(
                (h_a_gpu[i] - h_a_cpu[i]).abs() < 1e-2,
                "A mismatch at {}: gpu={} cpu={}",
                i,
                h_a_gpu[i],
                h_a_cpu[i],
            );
            assert!(
                (h_b_gpu[i] - h_b_cpu[i]).abs() < 1e-2,
                "B mismatch at {}: gpu={} cpu={}",
                i,
                h_b_gpu[i],
                h_b_cpu[i],
            );
        }
    }
}
