use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn conv2d_kernel(a: &[f32], b: &mut [f32], ni: u32, nj: u32) {
    let mut b = chunk_mut(b, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    let c11: f32 = 0.2;
    let c21: f32 = 0.5;
    let c31: f32 = -0.8;
    let c12: f32 = -0.3;
    let c22: f32 = 0.6;
    let c32: f32 = -0.9;
    let c13: f32 = 0.4;
    let c23: f32 = 0.7;
    let c33: f32 = 0.10;

    if i > 0 && i < ni - 1 && j > 0 && j < nj - 1 {
        b[0] = c11 * a[((i - 1) * nj + (j - 1)) as usize]
            + c21 * a[((i - 1) * nj + j) as usize]
            + c31 * a[((i - 1) * nj + (j + 1)) as usize]
            + c12 * a[(i * nj + (j - 1)) as usize]
            + c22 * a[(i * nj + j) as usize]
            + c32 * a[(i * nj + (j + 1)) as usize]
            + c13 * a[((i + 1) * nj + (j - 1)) as usize]
            + c23 * a[((i + 1) * nj + j) as usize]
            + c33 * a[((i + 1) * nj + (j + 1)) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn cpu_conv2d(a: &[f32], b: &mut [f32], ni: usize, nj: usize) {
        let c11: f32 = 0.2;
        let c21: f32 = 0.5;
        let c31: f32 = -0.8;
        let c12: f32 = -0.3;
        let c22: f32 = 0.6;
        let c32: f32 = -0.9;
        let c13: f32 = 0.4;
        let c23: f32 = 0.7;
        let c33: f32 = 0.10;

        for i in 1..ni - 1 {
            for j in 1..nj - 1 {
                b[i * nj + j] = c11 * a[(i - 1) * nj + (j - 1)]
                    + c21 * a[(i - 1) * nj + j]
                    + c31 * a[(i - 1) * nj + (j + 1)]
                    + c12 * a[i * nj + (j - 1)]
                    + c22 * a[i * nj + j]
                    + c32 * a[i * nj + (j + 1)]
                    + c13 * a[(i + 1) * nj + (j - 1)]
                    + c23 * a[(i + 1) * nj + j]
                    + c33 * a[(i + 1) * nj + (j + 1)];
            }
        }
    }

    fn run_conv2d(ni: usize, nj: usize) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = (0..ni * nj).map(|i| i as f32).collect();
        let mut h_b_gpu: Vec<f32> = vec![0.0; ni * nj];
        let mut h_b_cpu: Vec<f32> = vec![0.0; ni * nj];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let mut d_b = ctx
                .new_tensor_view(h_b_gpu.as_mut_slice())
                .expect("alloc b");

            let block_size: u32 = 16;
            let grid_x: u32 = (nj as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            conv2d_kernel::launch(config, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32)
                .expect("kernel launch failed");

            d_b.copy_to_host(&mut h_b_gpu).expect("copy failed");
        });

        cpu_conv2d(&h_a, &mut h_b_cpu, ni, nj);
        (h_b_gpu, h_b_cpu)
    }

    #[test]
    fn test_conv2d() {
        let ni = 64;
        let nj = 64;
        let (gpu, cpu) = run_conv2d(ni, nj);
        for i in 1..ni - 1 {
            for j in 1..nj - 1 {
                let idx = i * nj + j;
                assert!(
                    (gpu[idx] - cpu[idx]).abs() < 1e-2,
                    "Mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    gpu[idx],
                    cpu[idx]
                );
            }
        }
    }
}
