use gpu::prelude::*;

/// 3D convolution kernel. Each thread computes one element of B from a 3x3x3 stencil of A.
#[gpu::cuda_kernel]
pub fn conv3d_kernel(a: &[f32], b: &mut [f32], ni: u32, nj: u32, nk: u32) {
    let mut b = chunk_mut(b, MapLinear::new(1));
    let k = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let ji = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    let j = ji % nj;
    let i = ji / nj;

    let c11: f32 = 2.0;
    let c21: f32 = 5.0;
    let c31: f32 = -8.0;
    let c12: f32 = -3.0;
    let c22: f32 = 6.0;
    let c32: f32 = -9.0;
    let c13: f32 = 4.0;
    let c23: f32 = 7.0;
    let c33: f32 = 10.0;

    if i > 0 && i < ni - 1 && j > 0 && j < nj - 1 && k > 0 && k < nk - 1 {
        b[0] = c11 * a[((i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c13 * a[((i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c21 * a[((i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c23 * a[((i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c31 * a[((i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c33 * a[((i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)) as usize]
            + c12 * a[(i * (nk * nj) + (j - 1) * nk + k) as usize]
            + c22 * a[(i * (nk * nj) + j * nk + k) as usize]
            + c32 * a[(i * (nk * nj) + (j + 1) * nk + k) as usize]
            + c11 * a[((i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)) as usize]
            + c13 * a[((i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)) as usize]
            + c21 * a[((i - 1) * (nk * nj) + j * nk + (k + 1)) as usize]
            + c23 * a[((i + 1) * (nk * nj) + j * nk + (k + 1)) as usize]
            + c31 * a[((i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)) as usize]
            + c33 * a[((i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn conv3d_cpu(a: &[f32], b: &mut [f32], ni: usize, nj: usize, nk: usize) {
        let c11: f32 = 2.0;
        let c21: f32 = 5.0;
        let c31: f32 = -8.0;
        let c12: f32 = -3.0;
        let c22: f32 = 6.0;
        let c32: f32 = -9.0;
        let c13: f32 = 4.0;
        let c23: f32 = 7.0;
        let c33: f32 = 10.0;

        for i in 1..ni - 1 {
            for j in 1..nj - 1 {
                for k in 1..nk - 1 {
                    b[i * (nk * nj) + j * nk + k] = c11
                        * a[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c13 * a[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c21 * a[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c23 * a[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c31 * a[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c33 * a[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                        + c12 * a[i * (nk * nj) + (j - 1) * nk + k]
                        + c22 * a[i * (nk * nj) + j * nk + k]
                        + c32 * a[i * (nk * nj) + (j + 1) * nk + k]
                        + c11 * a[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)]
                        + c13 * a[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)]
                        + c21 * a[(i - 1) * (nk * nj) + j * nk + (k + 1)]
                        + c23 * a[(i + 1) * (nk * nj) + j * nk + (k + 1)]
                        + c31 * a[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)]
                        + c33 * a[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)];
                }
            }
        }
    }

    fn run_conv3d(ni: usize, nj: usize, nk: usize) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = (0..ni * nj * nk)
            .map(|idx| (idx % 17) as f32 * 0.5)
            .collect();
        let mut h_b_gpu = vec![0.0f32; ni * nj * nk];
        let mut h_b_cpu = vec![0.0f32; ni * nj * nk];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let mut d_b = ctx
                .new_tensor_view(h_b_gpu.as_mut_slice())
                .expect("alloc b");

            let bx: u32 = 16;
            let by: u32 = 16;
            let grid_x: u32 = (nk as u32 + bx - 1) / bx;
            let grid_y: u32 = ((ni * nj) as u32 + by - 1) / by;
            let config = gpu_host::gpu_config!(grid_x, grid_y, 1, bx, by, 1, 0);

            conv3d_kernel::launch(config, ctx, m, &d_a, &mut d_b, ni as u32, nj as u32, nk as u32)
                .expect("kernel launch");

            d_b.copy_to_host(&mut h_b_gpu).expect("copy failed");
        });

        conv3d_cpu(&h_a, &mut h_b_cpu, ni, nj, nk);
        (h_b_gpu, h_b_cpu)
    }

    #[test]
    fn test_conv3d() {
        let ni = 16;
        let nj = 16;
        let nk = 16;
        let (gpu, cpu) = run_conv3d(ni, nj, nk);

        for i in 1..ni - 1 {
            for j in 1..nj - 1 {
                for k in 1..nk - 1 {
                    let idx = i * (nk * nj) + j * nk + k;
                    assert!(
                        (gpu[idx] - cpu[idx]).abs() < 1e-1,
                        "Mismatch at ({}, {}, {}): gpu={} cpu={}",
                        i,
                        j,
                        k,
                        gpu[idx],
                        cpu[idx],
                    );
                }
            }
        }
    }
}
