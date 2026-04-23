use gpu::prelude::*;

// kernel1: E = A * B
#[gpu::cuda_kernel]
pub fn mm3_kernel1(
    a: &[f32],
    b: &[f32],
    e: &mut [f32],
    ni: u32,
    nj: u32,
    nk: u32,
) {
    let mut e = chunk_mut(e, Map2D::new(nj as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j < nj {
        let mut val = 0.0f32;
        let mut k: u32 = 0;
        while k < nk {
            val += a[(i * nk + k) as usize] * b[(k * nj + j) as usize];
            k += 1;
        }
        e[(0, 0)] = val;
    }
}

// kernel2: F = C * D
#[gpu::cuda_kernel]
pub fn mm3_kernel2(
    c: &[f32],
    d: &[f32],
    f: &mut [f32],
    nj: u32,
    nl: u32,
    nm: u32,
) {
    let mut f = chunk_mut(f, Map2D::new(nl as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < nj && j < nl {
        let mut val = 0.0f32;
        let mut k: u32 = 0;
        while k < nm {
            val += c[(i * nm + k) as usize] * d[(k * nl + j) as usize];
            k += 1;
        }
        f[(0, 0)] = val;
    }
}

// kernel3: G = E * F
#[gpu::cuda_kernel]
pub fn mm3_kernel3(
    e: &[f32],
    f: &[f32],
    g: &mut [f32],
    ni: u32,
    nj: u32,
    nl: u32,
) {
    let mut g = chunk_mut(g, Map2D::new(nl as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();

    if i < ni && j < nl {
        let mut val = 0.0f32;
        let mut k: u32 = 0;
        while k < nj {
            val += e[(i * nj + k) as usize] * f[(k * nl + j) as usize];
            k += 1;
        }
        g[(0, 0)] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_threemm(
        ni: usize,
        nj: usize,
        nk: usize,
        nl: usize,
        nm: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let h_a: Vec<f32> = vec![1.0; ni * nk];
        let h_b: Vec<f32> = vec![1.0; nk * nj];
        let h_c: Vec<f32> = vec![1.0; nj * nm];
        let h_d: Vec<f32> = vec![1.0; nm * nl];
        let mut h_e_gpu: Vec<f32> = vec![0.0; ni * nj];
        let mut h_f_gpu: Vec<f32> = vec![0.0; nj * nl];
        let mut h_g_gpu: Vec<f32> = vec![0.0; ni * nl];

        // CPU reference
        let mut h_e_cpu: Vec<f32> = vec![0.0; ni * nj];
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    h_e_cpu[i * nj + j] += h_a[i * nk + k] * h_b[k * nj + j];
                }
            }
        }
        let mut h_f_cpu: Vec<f32> = vec![0.0; nj * nl];
        for i in 0..nj {
            for j in 0..nl {
                for k in 0..nm {
                    h_f_cpu[i * nl + j] += h_c[i * nm + k] * h_d[k * nl + j];
                }
            }
        }
        let mut h_g_cpu: Vec<f32> = vec![0.0; ni * nl];
        for i in 0..ni {
            for j in 0..nl {
                for k in 0..nj {
                    h_g_cpu[i * nl + j] += h_e_cpu[i * nj + k] * h_f_cpu[k * nl + j];
                }
            }
        }

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b");
            let d_c = ctx.new_tensor_view(h_c.as_slice()).expect("alloc c");
            let d_d = ctx.new_tensor_view(h_d.as_slice()).expect("alloc d");
            let mut d_e = ctx
                .new_tensor_view(h_e_gpu.as_mut_slice())
                .expect("alloc e");
            let mut d_f = ctx
                .new_tensor_view(h_f_gpu.as_mut_slice())
                .expect("alloc f");
            let mut d_g = ctx
                .new_tensor_view(h_g_gpu.as_mut_slice())
                .expect("alloc g");

            let block_size: u32 = 16;

            // kernel1: E = A * B
            let grid_x: u32 = (nj as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            mm3_kernel1::launch(
                config, ctx, m, &d_a, &d_b, &mut d_e, ni as u32, nj as u32, nk as u32,
            )
            .expect("kernel1 launch failed");

            // kernel2: F = C * D
            let grid_x: u32 = (nl as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (nj as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            mm3_kernel2::launch(
                config, ctx, m, &d_c, &d_d, &mut d_f, nj as u32, nl as u32, nm as u32,
            )
            .expect("kernel2 launch failed");

            // kernel3: G = E * F
            let grid_x: u32 = (nl as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (ni as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            mm3_kernel3::launch(
                config, ctx, m, &d_e, &d_f, &mut d_g, ni as u32, nj as u32, nl as u32,
            )
            .expect("kernel3 launch failed");

            d_g.copy_to_host(&mut h_g_gpu).expect("copy failed");
        });

        (h_g_gpu, h_g_cpu)
    }

    #[test]
    fn test_threemm_ones() {
        let n = 16;
        let (gpu, cpu) = run_threemm(n, n, n, n, n);
        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-1,
                "Mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i],
                cpu[i],
            );
        }
        // E[i][j]=nk=16, F[i][j]=nm=16, G[i][j]=nj*16*16=4096
        assert!((gpu[0] - 4096.0).abs() < 1e-1);
    }
}
