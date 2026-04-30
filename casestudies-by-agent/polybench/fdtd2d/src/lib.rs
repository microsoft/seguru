use gpu::prelude::*;

/// Step 1: update ey. Row 0 uses fict[t], others use stencil with hz.
#[gpu::cuda_kernel]
pub fn fdtd_step1(fict: &[f32], ey: &mut [f32], hz: &[f32], nx: u32, ny: u32, t: u32) {
    let mut ey = chunk_mut(ey, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx && j < ny {
        if i == 0 {
            ey[0] = fict[t as usize];
        } else {
            ey[0] = ey[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[((i - 1) * ny + j) as usize]);
        }
    }
}

/// Step 2: update ex using hz stencil.
#[gpu::cuda_kernel]
pub fn fdtd_step2(ex: &mut [f32], hz: &[f32], nx: u32, ny: u32) {
    let mut ex = chunk_mut(ex, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx && j > 0 && j < ny {
        ex[0] = ex[0] - 0.5 * (hz[(i * ny + j) as usize] - hz[(i * ny + (j - 1)) as usize]);
    }
}

/// Step 3: update hz using ex and ey stencils.
#[gpu::cuda_kernel]
pub fn fdtd_step3(ex: &[f32], ey: &[f32], hz: &mut [f32], nx: u32, ny: u32) {
    let mut hz = chunk_mut(hz, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < nx - 1 && j < ny - 1 {
        hz[0] = hz[0]
            - 0.7
                * (ex[(i * ny + (j + 1)) as usize] - ex[(i * ny + j) as usize]
                    + ey[((i + 1) * ny + j) as usize]
                    - ey[(i * ny + j) as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn fdtd2d_cpu(
        fict: &[f32],
        ex: &mut [f32],
        ey: &mut [f32],
        hz: &mut [f32],
        nx: usize,
        ny: usize,
        tmax: usize,
    ) {
        for t in 0..tmax {
            for j in 0..ny {
                ey[0 * ny + j] = fict[t];
            }
            for i in 1..nx {
                for j in 0..ny {
                    ey[i * ny + j] = ey[i * ny + j] - 0.5 * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
                }
            }
            for i in 0..nx {
                for j in 1..ny {
                    ex[i * ny + j] = ex[i * ny + j] - 0.5 * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
                }
            }
            for i in 0..nx - 1 {
                for j in 0..ny - 1 {
                    hz[i * ny + j] = hz[i * ny + j]
                        - 0.7
                            * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j]
                                - ey[i * ny + j]);
                }
            }
        }
    }

    fn run_fdtd2d(
        fict: &[f32],
        ex: &mut [f32],
        ey: &mut [f32],
        hz: &mut [f32],
        nx: usize,
        ny: usize,
        tmax: usize,
    ) {
        cuda_ctx(0, |ctx, m| {
            let d_fict = ctx.new_tensor_view(fict).expect("alloc fict");
            let mut d_ex = ctx.new_tensor_view(ex.as_mut()).expect("alloc ex");
            let mut d_ey = ctx.new_tensor_view(ey.as_mut()).expect("alloc ey");
            let mut d_hz = ctx.new_tensor_view(hz.as_mut()).expect("alloc hz");

            let block_size: u32 = 16;
            let grid_x: u32 = (ny as u32 + block_size - 1) / block_size;
            let grid_y: u32 = (nx as u32 + block_size - 1) / block_size;

            for t in 0..tmax {
                let c1 = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                fdtd_step1::launch(
                    c1, ctx, m, &d_fict, &mut d_ey, &d_hz, nx as u32, ny as u32, t as u32,
                )
                .expect("step1");
                let c2 = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                fdtd_step2::launch(c2, ctx, m, &mut d_ex, &d_hz, nx as u32, ny as u32)
                    .expect("step2");
                let c3 = gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                fdtd_step3::launch(c3, ctx, m, &d_ex, &d_ey, &mut d_hz, nx as u32, ny as u32)
                    .expect("step3");
            }

            d_ex.copy_to_host(ex).expect("copy ex");
            d_ey.copy_to_host(ey).expect("copy ey");
            d_hz.copy_to_host(hz).expect("copy hz");
        });
    }

    #[test]
    fn test_fdtd2d() {
        let nx = 32;
        let ny = 32;
        let tmax = 5;

        let fict: Vec<f32> = (0..tmax).map(|t| t as f32).collect();

        let mut ex_gpu = vec![0.0f32; nx * ny];
        let mut ey_gpu = vec![0.0f32; nx * ny];
        let mut hz_gpu = vec![0.0f32; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                ex_gpu[i * ny + j] = (i * (j + 1)) as f32 / (nx as f32);
                ey_gpu[i * ny + j] = (i * (j + 2)) as f32 / (ny as f32);
                hz_gpu[i * ny + j] = (i * (j + 3)) as f32 / (nx as f32);
            }
        }
        let mut ex_cpu = ex_gpu.clone();
        let mut ey_cpu = ey_gpu.clone();
        let mut hz_cpu = hz_gpu.clone();

        run_fdtd2d(&fict, &mut ex_gpu, &mut ey_gpu, &mut hz_gpu, nx, ny, tmax);
        fdtd2d_cpu(&fict, &mut ex_cpu, &mut ey_cpu, &mut hz_cpu, nx, ny, tmax);

        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                assert!(
                    (ex_gpu[idx] - ex_cpu[idx]).abs() < 1e-1,
                    "ex mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    ex_gpu[idx],
                    ex_cpu[idx],
                );
                assert!(
                    (ey_gpu[idx] - ey_cpu[idx]).abs() < 1e-1,
                    "ey mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    ey_gpu[idx],
                    ey_cpu[idx],
                );
                assert!(
                    (hz_gpu[idx] - hz_cpu[idx]).abs() < 1e-1,
                    "hz mismatch at ({}, {}): gpu={} cpu={}",
                    i,
                    j,
                    hz_gpu[idx],
                    hz_cpu[idx],
                );
            }
        }
    }
}
