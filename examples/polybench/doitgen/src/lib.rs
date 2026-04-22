use gpu::prelude::*;

// sum[r*(nq*np) + q*np + p] = sum_s(A[r*(nq*np) + q*np + s] * C4[s*np + p])
// Pack r,q into Y dimension: qr = r*nq + q
#[gpu::cuda_kernel]
pub fn doitgen_kernel1(
    a: &[f32],
    c4: &[f32],
    sum_arr: &mut [f32],
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut sum_arr = chunk_mut(sum_arr, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        let mut val = 0.0f32;
        let mut s: usize = 0;
        while s < np {
            val += a[r * (nq * np) + q * np + s] * c4[s * np + p];
            s += 1;
        }
        sum_arr[0] = val;
    }
}

// A[r*(nq*np) + q*np + p] = sum[r*(nq*np) + q*np + p]
#[gpu::cuda_kernel]
pub fn doitgen_kernel2(
    sum_arr: &[f32],
    a: &mut [f32],
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        a[0] = sum_arr[r * (nq * np) + q * np + p];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_doitgen(nr: usize, nq: usize, np: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_a: Vec<f32> = vec![0.0; nr * nq * np];
        let mut h_c4: Vec<f32> = vec![0.0; np * np];
        for i in 0..nr {
            for j in 0..nq {
                for k in 0..np {
                    h_a[i * (nq * np) + j * np + k] = (i * j + k) as f32 / np as f32;
                }
            }
        }
        for i in 0..np {
            for j in 0..np {
                h_c4[i * np + j] = (i * j) as f32 / np as f32;
            }
        }

        // CPU reference
        let mut h_a_cpu = h_a.clone();
        for r in 0..nr {
            for q in 0..nq {
                let mut sum_tmp = vec![0.0f32; np];
                for p in 0..np {
                    let mut val = 0.0f32;
                    for s in 0..np {
                        val += h_a_cpu[r * (nq * np) + q * np + s] * h_c4[s * np + p];
                    }
                    sum_tmp[p] = val;
                }
                for p in 0..np {
                    h_a_cpu[r * (nq * np) + q * np + p] = sum_tmp[p];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_sum: Vec<f32> = vec![0.0; nr * nq * np];

        cuda_ctx(0, |ctx, m_module| {
            let d_a_ro = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a_ro");
            let mut d_a = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a");
            let d_c4 = ctx.new_tensor_view(h_c4.as_slice()).expect("alloc c4");
            let mut d_sum = ctx
                .new_tensor_view(h_sum.as_mut_slice())
                .expect("alloc sum");

            let bx: u32 = np as u32;
            let by: u32 = 16.min((nr * nq) as u32);
            let grid_x = (np as u32 + bx - 1) / bx;
            let grid_y = ((nr * nq) as u32 + by - 1) / by;

            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, bx, by, 1, 0);
            doitgen_kernel1::launch(
                config, ctx, m_module, &d_a_ro, &d_c4, &mut d_sum, nr, nq, np,
            )
            .expect("kernel1 failed");

            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, bx, by, 1, 0);
            doitgen_kernel2::launch(
                config, ctx, m_module, &d_sum, &mut d_a, nr, nq, np,
            )
            .expect("kernel2 failed");

            d_a.copy_to_host(&mut h_a_gpu).expect("copy failed");
        });

        (h_a_gpu, h_a_cpu)
    }

    #[test]
    fn test_doitgen() {
        let nr = 8;
        let nq = 8;
        let np = 8;
        let (gpu, cpu) = run_doitgen(nr, nq, np);

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "result is all zeros");
    }
}
