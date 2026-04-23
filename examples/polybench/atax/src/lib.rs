use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn atax_kernel1(a: &[f32], x: &[f32], tmp: &mut [f32], nx: u32, ny: u32) {
    let mut tmp = chunk_mut(tmp, MapLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < nx {
        let mut sum = 0.0f32;
        let mut j: u32 = 0;
        while j < ny {
            sum += a[(i * ny + j) as usize] * x[j as usize];
            j += 1;
        }
        tmp[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn atax_kernel2(a: &[f32], tmp: &[f32], y: &mut [f32], nx: u32, ny: u32) {
    let mut y = chunk_mut(y, MapLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < nx {
            sum += a[(i * ny + j) as usize] * tmp[i as usize];
            i += 1;
        }
        y[0] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_atax(h_a: &[f32], h_x: &[f32], h_y: &mut [f32], nx: usize, ny: usize) {
        let mut h_tmp = vec![0.0f32; nx];
        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a).expect("alloc a");
            let d_x = ctx.new_tensor_view(h_x).expect("alloc x");
            let mut d_tmp = ctx.new_tensor_view(h_tmp.as_mut_slice()).expect("alloc tmp");
            let mut d_y = ctx.new_tensor_view(h_y.as_mut()).expect("alloc y");

            let bs: u32 = 256;
            let config1 = gpu_host::gpu_config!(((nx as u32) + bs - 1) / bs, 1, 1, bs, 1, 1, 0);
            atax_kernel1::launch(config1, ctx, m, &d_a, &d_x, &mut d_tmp, nx as u32, ny as u32)
                .expect("kernel1 launch failed");

            let config2 = gpu_host::gpu_config!(((ny as u32) + bs - 1) / bs, 1, 1, bs, 1, 1, 0);
            atax_kernel2::launch(config2, ctx, m, &d_a, &d_tmp, &mut d_y, nx as u32, ny as u32)
                .expect("kernel2 launch failed");

            d_y.copy_to_host(h_y).expect("copy failed");
        });
    }

    #[test]
    fn test_atax() {
        let nx = 64;
        let ny = 64;
        let h_a = vec![1.0f32; nx * ny];
        let h_x = vec![1.0f32; ny];
        let mut h_y = vec![0.0f32; ny];

        run_atax(&h_a, &h_x, &mut h_y, nx, ny);

        let expected = (nx * ny) as f32; // 4096.0
        for j in 0..ny {
            assert!(
                (h_y[j] - expected).abs() < 1e-1,
                "Mismatch at {}: gpu={} expected={}",
                j,
                h_y[j],
                expected,
            );
        }
    }
}
