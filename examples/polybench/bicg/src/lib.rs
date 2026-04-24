use gpu::prelude::*;
use gpu::CacheStreamLoadStore;

#[gpu::cuda_kernel]
pub fn bicg_kernel1(a: &[f32], r: &[f32], s: &mut [f32], nx: u32, ny: u32) {
    let mut s = chunk_mut(s, MapContinuousLinear::new(1));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if j < ny {
        let mut sum = 0.0f32;
        let mut i: u32 = 0;
        while i < nx {
            sum += r[i as usize] * a[(i * ny + j) as usize].ldcs();
            i += 1;
        }
        s[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bicg_kernel2(a: &[f32], p: &[f32], q: &mut [f32], nx: u32, ny: u32) {
    let mut q = chunk_mut(q, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < nx {
        let mut sum = 0.0f32;
        let a_row: &[f32] = &a[(i * ny) as usize..((i + 1) * ny) as usize];
        let mut j_idx: usize = 0;
        while j_idx < ny as usize {
            sum += a_row[j_idx] * p[j_idx];
            j_idx += 1;
        }
        q[0] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_bicg(
        h_a: &[f32],
        h_r: &[f32],
        h_p: &[f32],
        h_s: &mut [f32],
        h_q: &mut [f32],
        nx: usize,
        ny: usize,
    ) {
        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a).expect("alloc a");
            let d_r = ctx.new_tensor_view(h_r).expect("alloc r");
            let d_p = ctx.new_tensor_view(h_p).expect("alloc p");
            let mut d_s = ctx.new_tensor_view(h_s.as_mut()).expect("alloc s");
            let mut d_q = ctx.new_tensor_view(h_q.as_mut()).expect("alloc q");

            let bs: u32 = 256;

            let config1 = gpu_host::gpu_config!(((ny as u32) + bs - 1) / bs, 1, 1, bs, 1, 1, 0);
            bicg_kernel1::launch(config1, ctx, m, &d_a, &d_r, &mut d_s, nx as u32, ny as u32)
                .expect("kernel1 launch failed");

            let config2 = gpu_host::gpu_config!(((nx as u32) + bs - 1) / bs, 1, 1, bs, 1, 1, 0);
            bicg_kernel2::launch(config2, ctx, m, &d_a, &d_p, &mut d_q, nx as u32, ny as u32)
                .expect("kernel2 launch failed");

            d_s.copy_to_host(h_s).expect("copy s failed");
            d_q.copy_to_host(h_q).expect("copy q failed");
        });
    }

    #[test]
    fn test_bicg() {
        let nx = 64;
        let ny = 64;
        let h_a = vec![1.0f32; nx * ny];
        let h_r = vec![1.0f32; nx];
        let h_p = vec![1.0f32; ny];
        let mut h_s = vec![0.0f32; ny];
        let mut h_q = vec![0.0f32; nx];

        run_bicg(&h_a, &h_r, &h_p, &mut h_s, &mut h_q, nx, ny);

        let expected_s = nx as f32; // 64.0
        let expected_q = ny as f32; // 64.0
        for j in 0..ny {
            assert!(
                (h_s[j] - expected_s).abs() < 1e-1,
                "s mismatch at {}: gpu={} expected={}",
                j,
                h_s[j],
                expected_s,
            );
        }
        for i in 0..nx {
            assert!(
                (h_q[i] - expected_q).abs() < 1e-1,
                "q mismatch at {}: gpu={} expected={}",
                i,
                h_q[i],
                expected_q,
            );
        }
    }
}
