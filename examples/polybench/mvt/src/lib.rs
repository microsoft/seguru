use gpu::CacheStreamLoadStore;
use gpu::prelude::*;

#[gpu::cuda_kernel]
pub fn mvt_kernel1(a: &[f32], x1: &mut [f32], y1: &[f32], n: u32) {
    let mut x1 = chunk_mut(x1, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < n {
        let mut sum = x1[0];
        let a_row: &[f32] = &a[(i * n) as usize..((i + 1) * n) as usize];
        let mut j_idx: usize = 0;
        while j_idx < n as usize {
            sum += a_row[j_idx] * y1[j_idx];
            j_idx += 1;
        }
        x1[0] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn mvt_kernel2(a: &[f32], x2: &mut [f32], y2: &[f32], n: u32) {
    let mut x2 = chunk_mut(x2, MapContinuousLinear::new(1));
    let i = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if i < n {
        let mut sum = x2[0];
        let mut j: u32 = 0;
        while j < n {
            sum += a[(j * n + i) as usize].ldcs() * y2[j as usize];
            j += 1;
        }
        x2[0] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_mvt(
        h_a: &[f32],
        h_x1: &mut [f32],
        h_x2: &mut [f32],
        h_y1: &[f32],
        h_y2: &[f32],
        n: usize,
    ) {
        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a).expect("alloc a");
            let mut d_x1 = ctx.new_tensor_view(h_x1.as_mut()).expect("alloc x1");
            let mut d_x2 = ctx.new_tensor_view(h_x2.as_mut()).expect("alloc x2");
            let d_y1 = ctx.new_tensor_view(h_y1).expect("alloc y1");
            let d_y2 = ctx.new_tensor_view(h_y2).expect("alloc y2");

            let bs: u32 = 256;
            let num_blocks: u32 = (n as u32 + bs - 1) / bs;

            let config1 = gpu_host::gpu_config!(num_blocks, 1, 1, bs, 1, 1, 0);
            mvt_kernel1::launch(config1, ctx, m, &d_a, &mut d_x1, &d_y1, n as u32)
                .expect("kernel1 launch failed");

            let config2 = gpu_host::gpu_config!(num_blocks, 1, 1, bs, 1, 1, 0);
            mvt_kernel2::launch(config2, ctx, m, &d_a, &mut d_x2, &d_y2, n as u32)
                .expect("kernel2 launch failed");

            d_x1.copy_to_host(h_x1).expect("copy x1 failed");
            d_x2.copy_to_host(h_x2).expect("copy x2 failed");
        });
    }

    #[test]
    fn test_mvt() {
        let n = 64;
        let h_a = vec![1.0f32; n * n];
        let mut h_x1 = vec![0.0f32; n];
        let mut h_x2 = vec![0.0f32; n];
        let h_y1 = vec![1.0f32; n];
        let h_y2 = vec![1.0f32; n];

        run_mvt(&h_a, &mut h_x1, &mut h_x2, &h_y1, &h_y2, n);

        let expected = n as f32; // 64.0
        for i in 0..n {
            assert!(
                (h_x1[i] - expected).abs() < 1e-1,
                "x1 mismatch at {}: gpu={} expected={}",
                i,
                h_x1[i],
                expected,
            );
            assert!(
                (h_x2[i] - expected).abs() < 1e-1,
                "x2 mismatch at {}: gpu={} expected={}",
                i,
                h_x2[i],
                expected,
            );
        }
    }
}
