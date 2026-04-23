use gpu::prelude::*;

/// Vector addition: c[i] = a[i] + b[i].
///
/// CUDA equivalent:
///   __global__ void vector_add(const float *a, const float *b, float *c, int n) {
///       int idx = blockIdx.x * blockDim.x + threadIdx.x;
///       if (idx < n) c[idx] = a[idx] + b[idx];
///   }
///
/// SeGuRu note: chunk_mut assigns each thread a local view; writes use local
/// index 0, while reads use the global index on immutable slices.
#[gpu::cuda_kernel]
pub fn vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let mut c = chunk_mut(c, MapLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        c[0] = a[idx as usize] + b[idx as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    #[test]
    fn test_vector_add_basic() {
        let n: usize = 1024;
        let h_a: Vec<f32> = vec![1.0; n];
        let h_b: Vec<f32> = vec![2.0; n];
        let mut h_c: Vec<f32> = vec![0.0; n];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a failed");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b failed");
            let mut d_c = ctx
                .new_tensor_view(h_c.as_mut_slice())
                .expect("alloc c failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, 0);
            vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32)
                .expect("vector_add kernel launch failed");

            d_c.copy_to_host(&mut h_c).expect("copy from device failed");
        });

        for (i, val) in h_c.iter().enumerate() {
            assert!(
                (*val - 3.0).abs() < 1e-6,
                "Mismatch at index {}: {} != 3.0",
                i,
                val
            );
        }
    }

    #[test]
    fn test_vector_add_large() {
        let n: usize = 1 << 16; // 65536
        let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let h_b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut h_c: Vec<f32> = vec![0.0; n];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a failed");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b failed");
            let mut d_c = ctx
                .new_tensor_view(h_c.as_mut_slice())
                .expect("alloc c failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, 0);
            vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32)
                .expect("vector_add kernel launch failed");

            d_c.copy_to_host(&mut h_c).expect("copy from device failed");
        });

        for (i, val) in h_c.iter().enumerate() {
            let expected = n as f32;
            assert!(
                (*val - expected).abs() < 1e-1,
                "Mismatch at index {}: {} != {}",
                i,
                val,
                expected
            );
        }
    }
}
