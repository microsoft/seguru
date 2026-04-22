use gpu::prelude::*;

#[gpu::cuda_kernel(dynamic_shared)]
pub fn reduce_per_grid<C: Copy + Sync + Default + 'static + core::ops::Add<Output = C>>(
    inputs: &[C],
    partial_sums: &mut [C],
) {
    let tid = thread_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let id = tid + block_dim * block_id::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let grid_size = block_dim * grid_dim * 2;
    let smem = smem_alloc.alloc::<C>(block_dim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut partial_sums_chunk = chunk_mut(
        partial_sums,
        reshape_map!([1] | [(block_dim, 1), grid_dim] => layout: [i0, t1, t0]),
    );

    let mut local_sum = C::default();
    for i in (id as usize..inputs.len()).step_by(grid_size as usize) {
        let left = inputs[i];
        let right = inputs[i + (grid_size / 2) as usize];
        local_sum = local_sum + left + right;
    }
    smem_chunk[0] = local_sum;
    sync_threads();
    for order in (0..10).rev() {
        let stride = 1 << order;
        if stride >= block_dim {
            continue;
        }
        let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
        if tid < stride {
            let right = smem_chunk[1];
            let left = smem_chunk[0];
            smem_chunk[0] = left + right;
        }
        sync_threads();
    }
    if tid == 0 {
        partial_sums_chunk[0] = *smem[0];
    }
}

/// Direct CUDA-style parallel reduction (sum) — mirrors reduce.cu closely.
///
/// CUDA equivalent:
///   __global__ void reduce_sum(const float *input, float *output, int n) {
///       extern __shared__ float smem[];
///       int tid = threadIdx.x;
///       int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
///       float sum = 0.0f;
///       if (idx < n) sum += input[idx];
///       if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
///       smem[tid] = sum;
///       __syncthreads();
///       for (int s = blockDim.x / 2; s > 0; s >>= 1) {
///           if (tid < s) smem[tid] += smem[tid + s];
///           __syncthreads();
///       }
///       if (tid == 0) output[blockIdx.x] = smem[0];
///   }
#[gpu::cuda_kernel(dynamic_shared)]
pub fn reduce_sum(input: &[f32], output: &mut [f32], n: usize) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = (block_id::<DimX>() * bdim * 2 + tid) as usize;

    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut output_chunk = chunk_mut(
        output,
        reshape_map!([1] | [grid_dim::<DimX>()] => layout: [i0, t0]),
    );

    // Load two elements per thread
    let mut sum = 0.0f32;
    if idx < n {
        sum += input[idx];
    }
    if idx + (bdim as usize) < n {
        sum += input[idx + bdim as usize];
    }
    smem_chunk[0] = sum;
    sync_threads();

    // Tree reduction in shared memory
    for order in (0..16).rev() {
        let stride = 1u32 << order;
        if stride >= bdim {
            continue;
        }
        let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
        if tid < stride {
            let right = smem_chunk[1];
            let left = smem_chunk[0];
            smem_chunk[0] = left + right;
        }
        sync_threads();
    }

    // Thread 0 writes block result
    if tid == 0 {
        output_chunk[0] = *smem[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;
    use rand::Rng;

    fn test_reduce_per_grid(n: u32, grid_dim: u32, block_dim: u32) {
        assert!(block_dim <= n / 2);
        assert!((grid_dim * block_dim * 2) <= n);
        let mut rng = rand::rng();
        let inputs: Vec<u32> = (0..n).map(|_| rng.random::<u32>() % 1024).collect();
        let mut partial_sums: Vec<u32> = vec![0; grid_dim as usize];
        cuda_ctx(0, |ctx, m| {
            let d_inputs = ctx
                .new_tensor_view(inputs.as_slice())
                .expect("alloc failed");
            let mut d_partial_sums = ctx
                .new_tensor_view(partial_sums.as_mut_slice())
                .expect("alloc failed");
            let smem = block_dim * core::mem::size_of::<u32>() as u32;
            let config = gpu_host::gpu_config!(grid_dim, 1, 1, block_dim, 1, 1, smem);
            reduce_per_grid::launch(config, ctx, m, &d_inputs, &mut d_partial_sums)
                .expect("reduce_per_grid kernel launch failed");
            d_partial_sums
                .copy_to_host(&mut partial_sums)
                .expect("copy from device failed");
        });

        assert_eq!(partial_sums.iter().sum::<u32>(), inputs.iter().sum::<u32>());
    }

    #[test]
    fn test_reduce_per_grid_all() {
        test_reduce_per_grid(32, 1, 16);
        test_reduce_per_grid(1024, 1, 512);
        test_reduce_per_grid(32, 1, 4);
        test_reduce_per_grid(1024, 1, 256);
        test_reduce_per_grid(1024, 16, 16);
        test_reduce_per_grid(1024, 32, 16);
    }

    fn test_reduce_sum_impl(n: usize, block_size: u32) {
        let num_blocks = ((n as u32) + block_size * 2 - 1) / (block_size * 2);
        let h_input: Vec<f32> = vec![1.0; n];
        let mut h_output: Vec<f32> = vec![0.0; num_blocks as usize];

        cuda_ctx(0, |ctx, m| {
            let d_input = ctx
                .new_tensor_view(h_input.as_slice())
                .expect("alloc failed");
            let mut d_output = ctx
                .new_tensor_view(h_output.as_mut_slice())
                .expect("alloc failed");

            let smem_bytes = block_size * core::mem::size_of::<f32>() as u32;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, smem_bytes);
            reduce_sum::launch(config, ctx, m, &d_input, &mut d_output, n)
                .expect("reduce_sum kernel launch failed");

            d_output
                .copy_to_host(&mut h_output)
                .expect("copy from device failed");
        });

        let total: f32 = h_output.iter().sum();
        assert!(
            (total - n as f32).abs() < 1e-2,
            "reduce_sum: got {}, expected {}",
            total,
            n as f32
        );
    }

    #[test]
    fn test_reduce_sum_small() {
        test_reduce_sum_impl(32, 16);
    }

    #[test]
    fn test_reduce_sum_medium() {
        test_reduce_sum_impl(1024, 256);
    }

    #[test]
    fn test_reduce_sum_large() {
        test_reduce_sum_impl(4096, 256);
    }

    #[test]
    fn test_reduce_sum_remainder() {
        // Non-multiple-of-block-size to catch OOB bugs
        test_reduce_sum_impl(100, 16);
    }
}
