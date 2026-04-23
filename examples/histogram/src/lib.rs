use gpu::prelude::*;
use gpu::sync::{Atomic, SharedAtomic};

const NUM_BINS: usize = 256;

/// Histogram with shared memory + atomics.
///
/// CUDA equivalent:
///   __global__ void histogram(const unsigned int *data, unsigned int *bins, int n) {
///       __shared__ unsigned int smem_bins[256];
///       int tid = threadIdx.x;
///       int idx = blockIdx.x * blockDim.x + threadIdx.x;
///       int stride = blockDim.x * gridDim.x;
///       for (int i = tid; i < NUM_BINS; i += blockDim.x) smem_bins[i] = 0;
///       __syncthreads();
///       for (int i = idx; i < n; i += stride) atomicAdd(&smem_bins[data[i]], 1);
///       __syncthreads();
///       for (int i = tid; i < NUM_BINS; i += blockDim.x) atomicAdd(&bins[i], smem_bins[i]);
///   }
#[gpu::cuda_kernel(dynamic_shared)]
pub fn histogram_kernel(data: &[u32], bins: &mut [u32], n: u32) {
    let smem_bins = smem_alloc.alloc::<u32>(NUM_BINS);

    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = block_id::<DimX>() * bdim + tid;
    let stride = bdim * grid_dim::<DimX>();

    // Initialize shared memory bins to zero
    {
        let mut smem_chunk = smem_bins.chunk_mut(MapContinuousLinear::new(1));
        let mut i: u32 = tid;
        let mut local_i: u32 = 0;
        while i < NUM_BINS as u32 {
            smem_chunk[local_i] = 0;
            i += bdim;
            local_i += 1;
        }
    }
    sync_threads();

    // Accumulate into shared memory bins using atomics
    let smem_atomic = SharedAtomic::new(smem_bins);
    let mut i: u32 = idx;
    while i < n {
        let bin = data[i as usize] as usize;
        smem_atomic.index(bin).atomic_addi(1u32);
        i += stride;
    }
    sync_threads();

    // Merge shared memory bins into global bins using atomics
    let bins_atomic = Atomic::new(bins);
    let mut i: u32 = tid;
    while i < NUM_BINS as u32 {
        let local_count = *smem_bins[i as usize];
        if local_count > 0 {
            bins_atomic.index(i as usize).atomic_addi(local_count);
        }
        i += bdim;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_histogram(h_data: &[u32], h_bins: &mut [u32], n: usize, block_size: u32) {
        cuda_ctx(0, |ctx, m| {
            let d_data = ctx
                .new_tensor_view(h_data)
                .expect("alloc data failed");
            let mut d_bins = ctx
                .new_tensor_view(h_bins.as_mut())
                .expect("alloc bins failed");

            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let num_blocks = num_blocks.min(1024);
            let smem_bytes = (NUM_BINS as u32) * core::mem::size_of::<u32>() as u32;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, smem_bytes);
            histogram_kernel::launch(config, ctx, m, &d_data, &mut d_bins, n as u32)
                .expect("histogram kernel launch failed");

            d_bins
                .copy_to_host(h_bins)
                .expect("copy from device failed");
        });
    }

    #[test]
    fn test_histogram_uniform() {
        let n: usize = 1 << 16; // 65536
        let h_data: Vec<u32> = (0..n).map(|i| (i % NUM_BINS) as u32).collect();
        let mut h_bins: Vec<u32> = vec![0u32; NUM_BINS];

        run_histogram(&h_data, &mut h_bins, n, 256);

        let expected = (n / NUM_BINS) as u32;
        for (i, count) in h_bins.iter().enumerate() {
            assert_eq!(
                *count, expected,
                "Bin {} has count {}, expected {}",
                i, count, expected
            );
        }
    }

    #[test]
    fn test_histogram_single_bin() {
        let n: usize = 1024;
        let h_data: Vec<u32> = vec![42u32; n];
        let mut h_bins: Vec<u32> = vec![0u32; NUM_BINS];

        run_histogram(&h_data, &mut h_bins, n, 256);

        assert_eq!(h_bins[42], n as u32, "Bin 42 should have all {} counts", n);
        let total: u32 = h_bins.iter().sum();
        assert_eq!(total, n as u32, "Total count should be {}", n);
    }
}
