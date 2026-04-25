use gpu::prelude::*;

use crate::{LANE_LOG, LANE_MASK, SCAN_THREADS};

/// Exclusive prefix sum scan kernel.
/// Performs an exclusive prefix sum on per-block pass histogram.
/// Launched with 256 blocks (one per digit), SCAN_THREADS (128) threads per block.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_scan(pass_hist: &mut [u32], thread_blocks: u32) {
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let lane_id = lane_id();

    // Allocate shared memory for the scan array
    let smem = smem_alloc.alloc::<u32>(SCAN_THREADS as usize);
    let mut ph = chunk_mut(pass_hist, MapLinear::new(1));

    let mut reduction = 0u32;
    let circular_lane_shift = (lane_id + 1) & LANE_MASK;
    let partitions_end = thread_blocks / block_dim * block_dim;
    let digit_offset = bid * thread_blocks;

    // Main loop: process full partitions
    // Use uniform loop variable so SeGuRu can verify all threads iterate equally
    let num_full_partitions = partitions_end / block_dim;
    let mut partition = 0u32;
    while partition < num_full_partitions {
        let i = partition * block_dim + tid;
        // Load from pass_hist into shared memory
        {
            let val = ph[(i + digit_offset) as usize];
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(tid as usize).atomic_assign(val);
        }
        sync_threads();

        // Inclusive warp scan
        {
            let val = *smem[tid as usize];
            let scanned = crate::utils::inclusive_warp_scan(val);
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(tid as usize).atomic_assign(scanned);
        }
        sync_threads();

        // Inter-warp scan phase: threads in first few lanes scan warp representatives
        if tid < (block_dim >> LANE_LOG) {
            let idx = ((tid + 1) << LANE_LOG) - 1;
            let val = *smem[idx as usize];
            let scanned = crate::utils::inclusive_warp_scan(val);
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(idx as usize).atomic_assign(scanned);
        }
        sync_threads();

        // Write back to pass_hist with exclusive scan logic
        {
            let scan_val = *smem[tid as usize];
            let is_last_in_warp = lane_id == LANE_MASK;

            // Get previous thread's value for cross-warp contribution
            let cross_warp = if tid >= 32u32 {
                *smem[(tid - 1) as usize]
            } else {
                0u32
            };

            // Shuffle to get lane 0's value of the predecessor
            let (shuffled_cross, _) = gpu::shuffle!(idx, cross_warp, 0u32, 32u32);

            let warp_sum = if is_last_in_warp { 0u32 } else { scan_val };
            let inter_warp_sum = if tid >= 32u32 { shuffled_cross } else { 0u32 };
            let final_val = warp_sum + inter_warp_sum + reduction;

            let output_idx = circular_lane_shift + (i & !LANE_MASK);
            ph[(output_idx + digit_offset) as usize] = final_val;
        }

        // Update reduction for next iteration
        reduction += *smem[(block_dim - 1) as usize];
        sync_threads();

        partition += 1;
    }
    let i = partitions_end + tid;

    // Tail handling: process remaining partial partition
    if i < thread_blocks {
        let val = ph[(i + digit_offset) as usize];
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atomic.index(tid as usize).atomic_assign(val);
    } else {
        // Ensure we have 0 in shared memory for threads beyond the tail
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atomic.index(tid as usize).atomic_assign(0u32);
    }
    sync_threads();

    // Inclusive warp scan on tail
    {
        let val = *smem[tid as usize];
        let scanned = crate::utils::inclusive_warp_scan(val);
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atomic.index(tid as usize).atomic_assign(scanned);
    }
    sync_threads();

    // Inter-warp scan phase for tail
    if tid < (block_dim >> LANE_LOG) {
        let idx = ((tid + 1) << LANE_LOG) - 1;
        let val = *smem[idx as usize];
        let scanned = crate::utils::inclusive_warp_scan(val);
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atomic.index(idx as usize).atomic_assign(scanned);
    }
    sync_threads();

    // Write tail results
    let output_idx = circular_lane_shift + (i & !LANE_MASK);
    if output_idx < thread_blocks {
        let scan_val = *smem[tid as usize];
        let is_last_in_warp = lane_id == LANE_MASK;

        // Get predecessor value
        let pred_val = if tid >= 32u32 {
            *smem[((tid & !LANE_MASK) - 1) as usize]
        } else {
            0u32
        };

        let warp_sum = if is_last_in_warp { 0u32 } else { scan_val };
        let inter_warp_sum = if tid >= 32u32 { pred_val } else { 0u32 };
        let final_val = warp_sum + inter_warp_sum + reduction;

        ph[(output_idx + digit_offset) as usize] = final_val;
    }
}
