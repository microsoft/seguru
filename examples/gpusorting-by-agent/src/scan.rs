use gpu::prelude::*;

use crate::{LANE_LOG, LANE_MASK, SCAN_THREADS};

// ============================================================================
// CUDA reference: DeviceRadixSort::Scan
// See cuda-ref/GPUSortingCUDA/Sort/DeviceRadixSort.cu lines 100–155
//
// __global__ void Scan(uint32_t* passHist, uint32_t threadBlocks) {
//     __shared__ uint32_t s_scan[128];
//     uint32_t reduction = 0;
//     const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;
//     const uint32_t partitionsEnd = threadBlocks / blockDim.x * blockDim.x;
//     const uint32_t digitOffset = blockIdx.x * threadBlocks;
//
//     for (uint32_t i = threadIdx.x; i < partitionsEnd; i += blockDim.x) {
//         s_scan[tid] = passHist[i + digitOffset];
//         s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
//         __syncthreads();
//         if (tid < (bdim >> LANE_LOG))
//             s_scan[(tid + 1 << LANE_LOG) - 1] =
//                 ActiveInclusiveWarpScan(s_scan[(tid + 1 << LANE_LOG) - 1]);
//         __syncthreads();
//         passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset] =
//             (getLaneId() != LANE_MASK ? s_scan[tid] : 0) +
//             (tid >= LANE_COUNT ? __shfl_sync(0xffffffff, s_scan[tid-1], 0) : 0) +
//             reduction;
//         reduction += s_scan[blockDim.x - 1];
//         __syncthreads();
//     }
//     // tail (partial partition, similar structure)
// }
//
// PORTING NOTES:
// - passHist needs both read and write. Use chunk_mut with reshape_map for the
//   structured per-digit access, and Atomic for the scatter-write with circular shift.
// - CUDA's tail section doesn't zero s_scan for OOB threads (stale data).
//   We zero explicitly for correctness.
// ============================================================================

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_scan(pass_hist: &mut [u32], thread_blocks: u32) {
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let lane_id = lane_id();

    // CUDA: __shared__ uint32_t s_scan[128];
    let smem = smem_alloc.alloc::<u32>(SCAN_THREADS as usize);

    // passHist is read+write → Atomic for data-dependent circular-shift scatter
    let ph = gpu::sync::Atomic::new(pass_hist);

    let mut reduction = 0u32;
    // CUDA: circularLaneShift = (getLaneId() + 1) & LANE_MASK
    let circular_lane_shift = (lane_id + 1) & LANE_MASK;
    // CUDA: partitionsEnd = threadBlocks / blockDim.x * blockDim.x
    let partitions_end = thread_blocks / block_dim * block_dim;
    // CUDA: digitOffset = blockIdx.x * threadBlocks
    let digit_offset = bid * thread_blocks;

    // Main loop: full partitions of block_dim elements
    let num_full_partitions = partitions_end / block_dim;
    let mut partition = 0u32;
    while partition < num_full_partitions {
        let i = partition * block_dim + tid;

        // CUDA: s_scan[tid] = passHist[i + digitOffset];
        {
            let val = ph.index((i + digit_offset) as usize).atomic_addi(0u32);
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(tid as usize).atomic_assign(val);
        }
        sync_threads();

        // CUDA: s_scan[tid] = InclusiveWarpScan(s_scan[tid]);
        {
            let val = *smem[tid as usize];
            let scanned = crate::utils::inclusive_warp_scan(val);
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(tid as usize).atomic_assign(scanned);
        }
        sync_threads();

        // CUDA: if (tid < (bdim >> LANE_LOG))
        //           s_scan[(tid+1 << LANE_LOG) - 1] = ActiveInclusiveWarpScan(...)
        // JIT hang workaround: thread-0 sequential inclusive scan on warp-tail positions.
        if tid == 0 {
            let n_warps = block_dim >> LANE_LOG; // 4
            let mut running = 0u32;
            let mut w = 0u32;
            while w < n_warps {
                let idx = (((w + 1) << LANE_LOG) - 1) as usize;
                let val = *smem[idx];
                running += val;
                let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
                s_atomic.index(idx).atomic_assign(running);
                w += 1;
            }
        }
        sync_threads();

        // CUDA: passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset] =
        //           (getLaneId() != LANE_MASK ? s_scan[tid] : 0) +
        //           (tid >= LANE_COUNT ? __shfl_sync(..., s_scan[tid-1], 0) : 0) +
        //           reduction;
        {
            let scan_val = *smem[tid as usize];
            let is_last_in_warp = lane_id == LANE_MASK;

            // Cross-warp: broadcast preceding warp's last element from lane 0
            let cross_warp = if tid >= 32u32 {
                *smem[(tid - 1) as usize]
            } else {
                0u32
            };
            let (shuffled_cross, _) = gpu::shuffle!(idx, cross_warp, 0u32, 32u32);

            let warp_sum = if is_last_in_warp { 0u32 } else { scan_val };
            let inter_warp_sum = if tid >= 32u32 { shuffled_cross } else { 0u32 };
            let final_val = warp_sum + inter_warp_sum + reduction;

            // Circular-shift scatter: data-dependent output position
            let output_idx = circular_lane_shift + (i & !LANE_MASK);
            ph.index((output_idx + digit_offset) as usize)
                .atomic_assign(final_val);
        }

        // CUDA: reduction += s_scan[blockDim.x - 1];
        reduction += *smem[(block_dim - 1) as usize];
        sync_threads();

        partition += 1;
    }

    // Tail: partial partition
    let i = partitions_end + tid;

    // CUDA: if (i < threadBlocks) s_scan[tid] = passHist[i + digitOffset];
    // Zero explicitly for OOB threads (CUDA stale data bug fix).
    {
        let val = if i < thread_blocks {
            ph.index((i + digit_offset) as usize).atomic_addi(0u32)
        } else {
            0u32
        };
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atom.index(tid as usize).atomic_assign(val);
    }
    sync_threads();

    // Inclusive warp scan on tail
    {
        let val = *smem[tid as usize];
        let scanned = crate::utils::inclusive_warp_scan(val);
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atom.index(tid as usize).atomic_assign(scanned);
    }
    sync_threads();

    // Inter-warp scan (sequential, thread 0)
    if tid == 0 {
        let n_warps = block_dim >> LANE_LOG;
        let mut running = 0u32;
        let mut w = 0u32;
        while w < n_warps {
            let idx = (((w + 1) << LANE_LOG) - 1) as usize;
            let val = *smem[idx];
            running += val;
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(idx).atomic_assign(running);
            w += 1;
        }
    }
    sync_threads();

    // Tail output: same circular-shift scatter pattern
    let output_idx = circular_lane_shift + (i & !LANE_MASK);
    if output_idx < thread_blocks {
        let scan_val = *smem[tid as usize];
        let is_last_in_warp = lane_id == LANE_MASK;

        // Cross-warp: read predecessor warp's last element directly
        let pred_val = if tid >= 32u32 {
            *smem[((tid & !LANE_MASK) - 1) as usize]
        } else {
            0u32
        };

        let warp_sum = if is_last_in_warp { 0u32 } else { scan_val };
        let inter_warp_sum = if tid >= 32u32 { pred_val } else { 0u32 };
        let final_val = warp_sum + inter_warp_sum + reduction;

        ph.index((output_idx + digit_offset) as usize)
            .atomic_assign(final_val);
    }
}
