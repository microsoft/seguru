use gpu::prelude::*;

use crate::LANE_LOG;

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
// - CUDA writes to passHist with circular-lane-shift addressing.
//   We restructure: compute exclusive prefix and write back to SAME position
//   via chunk_mut (no scatter). The final values are identical.
// - Exclusive value at position p = inclusive_scan[p-1] + cross_warp_prefix + reduction
// - passHist uses padded layout: stride = padded_thread_blocks (multiple of block_dim)
// ============================================================================

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_scan(pass_hist: &mut [u32], padded_thread_blocks: u32) {
    let tid = thread_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let lane_id = lane_id();

    // smem for within-block scan
    let smem = smem_alloc.alloc::<u32>(block_dim as usize);

    let warp_id = tid >> LANE_LOG;
    let local_size = padded_thread_blocks / block_dim;

    // chunk_mut maps chunk[k] → pass_hist[bid * padded + k * block_dim + tid]
    // Layout [t0, i0, t1]: index = tid + k * block_dim + bid * padded_thread_blocks
    let mut ph_chunk = chunk_mut(
        pass_hist,
        reshape_map!([local_size] | [block_dim, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
    );

    let mut reduction = 0u32;
    let mut partition = 0u32;
    while partition < local_size {
        // ---- Step 1: Load passHist into shared memory ----
        // CUDA: s_scan[tid] = passHist[i + digitOffset]
        let val = ph_chunk[partition];
        {
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(tid as usize).atomic_assign(val);
        }
        sync_threads();

        // ---- Step 2: Inclusive warp scan ----
        // CUDA: s_scan[tid] = InclusiveWarpScan(s_scan[tid])
        {
            let v = *smem[tid as usize];
            let scanned = crate::utils::inclusive_warp_scan(v);
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(tid as usize).atomic_assign(scanned);
        }
        sync_threads();

        // Save my inclusive value before inter-warp scan modifies warp tails
        let my_inclusive = *smem[tid as usize];

        // Workaround: replace ActiveInclusiveWarpScan (parallel warp scan on 4 threads)
        // with sequential scan on thread 0. SeGuRu JIT hangs with warp shuffle inside
        // narrow conditional (tid < n_warps). Impact: minor (only 4 values).
        if tid == 0 {
            let n_warps = block_dim >> LANE_LOG;
            let mut running = 0u32;
            let mut w = 0u32;
            while w < n_warps {
                let idx = (((w + 1) << LANE_LOG) - 1) as usize;
                let v = *smem[idx];
                running += v;
                let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
                s_atom.index(idx).atomic_assign(running);
                w += 1;
            }
        }
        sync_threads();

        // ---- Step 4: Compute exclusive prefix and write back ----
        // Workaround: replace circular-lane-shift scatter addressing with in-place
        // exclusive prefix computation. CUDA scatters to passHist[circularLaneShift + ...]
        // (data-dependent index). SeGuRu uses chunk_mut for structured writes instead,
        // computing the equivalent exclusive value at each thread's own position.
        // Exclusive within warp: shuffle_up(my_inclusive, 1) gives predecessor's inclusive.
        let (prev_inclusive, _) = gpu::shuffle!(up, my_inclusive, 1u32, 32);
        let exclusive_within_warp = if lane_id > 0 { prev_inclusive } else { 0u32 };

        // Cross-warp prefix: after inter-warp inclusive scan, smem[(w*32)-1] = total of warps 0..w-1.
        // For warp w, exclusive cross-warp = smem[(w*32)-1] for w>0, else 0.
        let cross_warp = if warp_id > 0 {
            *smem[((warp_id << LANE_LOG) - 1) as usize]
        } else {
            0u32
        };

        let final_exclusive = exclusive_within_warp + cross_warp + reduction;

        // Write back to same position via chunk_mut (no circular shift!)
        ph_chunk[partition] = final_exclusive;

        // CUDA: reduction += s_scan[blockDim.x - 1]
        reduction += *smem[(block_dim - 1) as usize];
        sync_threads();

        partition += 1;
    }
}
