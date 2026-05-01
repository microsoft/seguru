use gpu::prelude::*;

use crate::{LANE_LOG, PART_SIZE, RADIX, RADIX_MASK};

const VEC_PART_SIZE: u32 = PART_SIZE / 4;

// ============================================================================
// CUDA reference: DeviceRadixSort::Upsweep
// See cuda-ref/GPUSortingCUDA/Sort/DeviceRadixSort.cu lines 39–98
//
// __global__ void Upsweep(uint32_t* sort, uint32_t* globalHist,
//                          uint32_t* passHist, uint32_t size, uint32_t radixShift)
// {
//     __shared__ uint32_t s_globalHist[RADIX * 2];
//     for (uint32_t i = threadIdx.x; i < RADIX*2; i += blockDim.x) s_globalHist[i] = 0;
//     __syncthreads();
//
//     // histogram — 64 threads per sub-histogram (2 wave sub-hists)
//     {
//         uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];
//         if (blockIdx.x < gridDim.x - 1) {
//             const uint4* p = reinterpret_cast<const uint4*>(&sort[partStart]);
//             for (uint32_t i = tid; i < VEC_PART_SIZE; i += blockDim.x) {
//                 uint4 t = p[i];
//                 atomicAdd(&s_wavesHist[t.x >> shift & 0xFF], 1);
//                 atomicAdd(&s_wavesHist[t.y >> shift & 0xFF], 1);
//                 atomicAdd(&s_wavesHist[t.z >> shift & 0xFF], 1);
//                 atomicAdd(&s_wavesHist[t.w >> shift & 0xFF], 1);
//             }
//         }
//         if (blockIdx.x == gridDim.x - 1) {
//             for (uint32_t i = ...) atomicAdd(&s_wavesHist[sort[i] >> radixShift & RADIX_MASK], 1);
//         }
//     }
//     __syncthreads();
//
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
//         s_globalHist[i] += s_globalHist[i + RADIX];
//         passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
//         s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
//     }
//     __syncthreads();
//
//     if (threadIdx.x < (RADIX >> LANE_LOG))
//         s_globalHist[threadIdx.x << LANE_LOG] =
//             ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
//     __syncthreads();
//
//     for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
//         atomicAdd(&globalHist[i + (radixShift << 5)],
//                   s_globalHist[i] + (getLaneId() ? __shfl_sync(..., s_globalHist[i-1], 1) : 0));
// }
// ============================================================================

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_upsweep(
    sort: &[U32_4],
    global_hist: &mut [u32],
    pass_hist: &mut [u32],
    size: u32,
    radix_shift: u32,
    padded_thread_blocks: u32,
) {
    let tid = thread_id::<DimX>();
    let block_id = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let lane_id = lane_id();

    // CUDA: __shared__ uint32_t s_globalHist[RADIX * 2];
    // Two sub-histograms of RADIX bins each, used by two wave groups (tid/64).
    let smem = smem_alloc.alloc::<u32>(RADIX as usize * 2);

    // Zero shared memory: chunk_mut with MapLinear gives each thread a strided chunk
    {
        let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
        let num_per_thread = (RADIX * 2) / block_dim;
        let mut k = 0u32;
        while k < num_per_thread {
            smem_chunk[k as usize] = 0u32;
            k += 1;
        }
    }
    sync_threads();

    // CUDA: histogram — atomicAdd to per-wave shared histogram
    // Two wave groups (threads 0-63 → sub-hist 0, threads 64-127 → sub-hist 1).
    // Multiple threads in the same wave may hit the same bin → SharedAtomic needed.
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        // CUDA: uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];
        let wave_offset = (tid / 64) * RADIX;

        // CUDA: if (blockIdx.x < gridDim.x - 1) — non-last block, full partition
        // Uses U32_4 vectorized loads (4 keys per load, VEC_PART_SIZE iterations)
        // matching CUDA's uint4 reinterpret_cast pattern.
        if block_id < grid_dim - 1 {
            let vec_part_start = block_id * VEC_PART_SIZE;
            let mut i = tid + vec_part_start;
            let vec_part_end = vec_part_start + VEC_PART_SIZE;
            while i < vec_part_end {
                let keys = sort[i as usize];
                let k = keys.data();
                let d0 = (k[0] >> radix_shift) & RADIX_MASK;
                let d1 = (k[1] >> radix_shift) & RADIX_MASK;
                let d2 = (k[2] >> radix_shift) & RADIX_MASK;
                let d3 = (k[3] >> radix_shift) & RADIX_MASK;
                s_atomic.index((wave_offset + d0) as usize).atomic_addi(1u32);
                s_atomic.index((wave_offset + d1) as usize).atomic_addi(1u32);
                s_atomic.index((wave_offset + d2) as usize).atomic_addi(1u32);
                s_atomic.index((wave_offset + d3) as usize).atomic_addi(1u32);
                i += block_dim;
            }
        }

        // CUDA: if (blockIdx.x == gridDim.x - 1) — last block, partial
        // Last block uses scalar access since size may not be multiple of 4
        if block_id == grid_dim - 1 {
            // Scalar fallback for partial last block
            let scalar_start = block_id * PART_SIZE;
            let mut i = tid + scalar_start;
            while i < size {
                // Reinterpret: element i in u32 space = sort[i/4].data[i%4]
                let vec_idx = i / 4;
                let elem_idx = i % 4;
                let keys = sort[vec_idx as usize];
                let k = keys.data();
                let key = k[elem_idx as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                s_atomic
                    .index((wave_offset + digit) as usize)
                    .atomic_addi(1u32);
                i += block_dim;
            }
        }
    }
    sync_threads();

    // CUDA: reduce two sub-hists, write passHist, begin warp scan
    // passHist layout: passHist[digit * padded_thread_blocks + blockId].
    // Use (grid_dim, padded_thread_blocks) to set TD for the grid dimension,
    // ensuring stride between digits = padded_thread_blocks (not grid_dim).
    {
        let mut ph_chunk = chunk_mut(
            pass_hist,
            reshape_map!([RADIX / block_dim] | [block_dim, (grid_dim, padded_thread_blocks)] => layout: [t1, t0, i0]),
        );
        let mut k = 0u32;
        let mut i = tid;
        while i < RADIX {
            // CUDA: s_globalHist[i] += s_globalHist[i + RADIX];
            let combined = *smem[i as usize] + *smem[(i + RADIX) as usize];

            // CUDA: passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];
            ph_chunk[k] = combined;

            // CUDA: s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
            let scanned = crate::utils::inclusive_warp_scan_circular_shift(combined);
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(i as usize).atomic_assign(scanned);

            k += 1;
            i += block_dim;
        }
    }
    sync_threads();

    // Workaround: replace ActiveExclusiveWarpScan (parallel warp scan on 8 threads)
    // with sequential scan on thread 0. SeGuRu JIT hangs with warp shuffle inside
    // narrow conditional (tid < 8). Impact: minor (only 8 values).
    if tid == 0 {
        let n_warps = RADIX >> LANE_LOG; // 8
        let mut running = 0u32;
        let mut w = 0u32;
        while w < n_warps {
            let idx = (w << LANE_LOG) as usize;
            let val = *smem[idx];
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(idx).atomic_assign(running);
            running += val;
            w += 1;
        }
    }
    sync_threads();

    // CUDA: atomicAdd(&globalHist[i + (radixShift << 5)],
    //           s_globalHist[i] + (getLaneId() ? __shfl_sync(..., s_globalHist[i-1], 1) : 0));
    // globalHist accumulates across blocks → true atomic add on global memory.
    {
        let g_hist = gpu::sync::Atomic::new(global_hist);
        let mut i = tid;
        while i < RADIX {
            let scan_val = *smem[i as usize];

            // CUDA: (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i-1], 1) : 0)
            // Reads predecessor's value then broadcasts from lane 1.
            let prev_val = if i > 0 { *smem[(i - 1) as usize] } else { 0u32 };
            let (shuffled, _) = gpu::shuffle!(idx, prev_val, 1u32, 32);

            let final_val = if lane_id != 0 {
                scan_val + shuffled
            } else {
                scan_val
            };

            // CUDA: globalHist[i + (radixShift << 5)]
            g_hist
                .index((i + (radix_shift << 5)) as usize)
                .atomic_addi(final_val);

            i += block_dim;
        }
    }
}
