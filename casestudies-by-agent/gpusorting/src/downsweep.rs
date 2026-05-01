use gpu::prelude::*;
use gpu::MapLinear;

use crate::{
    BIN_HISTS_SIZE, BIN_KEYS_PER_THREAD, BIN_PART_SIZE, BIN_SUB_PART_SIZE, LANE_COUNT, LANE_LOG,
    RADIX, RADIX_LOG, RADIX_MASK,
};

// ============================================================================
// CUDA reference: DeviceRadixSort::DownsweepKeysOnly
// See cuda-ref/GPUSortingCUDA/Sort/DeviceRadixSort.cu lines 157–284
//
// Phases:
//   1. Load keys (fill with 0xFFFFFFFF for last-block out-of-bound)
//   2. WLMS ranking: per-warp histogram + per-key offset
//   3. Exclusive prefix sum across warp histograms → global scatter base
//   4. Scatter: keys → shared (by offset) → global (by base + local index)
//
// CUDA constants:
//   WARP_INDEX          = threadIdx.x / LANE_COUNT
//   BIN_PART_START      = blockIdx.x * BIN_PART_SIZE
//   BIN_SUB_PART_START  = WARP_INDEX * BIN_SUB_PART_SIZE
//
// Shared memory:
//   s_warpHistograms[BIN_PART_SIZE]   — reused for: warp hists, local scatter
//   s_localHistogram[RADIX]           — digit-to-global-offset mapping
//
// PORTING NOTES:
//   - __ballot_sync replaced with shuffle-based digit matching per user request
//   - __popc replaced with manual popcount via shuffle
//   - volatile shared memory access → SharedAtomic for warp-level atomicAdd
//   - Scatter to alt[] is data-dependent → Atomic::new is correct (not chunking-style)
// ============================================================================

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_downsweep(
    sort: &[u32],
    alt: &mut [u32],
    global_hist: &[u32],
    pass_hist: &[u32],
    size: u32,
    radix_shift: u32,
    padded_thread_blocks: u32,
) {
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let lane_id = lane_id();

    // CUDA: WARP_INDEX = threadIdx.x / LANE_COUNT
    let warp_id = tid >> LANE_LOG;
    // CUDA: BIN_PART_START = blockIdx.x * BIN_PART_SIZE
    let bin_part_start = bid * BIN_PART_SIZE;
    // CUDA: BIN_SUB_PART_START = WARP_INDEX * BIN_SUB_PART_SIZE
    let bin_sub_part_start = warp_id * BIN_SUB_PART_SIZE;

    // CUDA: __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
    //       __shared__ uint32_t s_localHistogram[RADIX];
    // Total smem: BIN_PART_SIZE + RADIX = 7680 + 256 = 7936 u32s
    let smem = smem_alloc.alloc::<u32>((BIN_PART_SIZE + RADIX) as usize);

    // Alias: s_warpHist is smem[0..BIN_HISTS_SIZE], s_localHist is smem[BIN_PART_SIZE..+RADIX]

    // ---- Phase 0: Clear warp histograms ----
    // CUDA: for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
    //           s_warpHistograms[i] = 0;
    {
        // BIN_HISTS_SIZE=4096, block_dim=512 → 8 elements per thread
        let elems_per_thread = BIN_HISTS_SIZE / block_dim;
        let mut smem_chunk = smem.chunk_mut(MapLinear::new(elems_per_thread as usize));
        let mut k = 0u32;
        while k < elems_per_thread {
            smem_chunk[k as usize] = 0u32;
            k += 1;
        }
    }
    sync_threads();

    // ---- Phase 1: Load keys ----
    // CUDA: keys[BIN_KEYS_PER_THREAD], loaded with stride LANE_COUNT from per-warp partition.
    // CUDA: for (i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
    //            i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
    //           keys[i] = sort[t];                        // non-last block
    //           keys[i] = t < size ? sort[t] : 0xffffffff; // last block
    let mut keys = [0u32; 15]; // BIN_KEYS_PER_THREAD = 15
    {
        let start = lane_id + bin_sub_part_start + bin_part_start;
        let mut i = 0u32;
        while i < BIN_KEYS_PER_THREAD {
            let t = start + i * LANE_COUNT;
            if bid < grid_dim - 1 {
                keys[i as usize] = sort[t as usize];
            } else {
                keys[i as usize] = if t < size { sort[t as usize] } else { 0xFFFFFFFF };
            }
            i += 1;
        }
    }

    // ---- Phase 2: WLMS (Warp-Level Matching Scan) ----
    // Workaround: replace __ballot_sync (1 hardware instruction) with 32 shuffle
    // broadcasts per ballot call. Cost: 8 bits × 32 lanes = 256 shuffles/key vs 8 ballots/key.
    // This is the dominant performance bottleneck (~32× instruction bloat vs CUDA).
    let mut offsets = [0u16; 15]; // BIN_KEYS_PER_THREAD = 15
    {
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        // CUDA: volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];
        let warp_hist_base = warp_id << RADIX_LOG;

        let mut i = 0u32;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;

            // Workaround: replace __ballot_sync with shuffle-based ballot emulation.
            // CUDA: warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
            // SeGuRu: iterate all 32 lanes, broadcast each bit via shuffle, build mask.
            let mut warp_flags = 0xFFFFFFFFu32;
            let mut k = 0u32;
            while k < RADIX_LOG {
                let my_bit = (keys[i as usize] >> (k + radix_shift)) & 1;
                let mut ballot = 0u32;
                let mut lane = 0u32;
                while lane < 32 {
                    let (other_bit, _) = gpu::shuffle!(idx, my_bit, lane, 32u32);
                    if other_bit != 0 {
                        ballot |= 1u32 << lane;
                    }
                    lane += 1;
                }
                let mask = if my_bit != 0 { 0u32 } else { 0xFFFFFFFFu32 };
                warp_flags &= mask ^ ballot;
                k += 1;
            }

            // CUDA: bits = __popc(warpFlags & getLaneMaskLt())
            // No workaround needed: count_ones() lowers to hardware ctpop.
            let lane_mask_lt = if lane_id > 0 {
                (1u32 << lane_id) - 1
            } else {
                0u32
            };
            let bits = (warp_flags & lane_mask_lt).count_ones();

            // CUDA: if (bits == 0) preIncrementVal = atomicAdd(&s_warpHist[digit], __popc(warpFlags))
            let total_count = warp_flags.count_ones();
            let mut pre_inc_val = 0u32;
            if bits == 0 {
                pre_inc_val = s_atom
                    .index((warp_hist_base + digit) as usize)
                    .atomic_addi(total_count);
            }

            // Workaround: replace __ffs (1 hardware instruction) with manual loop.
            // cttz/trailing_zeros not supported in SeGuRu codegen.
            let mut first_lane = 0u32;
            {
                let mut tmp = warp_flags;
                let mut pos = 0u32;
                while pos < 32 {
                    if tmp & 1 != 0 {
                        first_lane = pos;
                        break;
                    }
                    tmp >>= 1;
                    pos += 1;
                }
            }
            let (shared_pre_inc, _) = gpu::shuffle!(idx, pre_inc_val, first_lane, 32u32);
            offsets[i as usize] = (shared_pre_inc + bits) as u16;

            i += 1;
        }
    }
    sync_threads();

    // ---- Phase 3: Exclusive prefix sum across warp histograms ----
    // CUDA: if (threadIdx.x < RADIX) {
    //           reduction = s_warpHistograms[threadIdx.x];
    //           for (i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
    //               reduction += s_warpHistograms[i];
    //               s_warpHistograms[i] = reduction - s_warpHistograms[i];
    //           }
    //           s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    //       }
    if tid < RADIX {
        let mut reduction = *smem[tid as usize];
        let mut j = tid + RADIX;
        while j < BIN_HISTS_SIZE {
            let val = *smem[j as usize];
            reduction += val;
            // Exclusive prefix: s_warpHistograms[j] = running_total_before_this_warp
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(j as usize).atomic_assign(reduction - val);
            j += RADIX;
        }

        // Begin exclusive prefix sum across the reductions
        let scanned = crate::utils::inclusive_warp_scan_circular_shift(reduction);
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atom.index(tid as usize).atomic_assign(scanned);
    }
    sync_threads();

    // Workaround: replace ActiveExclusiveWarpScan (parallel) with sequential scan
    // on thread 0. SeGuRu JIT hangs with warp shuffle inside narrow conditional.
    if tid == 0 {
        let n_warps = RADIX >> LANE_LOG; // 8
        let mut running = 0u32;
        let mut w = 0u32;
        while w < n_warps {
            let idx = (w << LANE_LOG) as usize;
            let val = *smem[idx];
            let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atom.index(idx).atomic_assign(running);
            running += val;
            w += 1;
        }
    }
    sync_threads();

    // CUDA: if (threadIdx.x < RADIX && getLaneId())
    //           s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x-1], 1);
    if tid < RADIX {
        let scan_val = *smem[tid as usize];
        let prev_val = if tid > 0 {
            *smem[(tid - 1) as usize]
        } else {
            0u32
        };
        let (shuffled, _) = gpu::shuffle!(idx, prev_val, 1u32, 32);
        let final_val = if lane_id != 0 {
            scan_val + shuffled
        } else {
            scan_val
        };
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atom.index(tid as usize).atomic_assign(final_val);
    }
    sync_threads();

    // ---- Phase 3b: Update offsets with warp histogram prefix + total prefix ----
    // CUDA: if (WARP_INDEX)
    //           offsets[i] += s_warpHist[digit] + s_warpHistograms[digit];
    //       else
    //           offsets[i] += s_warpHistograms[digit];
    {
        let warp_hist_base = warp_id << RADIX_LOG;
        let mut i = 0u32;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
            let total_prefix = *smem[digit as usize];
            if warp_id != 0 {
                let warp_prefix = *smem[(warp_hist_base + digit) as usize];
                offsets[i as usize] += (warp_prefix + total_prefix) as u16;
            } else {
                offsets[i as usize] += total_prefix as u16;
            }
            i += 1;
        }
    }

    // ---- Phase 3c: Load threadblock global scatter base ----
    // CUDA: if (threadIdx.x < RADIX)
    //           s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
    //               passHist[threadIdx.x * gridDim.x + blockIdx.x] -
    //               s_warpHistograms[threadIdx.x];
    if tid < RADIX {
        let global_base = global_hist[(tid + (radix_shift << 5)) as usize];
        let pass_base = pass_hist[(tid * padded_thread_blocks + bid) as usize];
        let local_reduction = *smem[tid as usize]; // s_warpHistograms[threadIdx.x] = inclusive count total
        let base = global_base + pass_base - local_reduction;

        // Write to s_localHistogram at smem[BIN_PART_SIZE + tid]
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atom
            .index((BIN_PART_SIZE + tid) as usize)
            .atomic_assign(base);
    }
    sync_threads();

    // ---- Phase 4a: Scatter keys into shared memory ----
    // CUDA: for (i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    //           s_warpHistograms[offsets[i]] = keys[i];
    {
        let s_atom = gpu::sync::SharedAtomic::new(&mut *smem);
        let mut i = 0u32;
        while i < BIN_KEYS_PER_THREAD {
            s_atom
                .index(offsets[i as usize] as usize)
                .atomic_assign(keys[i as usize]);
            i += 1;
        }
    }
    sync_threads();

    // ---- Phase 4b: Scatter from shared memory to global memory ----
    // Workaround: replace plain global store with Atomic::atomic_assign.
    // CUDA uses `alt[idx] = key` (plain store, no conflicts since each key gets a unique position).
    // SeGuRu requires Atomic wrapper for data-dependent (scatter) writes where the compiler
    // cannot statically prove non-overlapping indices. This adds atomic exchange overhead.
    {
        let alt_out = gpu::sync::Atomic::new(alt);
        let scatter_count = if bid < grid_dim - 1 {
            BIN_PART_SIZE
        } else {
            size - bin_part_start
        };
        let mut i = tid;
        while i < scatter_count {
            let key = *smem[i as usize];
            let digit = (key >> radix_shift) & RADIX_MASK;
            // s_localHistogram[digit] = smem[BIN_PART_SIZE + digit]
            let base = *smem[(BIN_PART_SIZE + digit) as usize];
            alt_out
                .index((base + i) as usize)
                .atomic_assign(key);
            i += block_dim;
        }
    }
}
