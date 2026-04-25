use gpu::prelude::*;

use crate::{
    BIN_HISTS_SIZE, BIN_KEYS_PER_THREAD, BIN_PART_SIZE, BIN_SUB_PART_SIZE, LANE_LOG, RADIX,
    RADIX_LOG, RADIX_MASK,
};

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_downsweep(
    sort: &[u32],
    alt: &mut [u32],
    global_hist: &[u32],
    pass_hist: &[u32],
    size: u32,
    radix_shift: u32,
) {
    let tid = thread_id::<DimX>();
    let block_id = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let lane_id = lane_id();
    let warp_idx = tid >> LANE_LOG;

    // Shared memory: 7680 entries (reused for warp histograms then key staging) + 256 for local hist
    let s_warp_hists = smem_alloc.alloc::<u32>(BIN_PART_SIZE as usize);
    let s_local_hist = smem_alloc.alloc::<u32>(RADIX as usize);

    let bin_part_start = block_id * BIN_PART_SIZE;
    let bin_sub_part_start = warp_idx * BIN_SUB_PART_SIZE;
    let warp_hist_offset = warp_idx << RADIX_LOG;

    // Phase 1: Clear per-warp histograms (first BIN_HISTS_SIZE = 4096 entries)
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        let mut i = tid;
        while i < BIN_HISTS_SIZE {
            s_atomic.index(i as usize).atomic_assign(0u32);
            i += block_dim;
        }
    }

    // Phase 2: Load keys (BIN_KEYS_PER_THREAD = 15 per thread)
    let mut keys = [0u32; BIN_KEYS_PER_THREAD as usize];

    if block_id < grid_dim - 1 {
        let mut i: u32 = 0;
        let mut t = lane_id + bin_sub_part_start + bin_part_start;
        while i < BIN_KEYS_PER_THREAD {
            keys[i as usize] = sort[t as usize];
            i += 1;
            t += 32; // LANE_COUNT
        }
    }

    if block_id == grid_dim - 1 {
        let mut i: u32 = 0;
        let mut t = lane_id + bin_sub_part_start + bin_part_start;
        while i < BIN_KEYS_PER_THREAD {
            keys[i as usize] = if t < size { sort[t as usize] } else { 0xFFFFFFFFu32 };
            i += 1;
            t += 32;
        }
    }

    sync_threads();

    // Phase 3: Ranking — atomicAdd on per-warp histogram bin, old value = local rank
    let mut offsets = [0u32; BIN_KEYS_PER_THREAD as usize];
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
            let rank = s_atomic
                .index((warp_hist_offset + digit) as usize)
                .atomic_addi(1u32);
            offsets[i as usize] = rank;
            i += 1;
        }
    }
    sync_threads();

    // Phase 4: Exclusive prefix sum across warp histograms
    // Step 4a: Reduce across warps per digit, converting to exclusive offsets
    if tid < RADIX {
        let mut reduction = *s_warp_hists[tid as usize];
        let mut i = tid + RADIX;
        while i < BIN_HISTS_SIZE {
            let val = *s_warp_hists[i as usize];
            reduction += val;
            let new_val = reduction - val;
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
            s_atomic.index(i as usize).atomic_assign(new_val);
            i += RADIX;
        }
        // Inclusive warp scan with circular shift on reduction
        let scanned = crate::utils::inclusive_warp_scan_circular_shift(reduction);
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        s_atomic.index(tid as usize).atomic_assign(scanned);
    }
    sync_threads();

    // Step 4b: Exclusive scan of warp-level sums (8 groups of 32)
    if tid < (RADIX >> LANE_LOG) {
        let idx = (tid << LANE_LOG) as usize;
        let val = *s_warp_hists[idx];
        let exc = crate::utils::exclusive_warp_scan(val);
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        s_atomic.index(idx).atomic_assign(exc);
    }
    sync_threads();

    // Step 4c: Combine inter-warp scan with intra-warp values using shuffle
    if tid < RADIX && lane_id != 0 {
        let prev_val = *s_warp_hists[(tid - 1) as usize];
        let (shuffled, _) = gpu::shuffle!(idx, prev_val, 1u32, 32u32);
        let cur_val = *s_warp_hists[tid as usize];
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        s_atomic
            .index(tid as usize)
            .atomic_assign(cur_val + shuffled);
    }
    sync_threads();

    // Phase 5: Update offsets with warp and block prefixes
    if warp_idx != 0 {
        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
            let warp_prefix = *s_warp_hists[(warp_hist_offset + digit) as usize];
            let block_prefix = *s_warp_hists[digit as usize];
            offsets[i as usize] += warp_prefix + block_prefix;
            i += 1;
        }
    } else {
        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
            offsets[i as usize] += *s_warp_hists[digit as usize];
            i += 1;
        }
    }

    // Phase 6: Load global offsets into s_local_hist
    if tid < RADIX {
        let global_offset = global_hist[(tid + (radix_shift << 5)) as usize];
        let pass_offset = pass_hist[(tid * grid_dim + block_id) as usize];
        let block_hist = *s_warp_hists[tid as usize];
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_local_hist);
        s_atomic
            .index(tid as usize)
            .atomic_assign(global_offset + pass_offset - block_hist);
    }
    sync_threads();

    // Phase 7: Scatter keys into shared memory using offsets
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            s_atomic
                .index(offsets[i as usize] as usize)
                .atomic_assign(keys[i as usize]);
            i += 1;
        }
    }
    sync_threads();

    // Phase 8: Scatter from shared memory to global output
    {
        let g_atomic = gpu::sync::Atomic::new(alt);

        if block_id < grid_dim - 1 {
            let mut i = tid;
            while i < BIN_PART_SIZE {
                let key = *s_warp_hists[i as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                let global_pos = *s_local_hist[digit as usize] + i;
                g_atomic
                    .index(global_pos as usize)
                    .atomic_assign(key);
                i += block_dim;
            }
        }

        if block_id == grid_dim - 1 {
            let final_part_size = size - bin_part_start;
            let mut i = tid;
            while i < final_part_size {
                let key = *s_warp_hists[i as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                let global_pos = *s_local_hist[digit as usize] + i;
                g_atomic
                    .index(global_pos as usize)
                    .atomic_assign(key);
                i += block_dim;
            }
        }
    }
}
