use gpu::prelude::*;

use crate::{LANE_LOG, PART_SIZE, RADIX, RADIX_MASK};

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_upsweep(
    sort: &[u32],
    global_hist: &mut [u32],
    pass_hist: &mut [u32],
    size: u32,
    radix_shift: u32,
) {
    let tid = thread_id::<DimX>();
    let block_id = block_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let lane_id = lane_id();

    // Allocate shared memory: RADIX * 2 = 512 entries
    // Two 256-bin histograms, one per wave (wave = group of 64 threads)
    let smem = smem_alloc.alloc::<u32>(512);

    // Clear shared memory
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        let mut i = tid;
        while i < RADIX * 2 {
            s_atomic.index(i as usize).atomic_assign(0u32);
            i += block_dim;
        }
    }
    sync_threads();

    // Phase 1: Histogram using shared memory atomics
    // Each wave (64 threads) has its own 256-bin histogram
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        let wave_offset = (tid / 64) * RADIX;

        if block_id < grid_dim - 1 {
            // Non-last block: process PART_SIZE keys
            let part_start = block_id * PART_SIZE;
            let part_end = part_start + PART_SIZE;
            let mut i = tid + part_start;
            while i < part_end {
                let t = sort[i as usize];
                let digit = (t >> radix_shift) & RADIX_MASK;
                s_atomic
                    .index((wave_offset + digit) as usize)
                    .atomic_addi(1u32);
                i += block_dim;
            }
        }

        if block_id == grid_dim - 1 {
            // Last block: process remaining keys up to size
            let mut i = tid + block_id * PART_SIZE;
            while i < size {
                let t = sort[i as usize];
                let digit = (t >> radix_shift) & RADIX_MASK;
                s_atomic
                    .index((wave_offset + digit) as usize)
                    .atomic_addi(1u32);
                i += block_dim;
            }
        }
    }
    sync_threads();

    // Phase 2: Reduce two wave histograms, write pass_hist, warp scan
    // Step 2a: Read both histograms, combine, write to pass_hist and smem[i]
    {
        let mut ph = chunk_mut(pass_hist, MapLinear::new(1));
        let mut i = tid;
        while i < RADIX {
            let val0 = *smem[i as usize];
            let val1 = *smem[(i + RADIX) as usize];
            let combined = val0 + val1;
            ph[(i * grid_dim + block_id) as usize] = combined;
            // Store combined back using atomic_assign
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(i as usize).atomic_assign(combined);
            i += block_dim;
        }
    }
    sync_threads();

    // Step 2b: Inclusive warp scan with circular shift on combined histogram
    {
        let mut i = tid;
        while i < RADIX {
            let val = *smem[i as usize];
            let scanned = crate::utils::inclusive_warp_scan_circular_shift(val);
            let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
            s_atomic.index(i as usize).atomic_assign(scanned);
            i += block_dim;
        }
    }
    sync_threads();

    // Phase 3: Exclusive scan of warp-level sums
    if tid < (RADIX >> LANE_LOG) {
        let idx = (tid << LANE_LOG) as usize;
        let val = *smem[idx];
        let exc = crate::utils::exclusive_warp_scan(val);
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *smem);
        s_atomic.index(idx).atomic_assign(exc);
    }
    sync_threads();

    // Phase 4: Final atomic add to global histogram
    {
        let g_hist = gpu::sync::Atomic::new(global_hist);
        let mut i = tid;
        while i < RADIX {
            let scan_val = *smem[i as usize];
            // Get previous lane's value via shuffle
            let prev_idx = i.wrapping_sub(1);
            let prev_val = *smem[prev_idx as usize];
            let (shuffled, _) = gpu::shuffle!(idx, prev_val, 1u32, 32);

            let final_val = if lane_id != 0 {
                scan_val + shuffled
            } else {
                scan_val
            };

            let hist_idx = i + (radix_shift << 5);
            g_hist.index(hist_idx as usize).atomic_addi(final_val);

            i += block_dim;
        }
    }
}