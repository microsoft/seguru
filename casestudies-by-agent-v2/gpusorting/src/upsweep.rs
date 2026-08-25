//! Pass 1 of each radix digit: per-block digit histograms.
//!
//! Mirrors `DeviceRadixSort::Upsweep` from the GPUSorting CUDA reference, but the
//! host pads the key array up to a whole number of `PART_SIZE` partitions (filling
//! with `u32::MAX`), so this kernel has **no ragged-tail branch at all** and every
//! load is a full `U32_4` vector load.
//!
//! Outputs:
//! * `pass_hist[digit * padded_thread_blocks + block]` — this block's count of `digit`.
//! * `global_hist[digit + (radix_shift << 5)]` — grid-wide *exclusive* prefix over
//!   digits, accumulated with global atomics.

use crunchy::unroll;
use gpu::prelude::*;

use crate::utils::{inclusive_warp_scan_circular_shift, LANE_LOG};
use crate::{PART_SIZE, RADIX, RADIX_MASK, UPSWEEP_THREADS};

const VEC_PART_SIZE: u32 = PART_SIZE / 4;
/// Two independent sub-histograms halve shared-atomic contention.
const SUB_HISTS: u32 = 2;

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_upsweep(
    sort: &[U32_4],
    global_hist: &mut [u32],
    pass_hist: &mut [u32],
    radix_shift: u32,
    padded_thread_blocks: u32,
) {
    assert!(Config::BDIM_X == UPSWEEP_THREADS);
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();
    let lane = lane_id();

    let smem = smem_alloc.alloc::<u32>((RADIX * SUB_HISTS) as usize);

    // Zero both sub-histograms; MapLinear(1) gives thread `tid` the strided
    // elements tid, tid+BDIM, tid+2*BDIM, ... which is fully coalesced.
    {
        let mut z = smem.chunk_mut(MapLinear::new(1));
        unroll! {
            for k in 0..4 {
                z[k] = 0u32;
            }
        }
    }
    sync_threads();

    // Histogram the block's partition. Threads 0..63 accumulate into sub-histogram
    // 0 and threads 64..127 into sub-histogram 1.
    {
        let hist = gpu::sync::SharedAtomic::new(&mut *smem);
        let wave = (tid / 64) * RADIX;
        let start = bid * VEC_PART_SIZE;
        let mut i = start + tid;
        let end = start + VEC_PART_SIZE;
        while i < end {
            let k = sort[i as usize];
            let v = k.data();
            unroll! {
                for j in 0..4 {
                    let d = (v[j] >> radix_shift) & RADIX_MASK;
                    hist.index((wave + d) as usize).atomic_addi(1u32);
                }
            }
            i += UPSWEEP_THREADS;
        }
    }
    sync_threads();

    // Fold the two sub-histograms together, publish the per-block counts, and
    // start the digit-wise scan that `global_hist` needs.
    {
        let mut ph = chunk_mut(
            pass_hist,
            reshape_map!([2u32] | [UPSWEEP_THREADS, (grid_dim::<DimX>(), padded_thread_blocks)] => layout: [t1, t0, i0]),
        );
        unroll! {
            for k in 0..2 {
                let i = tid + (k as u32) * UPSWEEP_THREADS;
                let total = *smem[i as usize] + *smem[(i + RADIX) as usize];
                ph[k as u32] = total;
                let scanned = inclusive_warp_scan_circular_shift(total);
                let w = gpu::sync::SharedAtomic::new(&mut *smem);
                w.index(i as usize).atomic_assign(scanned);
            }
        }
    }
    sync_threads();

    // Exclusive scan of the eight 32-digit group totals, which the circular shift
    // above parked in lanes 0 of each group. The guard is `tid < 32`, i.e. exactly
    // warp 0, so the shuffles inside the scan still see a full warp; lanes with no
    // group feed in the identity element.
    if tid < 32 {
        let groups = RADIX >> LANE_LOG; // 8
        let v = if tid < groups {
            *smem[(tid << LANE_LOG) as usize]
        } else {
            0u32
        };
        let s = crate::utils::exclusive_warp_scan(v);
        if tid < groups {
            let w = gpu::sync::SharedAtomic::new(&mut *smem);
            w.index((tid << LANE_LOG) as usize).atomic_assign(s);
        }
    }
    sync_threads();

    // Accumulate this block's contribution into the grid-wide digit offsets.
    {
        let g = gpu::sync::Atomic::new(global_hist);
        let base = radix_shift << 5;
        unroll! {
            for k in 0..2 {
                let i = tid + (k as u32) * UPSWEEP_THREADS;
                let mine = *smem[i as usize];
                // Lane 0 of each 32-digit group already holds the group's exclusive
                // prefix; every other lane needs it broadcast from lane 1's read of
                // its predecessor.
                let prev = if i > 0 { *smem[(i - 1) as usize] } else { 0u32 };
                let (group_base, _) = gpu::shuffle!(idx, prev, 1u32, 32);
                let val = if lane != 0 { mine + group_base } else { mine };
                g.index((base + i) as usize).atomic_addi(val);
            }
        }
    }
}

const RADIX_OVER_THREADS: usize = (RADIX / UPSWEEP_THREADS) as usize;
const _: () = assert!(RADIX_OVER_THREADS == 2);
