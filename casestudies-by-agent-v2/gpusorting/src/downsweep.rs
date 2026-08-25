//! Pass 3 of each radix digit: rank the keys of a partition and scatter them.
//!
//! Mirrors `DeviceRadixSort::DownsweepKeysOnly`. Structure per CTA:
//!
//! 1. load `BIN_KEYS_PER_THREAD` keys per thread into registers;
//! 2. warp-level multi-split (WLMS): each warp ranks its own keys against a private
//!    256-bin shared histogram using `ballot`;
//! 3. exclusive scan of the 16 per-warp histograms, then add the block's global base
//!    (from `global_hist` and the scanned `pass_hist`);
//! 4. scatter through shared memory so the final global write is coalesced.
//!
//! Because the host pads the key array to a whole number of `BIN_PART_SIZE`
//! partitions with `u32::MAX`, this kernel needs no ragged-tail predicate.

use crunchy::unroll;
use gpu::prelude::*;

use crate::utils::{
    exclusive_warp_scan, inclusive_warp_scan_circular_shift, lane_mask_lt, lowest_set_bit,
    LANE_COUNT, LANE_LOG,
};
use crate::{
    BIN_HISTS_SIZE, BIN_KEYS_PER_THREAD, BIN_PART_SIZE, BIN_SUB_PART_SIZE, DOWNSWEEP_THREADS,
    RADIX, RADIX_LOG, RADIX_MASK, SMEM_WORDS,
};

const KPT: usize = BIN_KEYS_PER_THREAD as usize;
const HIST_CLEAR_PER_THREAD: usize = (BIN_HISTS_SIZE / DOWNSWEEP_THREADS) as usize;

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_downsweep(
    sort: &[u32],
    alt: &mut [u32],
    global_hist: &[u32],
    pass_hist: &[u32],
    radix_shift: u32,
    padded_thread_blocks: u32,
) {
    assert!(Config::BDIM_X == DOWNSWEEP_THREADS);
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();
    let lane = lane_id();
    let warp = tid >> LANE_LOG;

    // s_hist  = smem[0 .. BIN_HISTS_SIZE]   (16 warps x 256 bins), later reused as
    // s_keys  = smem[0 .. BIN_PART_SIZE]    (the local scatter buffer)
    // s_base  = smem[BIN_PART_SIZE .. +RADIX]
    let smem = smem_alloc.alloc::<u32>(SMEM_WORDS as usize);

    {
        let mut z = smem.chunk_mut(MapLinear::new(1));
        unroll! {
            for k in 0..8 {
                if k < HIST_CLEAR_PER_THREAD {
                    z[k] = 0u32;
                }
            }
        }
    }
    sync_threads();

    // ---- 1. Load this thread's keys ------------------------------------------------
    // Lane-major within a warp's sub-partition, so each of the 15 loads is a single
    // 128-byte transaction per warp.
    let mut keys = [0u32; KPT];
    {
        let start = bid * BIN_PART_SIZE + warp * BIN_SUB_PART_SIZE + lane;
        unroll! {
            for i in 0..8 {
                keys[i] = sort[(start + (i as u32) * LANE_COUNT) as usize];
            }
        }
    }

    // ---- 2. Warp-level multi-split -------------------------------------------------
    let mut offsets = [0u32; KPT];
    {
        let hist = gpu::sync::SharedAtomic::new(&mut *smem);
        let wbase = warp << RADIX_LOG;
        let lt = lane_mask_lt();
        unroll! {
            for i in 0..8 {
                let key = keys[i];
                let digit = (key >> radix_shift) & RADIX_MASK;

                // Peel the digit bit by bit: after 8 ballots, `flags` holds exactly the
                // lanes of this warp whose key shares this key's digit.
                let mut flags = 0xFFFF_FFFFu32;
                unroll! {
                    for b in 0..8 {
                        let set = (key >> ((b as u32) + radix_shift)) & 1 != 0;
                        let ballot = ballot_sync(0xFFFF_FFFF, set);
                        let inv = if set { 0u32 } else { 0xFFFF_FFFFu32 };
                        flags &= inv ^ ballot;
                    }
                }

                let rank = (flags & lt).count_ones();
                // The lowest matching lane reserves space for the whole match group and
                // broadcasts the reservation. `lowest_set_bit` is a hardware ffs.
                let mut reserved = 0u32;
                if rank == 0 {
                    reserved = hist
                        .index((wbase + digit) as usize)
                        .atomic_addi(flags.count_ones());
                }
                let leader = lowest_set_bit(flags);
                let (group_base, _) = gpu::shuffle!(idx, reserved, leader, 32u32);
                offsets[i] = group_base + rank;
            }
        }
    }
    sync_threads();

    // ---- 3. Exclusive scan across the per-warp histograms ---------------------------
    // Threads 0..255 (warps 0..7) each own one digit. The guard is warp-uniform, so
    // the warp scan below still runs on complete warps.
    if tid < RADIX {
        let mut running = *smem[tid as usize];
        let mut j = tid + RADIX;
        while j < BIN_HISTS_SIZE {
            let v = *smem[j as usize];
            running += v;
            let w = gpu::sync::SharedAtomic::new(&mut *smem);
            w.index(j as usize).atomic_assign(running - v);
            j += RADIX;
        }
        let scanned = inclusive_warp_scan_circular_shift(running);
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index(tid as usize).atomic_assign(scanned);
    }
    sync_threads();

    if tid < 32 {
        let groups = RADIX >> LANE_LOG; // 8
        let v = if tid < groups {
            *smem[(tid << LANE_LOG) as usize]
        } else {
            0u32
        };
        let s = exclusive_warp_scan(v);
        if tid < groups {
            let w = gpu::sync::SharedAtomic::new(&mut *smem);
            w.index((tid << LANE_LOG) as usize).atomic_assign(s);
        }
    }
    sync_threads();

    if tid < RADIX {
        let mine = *smem[tid as usize];
        let prev = if tid > 0 { *smem[(tid - 1) as usize] } else { 0u32 };
        let (group_base, _) = gpu::shuffle!(idx, prev, 1u32, 32);
        let total = if lane != 0 { mine + group_base } else { mine };
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index(tid as usize).atomic_assign(total);
    }
    sync_threads();

    // Fold the digit-wide and warp-wide prefixes into each key's local rank.
    {
        let wbase = warp << RADIX_LOG;
        unroll! {
            for i in 0..8 {
                let digit = (keys[i] >> radix_shift) & RADIX_MASK;
                let digit_prefix = *smem[digit as usize];
                if warp != 0 {
                    offsets[i] += *smem[(wbase + digit) as usize] + digit_prefix;
                } else {
                    offsets[i] += digit_prefix;
                }
            }
        }
    }

    // Where this block's run of each digit starts in the output.
    if tid < RADIX {
        let g = global_hist[(tid + (radix_shift << 5)) as usize];
        let p = pass_hist[(tid * padded_thread_blocks + bid) as usize];
        let local = *smem[tid as usize];
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index((BIN_PART_SIZE + tid) as usize)
            .atomic_assign(g + p - local);
    }
    sync_threads();

    // ---- 4a. Scatter into shared memory (the histograms are dead by now) ------------
    {
        let s = gpu::sync::SharedAtomic::new(&mut *smem);
        unroll! {
            for i in 0..8 {
                s.index(offsets[i] as usize).atomic_assign(keys[i]);
            }
        }
    }
    sync_threads();

    // ---- 4b. Scatter out to global memory ------------------------------------------
    // Consecutive threads read consecutive shared slots, so keys sharing a digit land
    // in consecutive global addresses: the writes are coalesced runs. The index is
    // data dependent, so SeGuRu cannot prove disjointness statically and the store
    // goes through `Atomic`.
    {
        let out = gpu::sync::Atomic::new(alt);
        unroll! {
            for k in 0..8 {
                let i = tid + (k as u32) * DOWNSWEEP_THREADS;
                let key = *smem[i as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                let base = *smem[(BIN_PART_SIZE + digit) as usize];
                out.index((base + i) as usize).atomic_assign(key);
            }
        }
    }
}
