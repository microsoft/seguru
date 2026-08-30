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
//!
//! # Comparison with the CUDA original
//!
//! Each section below carries the corresponding lines of
//! `DeviceRadixSort::DownsweepKeysOnly` from
//! <https://github.com/b0nes164/GPUSorting> (`GPUSortingCUDA/Sort/DeviceRadixSort.cu`),
//! quoted under its MIT licence, Copyright (c) 2024 Thomas Smith. They are here so
//! the safe-Rust port can be read against the hand-written CUDA it mirrors: the
//! algorithm is intended to be identical instruction-for-instruction, and where the
//! Rust has to say something different, the comment says why.
//!
//! Two differences recur and are not repeated at every site:
//!
//! * **Shared memory is one allocation.** CUDA declares `s_warpHistograms` and
//!   `s_localHistogram` separately and aliases the first with the scatter buffer;
//!   SeGuRu takes a single `smem_alloc.alloc::<u32>(SMEM_WORDS)` and gives the
//!   regions names in comments, because overlapping typed allocations cannot be
//!   expressed safely.
//! * **Data-dependent shared writes go through `Atomic`.** CUDA writes
//!   `s_warpHistograms[i] = x` with a plain store wherever `i` is a runtime value.
//!   SeGuRu's checker cannot prove two lanes never pick the same `i`, so those
//!   stores are spelled `atomic_assign`. Note this is *not* a read-modify-write and
//!   lowers to an ordinary `st.shared`; see section 4a for what it costs.

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
    //
    // CUDA:
    //     __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
    //     __shared__ uint32_t s_localHistogram[RADIX];
    //     volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];
    //
    // `s_warpHist` is a per-warp view into the shared array. SeGuRu has no aliasing
    // pointer to hand out, so every use below indexes `smem` at `wbase + ..` instead,
    // with `wbase = warp << RADIX_LOG` recomputed where it is needed.
    let smem = smem_alloc.alloc::<u32>(SMEM_WORDS as usize);

    // CUDA:
    //     for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)
    //         s_warpHistograms[i] = 0;
    //
    // The strided loop is a `chunk_mut` here: `MapLinear::new(1)` hands thread `t`
    // the elements `t, t + blockDim, ...`, which is the same access pattern, but
    // proved disjoint at compile time rather than by inspection. That is why this
    // clear needs no `Atomic` while the writes further down do.
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
    //
    // CUDA:
    //     if (blockIdx.x < gridDim.x - 1) {
    //         for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
    //              i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
    //             keys[i] = sort[t];
    //     }
    //     if (blockIdx.x == gridDim.x - 1) {
    //         for (...)
    //             keys[i] = t < size ? sort[t] : 0xffffffff;
    //     }
    //
    // The original carries two copies of the load, the second predicating every
    // access to synthesise `0xffffffff` dummy keys for a ragged final partition.
    // This port pads the key array on the host instead, so the tail block is a full
    // partition of real memory and one unpredicated loop covers every block. The
    // trick relies on the same property the CUDA comment cites: the sort is stable
    // and `0xffffffff` has the largest digit, so dummies always scatter last.
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
    // CUDA:
    //     unsigned warpFlags = 0xffffffff;
    //     for (int k = 0; k < RADIX_LOG; ++k) {
    //         const bool t2 = keys[i] >> k + radixShift & 1;
    //         warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
    //     }
    //     const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
    //     uint32_t preIncrementVal;
    //     if (bits == 0)
    //         preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK],
    //                                     __popc(warpFlags));
    //     offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
    //
    // Translated line for line: `flags`/`rank`/`reserved`/`leader` are `warpFlags`/
    // `bits`/`preIncrementVal`/`__ffs(warpFlags) - 1`. This `atomicAdd` is a genuine
    // read-modify-write and is atomic in both versions -- only one lane per match
    // group performs it, and the result is broadcast with a shuffle.
    //
    // One difference: CUDA leaves `preIncrementVal` uninitialised in the lanes that
    // skip the `atomicAdd`, relying on the shuffle to only ever read it from the
    // leader. Rust requires `reserved` to be initialised, so it starts at 0; the
    // shuffle overwrites it and the zero is never observed.
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
    //
    // CUDA:
    //     if (threadIdx.x < RADIX) {
    //         uint32_t reduction = s_warpHistograms[threadIdx.x];
    //         for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
    //             reduction += s_warpHistograms[i];
    //             s_warpHistograms[i] = reduction - s_warpHistograms[i];
    //         }
    //         s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    //     }
    //
    // `i` runs over a thread-dependent set of slots, so the two stores become
    // `atomic_assign`. They are disjoint by construction -- thread `t` only ever
    // touches slots congruent to `t` mod RADIX -- but that is a fact about the
    // arithmetic, not something the checker derives.
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

    // CUDA:
    //     if (threadIdx.x < (RADIX >> LANE_LOG))
    //         s_warpHistograms[threadIdx.x << LANE_LOG] =
    //             ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
    //
    // CUDA runs this on the 8 active lanes and uses `ActiveExclusiveWarpScan`, whose
    // shuffles are masked to just those lanes. SeGuRu's `shuffle!` takes the full
    // warp, so the guard is widened to `tid < 32` and the inactive lanes feed in 0,
    // which leaves the scan of the first 8 values unchanged. Same result, one extra
    // predicated store avoided by re-testing `tid < groups`.
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

    // CUDA:
    //     if (threadIdx.x < RADIX && getLaneId())
    //         s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
    //
    // CUDA folds the group base in only on lanes with `getLaneId() != 0`, using a
    // shuffle mask that excludes lane 0. The port reads `prev` unconditionally
    // (guarded against `tid == 0`), shuffles on the full warp, and selects with
    // `if lane != 0`, because the mask argument is not expressible here.
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
    //
    // CUDA:
    //     if (WARP_INDEX) {
    //         for (...) {
    //             const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
    //             offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
    //         }
    //     } else {
    //         for (...)
    //             offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
    //     }
    //
    // Identical, including the warp-0 special case: warp 0's own histogram *is* the
    // block-wide prefix array, so adding `s_warpHist[t2]` would double-count it.
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
    //
    // CUDA:
    //     if (threadIdx.x < RADIX)
    //         s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
    //             passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
    //
    // `s_localHistogram` is the separate CUDA allocation; here it is the tail of the
    // single buffer, at `smem[BIN_PART_SIZE + tid]`. Note `padded_thread_blocks`
    // replaces `gridDim.x`: the host pads the partition count, so the pass-histogram
    // stride is not the launch width.
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
    // CUDA:
    //     for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
    //         s_warpHistograms[offsets[i]] = keys[i];
    //
    // **This is the one place where the port is materially slower than the CUDA.**
    // The original is a plain `st.shared`. `offsets[i]` is a permutation of the tile
    // -- it was built as global_prefix[digit] + preceding_blocks[digit] + rank, so
    // digits occupy disjoint intervals and ranks within a digit are distinct -- and
    // the CUDA author simply knows that. SeGuRu cannot derive it, because it is a
    // property of the ranking arithmetic three sections above, so the store is
    // spelled `atomic_assign`.
    //
    // `atomic_assign` is a store, not a read-modify-write. **It costs nothing.** This
    // was measured rather than assumed: the whole scatter was rewritten to use a
    // `MapExplicit` chunk that carries the runtime destinations, removing the
    // `Atomic` entirely, and the generated PTX was byte-for-byte equivalent -- 29
    // atom/red instructions and 16 `st.shared.u32` either way, because
    // `atomic_assign` already lowers to a plain `st.shared.u32`. The 256 Mi sort
    // measured 19.606 ms with the map and 19.603 ms with the `Atomic`.
    //
    // So an earlier estimate that this `Atomic` cost ~40% of sort time (24.5 ms ->
    // 17.4 ms) was simply wrong, and the 2.2x gap against CUB is somewhere else.
    // `Atomic` here is a checker artefact with no codegen consequence.
    //
    // The map version is therefore *not* used: it buys no speed and costs an
    // `unsafe` block, since `MapExplicit::new` carries the uniqueness obligation.
    // The two toolchain fixes it needed were kept, because they were real bugs --
    // see finding 8 in ../FINDINGS.md.
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
    //
    // CUDA:
    //     if (blockIdx.x < gridDim.x - 1) {
    //         #pragma unroll BIN_KEYS_PER_THREAD
    //         for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
    //             alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] =
    //                 s_warpHistograms[i];
    //     }
    //     if (blockIdx.x == gridDim.x - 1) { ... same, bounded by finalPartSize ... }
    //
    // Again two copies in the original for the ragged tail, one here thanks to host
    // padding. The same `Atomic` remark as 4a applies, and this one is harder to
    // remove: the destination is in *global* memory, so it would need a map over the
    // output slice rather than over a shared buffer.
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
