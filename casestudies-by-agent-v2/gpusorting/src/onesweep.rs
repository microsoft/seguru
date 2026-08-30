//! OneSweep: LSD radix sort with decoupled look-back, in safe Rust.
//!
//! Ported from `OneSweep.cu` in Thomas Smith's GPUSorting (MIT,
//! <https://github.com/b0nes164/GPUSorting>). CUDA excerpts below are quoted
//! from that file for comparison.
//!
//! # Why this exists
//!
//! The reduce-then-scan sort in `upsweep.rs`/`scan.rs`/`downsweep.rs` reads the
//! keys **twice** per digit pass: once in the upsweep to build a per-tile
//! histogram, once in the downsweep to scatter. OneSweep reads them **once**. A
//! single [`global_histogram`] pass computes all four digits' histograms up front
//! -- legal because a global histogram is order-invariant -- and then each digit
//! pass is one fused kernel that obtains its tile's global offset by *decoupled
//! look-back* over its predecessors instead of from a precomputed global scan.
//!
//! At 256 Mi keys that is 9.7 GB of key traffic instead of 12.9 GB, and 4 kernel
//! launches instead of 12. It is the algorithm CUB actually dispatches on
//! sm_80 (`DeviceRadixSortOnesweepKernel`), which is why the reduce-then-scan
//! port trails CUB by 1.85x for reasons that have nothing to do with safety.
//!
//! # The three constructs that were assumed to be impossible
//!
//! All three turned out to be expressible; see `onesweep_probe.rs`.
//!
//! 1. **Decoupled look-back** -- an unbounded spin loop whose exit condition
//!    depends on what a *different block* published. SeGuRu accepts it: the loop
//!    contains no barrier, so thread-dependent trip counts cannot deadlock, and
//!    the flag slot is reached through [`gpu::sync::Atomic`], which is exactly
//!    the escape hatch for data-dependent global access.
//! 2. **An atomic load.** SeGuRu exposes no atomic read, but `atomic_ori(0)` is
//!    a read-modify-write that leaves the value alone and returns the old one.
//!    It lowers to `atom.global.or.b32`, which bypasses the non-coherent L1 --
//!    the same guarantee the CUDA gets from declaring `passHistogram` volatile.
//! 3. **Dynamic tile acquisition.** The CUDA takes its partition index from
//!    `atomicAdd(&index[pass], 1)` rather than `blockIdx`, so that a tile's
//!    predecessors are guaranteed to have been *scheduled* before it. That is
//!    what makes the look-back terminate. It is preserved here.
//!
//! What remains is a genuine assumption, not a toolchain limitation: look-back
//! deadlocks if a predecessor tile is never resident. CUDA does not promise
//! forward progress between blocks. Dynamic tile acquisition is what makes it
//! true in practice, and it is the reason upstream ships `EmulatedDeadlocking.cu`.

use crunchy::unroll;
use gpu::prelude::*;

use crate::utils::{
    exclusive_warp_scan, inclusive_warp_scan_circular_shift, lane_mask_lt, lowest_set_bit,
    LANE_COUNT, LANE_LOG,
};
use crate::{
    BIN_HISTS_SIZE, BIN_PART_SIZE, BIN_SUB_PART_SIZE, DOWNSWEEP_THREADS, PART_SIZE, RADIX,
    RADIX_LOG, RADIX_MASK, RADIX_PASSES, U32_4, UPSWEEP_THREADS,
};

const KPT: usize = (BIN_PART_SIZE / DOWNSWEEP_THREADS) as usize;
const VEC_PART_SIZE: u32 = PART_SIZE / 4;

/// Two sub-histograms halve shared-atomic contention, as in `upsweep.rs`.
const SUB_HISTS: u32 = 2;
/// One sub-histogram covers all four digit positions.
const GH_STRIDE: u32 = RADIX * RADIX_PASSES;
/// Shared words used by [`global_histogram`].
pub const GH_SMEM_WORDS: u32 = GH_STRIDE * SUB_HISTS;

/// Neither a reduction nor an inclusive prefix has been published for this tile.
const FLAG_NOT_READY: u32 = 0;
/// The tile has published its own digit counts, but not a global prefix.
const FLAG_REDUCTION: u32 = 1;
/// The payload is the *inclusive* global prefix through this tile.
const FLAG_INCLUSIVE: u32 = 2;
const FLAG_MASK: u32 = 3;

/// Shared words used by [`digit_binning_pass`]: the tile buffer, the digit base
/// offsets, and one slot for the dynamically acquired partition index.
pub const BIN_SMEM_WORDS: u32 = if BIN_PART_SIZE > BIN_HISTS_SIZE {
    BIN_PART_SIZE
} else {
    BIN_HISTS_SIZE
} + RADIX
    + 1;

/// Slot holding this block's acquired partition index.
const PART_SLOT: u32 = BIN_SMEM_WORDS - 1;
/// Base of the `RADIX` digit start offsets.
const BASE_SLOT: u32 = BIN_SMEM_WORDS - 1 - RADIX;

/// Length of the look-back buffer for one digit pass, in `u32`.
///
/// Tile `i` publishes into slot `i + 1`; slot `0` holds the global digit base
/// that [`onesweep_scan`] seeds with `FLAG_INCLUSIVE`, which is what terminates
/// the look-back.
pub fn pass_hist_len(tiles: u32) -> usize {
    (RADIX_PASSES * (tiles + 1) * RADIX) as usize
}

/// All four digit histograms of the whole key array, in one pass.
///
/// This is the entire reason OneSweep reads the keys once per digit rather than
/// twice: a *global* histogram does not depend on the order of the keys, so all
/// four passes' histograms can be computed before any key moves. A *per-tile*
/// histogram, which is what the reduce-then-scan upsweep builds, does not have
/// that property -- tile membership changes after every permutation, so it must
/// be rebuilt every pass.
///
/// CUDA (`OneSweep::GlobalHistogram`, abridged -- upstream keeps four separate
/// `RADIX * 2` shared arrays and unrolls the 16 byte extractions by hand):
/// ```text
/// __shared__ uint32_t s_globalHistFirst[RADIX * 2]; ... Sec, Third, Fourth
/// for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x) { ... = 0; }
/// __syncthreads();
/// uint32_t* s_wavesHistFirst = &s_globalHistFirst[threadIdx.x / 64 * RADIX];
/// ...
/// uint4 t[1] = { reinterpret_cast<uint4*>(sort)[i] };
/// atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
/// ...
/// for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x)
///     atomicAdd(&globalHistogram[i], s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);
/// ```
///
/// Here the four arrays are one buffer indexed `sub * GH_STRIDE + pass * RADIX +
/// digit`, and the byte extraction is a shift-and-mask loop, which generates the
/// same code. The host pads the key array to a whole number of `PART_SIZE`
/// partitions with `u32::MAX`, so there is no ragged-tail branch: the padding
/// keys are counted in the histogram, scatter to the very end because the sort
/// is stable, and are dropped on the way back to the host.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn global_histogram(sort: &[U32_4], global_hist: &mut [u32]) {
    assert!(Config::BDIM_X == UPSWEEP_THREADS);
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();

    let smem = smem_alloc.alloc::<u32>(GH_SMEM_WORDS as usize);

    {
        let mut z = smem.chunk_mut(MapLinear::new(1));
        unroll! {
            for k in 0..16 {
                z[k] = 0u32;
            }
        }
    }
    sync_threads();

    // Threads 0..63 accumulate into sub-histogram 0, 64..127 into 1.
    {
        let hist = gpu::sync::SharedAtomic::new(&mut *smem);
        let wave = (tid / 64) * GH_STRIDE;
        let start = bid * VEC_PART_SIZE;
        let mut i = start + tid;
        let end = start + VEC_PART_SIZE;
        while i < end {
            let v = sort[i as usize].data();
            unroll! {
                for j in 0..4 {
                    let key = v[j];
                    unroll! {
                        for p in 0..4 {
                            let d = (key >> ((p as u32) * RADIX_LOG)) & RADIX_MASK;
                            hist.index((wave + (p as u32) * RADIX + d) as usize)
                                .atomic_addi(1u32);
                        }
                    }
                }
            }
            i += UPSWEEP_THREADS;
        }
    }
    sync_threads();

    // Fold the two sub-histograms and accumulate into the device histogram.
    {
        let gh = gpu::sync::Atomic::new(global_hist);
        let mut i = tid;
        while i < GH_STRIDE {
            let a = *smem[i as usize];
            let b = *smem[(i + GH_STRIDE) as usize];
            gh.index(i as usize).atomic_addi(a + b);
            i += UPSWEEP_THREADS;
        }
    }
}

/// Turn the global histogram into each pass's digit base offsets, and seed the
/// look-back chain.
///
/// Launched with one block per digit pass. The exclusive prefix over the 256
/// digit counts is written into tile slot `0` already tagged `FLAG_INCLUSIVE`,
/// so a look-back that walks all the way back to tile 0 terminates there with
/// the correct global base.
///
/// CUDA (`OneSweep::Scan`):
/// ```text
/// s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
/// __syncthreads();
/// if (threadIdx.x < (RADIX >> LANE_LOG))
///     s_scan[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_scan[threadIdx.x << LANE_LOG]);
/// __syncthreads();
/// firstPassHistogram[threadIdx.x] =
///     (s_scan[threadIdx.x] + (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1) : 0)) << 2 | FLAG_INCLUSIVE;
/// ```
///
/// Upstream takes four separate output pointers and picks between them with a
/// `switch (blockIdx.x)`; one buffer indexed by `pass` is the same thing. The
/// `tid < 32` guard rather than `tid < 8` is the usual SeGuRu accommodation --
/// `shuffle!` takes the whole warp, so the inactive lanes feed in zeros, which
/// leaves the scan of the first 8 values unchanged.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn onesweep_scan(global_hist: &[u32], pass_hist: &mut [u32], tiles: u32) {
    assert!(Config::BDIM_X == RADIX);
    let tid = thread_id::<DimX>();
    let pass = block_id::<DimX>();
    let lane = lane_id();

    let smem = smem_alloc.alloc::<u32>(RADIX as usize);

    {
        let scanned = inclusive_warp_scan_circular_shift(global_hist[(tid + pass * RADIX) as usize]);
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

    {
        let mine = *smem[tid as usize];
        let prev = if tid > 0 { *smem[(tid - 1) as usize] } else { 0u32 };
        let (group_base, _) = gpu::shuffle!(idx, prev, 1u32, 32);
        let base = if lane != 0 { mine + group_base } else { mine };

        // Tile 0's slot of this pass. Stride between passes is (tiles + 1) * RADIX.
        let ph = gpu::sync::Atomic::new(pass_hist);
        let slot = pass * (tiles + 1) * RADIX + tid;
        ph.index(slot as usize)
            .atomic_assign((base << 2) | FLAG_INCLUSIVE);
    }
}

/// One digit pass: rank, publish, look back, scatter.
///
/// Sections 0-3 and 5 are the same as `downsweep::radix_downsweep`, which is a
/// transliteration of the reduce-then-scan `DownsweepKeysOnly`; the two kernels
/// share the warp-level multi-split and the shared-memory staging verbatim. What
/// differs is how the tile learns where its digits start in the output:
///
/// * the downsweep **reads** `globalHist[digit] + passHist[digit][tile]`, both
///   precomputed by two earlier kernels over the whole array;
/// * this kernel **publishes** its own digit counts and then **looks back** over
///   its predecessors' publications until it finds one that is already a
///   complete prefix.
///
/// That is the whole of the algorithmic difference, and it is what removes the
/// upsweep and the scan from every pass.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn digit_binning_pass(
    sort: &[u32],
    alt: &mut [u32],
    pass_hist: &mut [u32],
    index: &mut [u32],
    radix_shift: u32,
    tiles: u32,
    // Measurement hook: when 0, the backwards walk starts at slot 0 instead of at
    // the immediate predecessor. Slot 0 is seeded INCLUSIVE by `onesweep_scan`, so
    // the look-back terminates on its first read and no tile ever waits, while the
    // publish and the read traffic stay exactly as they are in a real run. The sort
    // is then wrong, but the difference against a normal run prices the *waiting*.
    // Always 1 in `onesweep_sort`.
    do_lookback: u32,
) {
    assert!(Config::BDIM_X == DOWNSWEEP_THREADS);
    let tid = thread_id::<DimX>();
    let lane = lane_id();
    let warp = tid >> LANE_LOG;
    let pass = radix_shift >> 3;

    let smem = smem_alloc.alloc::<u32>(BIN_SMEM_WORDS as usize);
    // `Atomic::new` consumes the slice, and both the publish in section 3 and the
    // look-back in section 5 need it, so the view is taken once up front.
    let ph = gpu::sync::Atomic::new(pass_hist);
    let pass_base = pass * (tiles + 1) * RADIX;

    // ---- 0. Clear the warp histograms and acquire a partition ----------------------
    {
        let mut z = smem.chunk_mut(MapLinear::new(1));
        unroll! {
            for k in 0..8 {
                z[k] = 0u32;
            }
        }
    }

    // CUDA:
    //     if (threadIdx.x == 0)
    //         s_warpHistograms[BIN_PART_SIZE - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
    //     __syncthreads();
    //     const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];
    //
    // The tile index is *acquired*, not derived from `blockIdx`. This matters for
    // termination rather than for correctness: it guarantees that a tile's
    // predecessors were scheduled before it, so the look-back below cannot wait on
    // a block that has not started. Upstream stashes it in the last slot of the
    // key buffer; here it gets its own slot, since our tile buffer is exactly
    // `BIN_PART_SIZE` and has no spare element.
    if tid == 0 {
        let idx = gpu::sync::Atomic::new(index);
        let acquired = idx.index(pass as usize).atomic_addi(1u32);
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index(PART_SLOT as usize).atomic_assign(acquired);
    }
    sync_threads();
    let part = *smem[PART_SLOT as usize];

    // ---- 1. Load this thread's keys ------------------------------------------------
    // Lane-major within the warp's sub-partition, so each load is one 128-byte
    // transaction per warp. Host padding removes the ragged-tail branch, exactly as
    // in the downsweep.
    let mut keys = [0u32; KPT];
    {
        let start = part * BIN_PART_SIZE + warp * BIN_SUB_PART_SIZE + lane;
        unroll! {
            for i in 0..8 {
                keys[i] = sort[(start + (i as u32) * LANE_COUNT) as usize];
            }
        }
    }

    // ---- 2. Warp-level multi-split -------------------------------------------------
    // Identical to `downsweep.rs` section 2; see the commentary there.
    let mut offsets = [0u32; KPT];
    {
        let hist = gpu::sync::SharedAtomic::new(&mut *smem);
        let wbase = warp << RADIX_LOG;
        let lt = lane_mask_lt();
        unroll! {
            for i in 0..8 {
                let key = keys[i];
                let digit = (key >> radix_shift) & RADIX_MASK;

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

    // ---- 3. Exclusive scan across the per-warp histograms, and publish -------------
    // CUDA:
    //     if (threadIdx.x < RADIX) {
    //         uint32_t reduction = s_warpHistograms[threadIdx.x];
    //         for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX) {
    //             reduction += s_warpHistograms[i];
    //             s_warpHistograms[i] = reduction - s_warpHistograms[i];
    //         }
    //         atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
    //             FLAG_REDUCTION | reduction << 2);
    //         s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    //     }
    //
    // `running` ends as this block's total count of digit `tid`, which is precisely
    // what the successors' look-back needs. Publishing it here -- before the ranking
    // work below and well before the scatter -- is what lets the whole grid make
    // progress concurrently.
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

        let slot = pass_base + (part + 1) * RADIX + tid;
        ph.index(slot as usize)
            .atomic_addi(FLAG_REDUCTION | (running << 2));

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

    // Save the block-local digit prefix before the scatter overwrites it.
    //
    // Upstream keeps `s_localHistogram` in its own `__shared__` array, so it
    // survives the scatter into `s_warpHistograms` and the look-back can subtract
    // from it afterwards. Our tile buffer and histogram region are the same
    // `BIN_PART_SIZE == BIN_HISTS_SIZE` words, so the value has to be moved out of
    // the way first. Same arithmetic, one extra copy.
    if tid < RADIX {
        let local = *smem[tid as usize];
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index((BASE_SLOT + tid) as usize).atomic_assign(local);
    }
    sync_threads();

    // ---- 4. Stage the tile in shared memory ----------------------------------------
    // See `downsweep.rs` 4a: `atomic_assign` is a plain `st.shared.u32` and costs
    // nothing; it is a checker artefact, measured and confirmed by PTX diff.
    {
        let s = gpu::sync::SharedAtomic::new(&mut *smem);
        unroll! {
            for i in 0..8 {
                s.index(offsets[i] as usize).atomic_assign(keys[i]);
            }
        }
    }
    sync_threads();

    // ---- 5. Decoupled look-back ----------------------------------------------------
    // CUDA:
    //     if (threadIdx.x < RADIX) {
    //         uint32_t reduction = 0;
    //         for (uint32_t k = partitionIndex; k >= 0; ) {
    //             const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];
    //             if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
    //                 reduction += flagPayload >> 2;
    //                 atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
    //                     1 | (reduction << 2));
    //                 s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
    //                 break;
    //             }
    //             if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
    //                 reduction += flagPayload >> 2;
    //                 k--;
    //             }
    //         }
    //     }
    //
    // Thread `tid` owns digit `tid` and walks *its own column* of the flag array
    // backwards. Each thread spins a different number of times on a value written
    // by a different block, and there is no barrier inside the loop -- which is
    // exactly why the thread-dependent trip count is safe and why SeGuRu accepts
    // it.
    //
    // The `atomic_addi(1 | reduction << 2)` promotes this tile's own slot from
    // REDUCTION to INCLUSIVE in one step: the flag field goes 1 + 1 = 2 and the
    // payload becomes `own_count + prefix`, which is the inclusive prefix. A
    // successor that reaches this slot can then stop immediately.
    //
    // `atomic_ori(0)` is the read. SeGuRu has no atomic load, but an OR with zero
    // is an RMW that returns the old value unchanged, and it lowers to
    // `atom.global.or.b32` -- device-scope and L1-bypassing, which is the property
    // the CUDA gets by declaring the buffer `volatile`. A plain load would be
    // allowed to hit a stale non-coherent L1 line and spin forever.
    if tid < RADIX {
        let mut reduction = 0u32;
        let mut k = if do_lookback != 0 { part } else { 0 };
        loop {
            let flag = ph.index((pass_base + k * RADIX + tid) as usize).atomic_ori(0u32);
            let kind = flag & FLAG_MASK;
            if kind == FLAG_INCLUSIVE {
                reduction += flag >> 2;
                ph.index((pass_base + (part + 1) * RADIX + tid) as usize)
                    .atomic_addi(1u32 | (reduction << 2));
                break;
            }
            if kind == FLAG_REDUCTION {
                reduction += flag >> 2;
                // Slot 0 is seeded INCLUSIVE by `onesweep_scan`, so this cannot
                // underflow; the guard is belt and braces.
                if k == 0 {
                    break;
                }
                k -= 1;
            }
            // FLAG_NOT_READY: spin on the same slot until the predecessor publishes.
        }

        let local = *smem[(BASE_SLOT + tid) as usize];
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index((BASE_SLOT + tid) as usize)
            .atomic_assign(reduction - local);
    }
    sync_threads();

    // ---- 6. Scatter out to global memory -------------------------------------------
    // Consecutive threads read consecutive shared slots, so keys sharing a digit
    // land in consecutive global addresses: coalesced runs.
    {
        let out = gpu::sync::Atomic::new(alt);
        unroll! {
            for k in 0..8 {
                let i = tid + (k as u32) * DOWNSWEEP_THREADS;
                let key = *smem[i as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                let base = *smem[(BASE_SLOT + digit) as usize];
                out.index((base + i) as usize).atomic_assign(key);
            }
        }
    }

    let _ = FLAG_NOT_READY;
}
