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
//! All three turned out to be expressible.
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
    LANE_COUNT, LANE_LOG, exclusive_warp_scan, inclusive_warp_scan_circular_shift, lane_mask_lt,
    lowest_set_bit,
};
use crate::{
    BIN_HISTS_SIZE, BIN_PART_SIZE, BIN_SUB_PART_SIZE, DOWNSWEEP_THREADS, PART_SIZE, RADIX,
    RADIX_LOG, RADIX_MASK, RADIX_PASSES, U32_4, UPSWEEP_THREADS,
};

const KPT: usize = (BIN_PART_SIZE / DOWNSWEEP_THREADS) as usize;
const VEC_PART_SIZE: u32 = PART_SIZE / 4;

/// Two sub-histograms halve shared-atomic contention, as in `upsweep.rs`.
const SUB_HISTS: u32 = 2;
/// One digit pass's pair of sub-histograms: upstream's `RADIX * 2` array.
const GH_PASS_WORDS: u32 = RADIX * SUB_HISTS;
/// Shared words used by [`global_histogram`]: upstream's four `RADIX * 2`
/// arrays, concatenated.
const GH_SMEM_WORDS: usize = (GH_PASS_WORDS * RADIX_PASSES) as usize;

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
/// Here the four arrays are one buffer indexed `pass * GH_PASS_WORDS + sub *
/// RADIX + digit`, which is the four of them concatenated, and the byte
/// extraction is a shift-and-mask loop rather than sixteen hand-written
/// `reinterpret_cast<uint8_t*>` reads. Byte `4 * j + p` of the `uint4` is byte
/// `p` of key `j` on a little-endian device, so the two agree.
///
/// The clear loop and its barrier are gone: [`GpuShared::init`] is a block-wide
/// collective that hands the elements out round-robin and ends with its own
/// `sync_threads`, so the buffer is zeroed and published before it returns.
///
/// The shared accumulation keeps its atomic. A bin index is a *key digit*, so
/// two threads holding the same digit must reach the same address; that is not
/// statically disjoint under any chunk map, and privatising per thread would
/// need 512 KB against the 164 KB a block has. Upstream reaches the same
/// conclusion -- all sixteen of its shared updates are `atomicAdd`. The reduce
/// loop below is the opposite case: it only *reads* shared memory, which needs
/// no permission at all, and its `atomicAdd` is on global memory because every
/// block accumulates into the same device histogram.
///
/// The host pads the key array to a whole number of `PART_SIZE` partitions with
/// `u32::MAX`, so there is no ragged-tail branch: the padding keys are counted
/// in the histogram, scatter to the very end because the sort is stable, and are
/// dropped on the way back to the host.
#[gpu::cuda_kernel]
pub fn global_histogram(sort: &[U32_4], global_hist: &mut [u32]) {
    assert!(Config::BDIM_X == UPSWEEP_THREADS);
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();

    let mut smem = GpuShared::<[u32; GH_SMEM_WORDS]>::init(0u32);

    // 64 threads to a sub-histogram, as upstream.
    {
        let hist: gpu::sync::SharedAtomic<[u32]> = gpu::sync::SharedAtomic::new(&mut smem);
        let wave = (tid / 64) * RADIX;
        let start = bid * VEC_PART_SIZE;
        let end = start + VEC_PART_SIZE;
        let mut i = start + tid;
        while i < end {
            let k = sort[i as usize];
            let v = k.data();
            unroll! {
                for j in 0..4 {
                    let key = v[j];
                    unroll! {
                        for p in 0..4 {
                            let d = (key >> ((p as u32) * RADIX_LOG)) & RADIX_MASK;
                            hist.index(((p as u32) * GH_PASS_WORDS + wave + d) as usize)
                                .atomic_addi(1u32);
                        }
                    }
                }
            }
            i += UPSWEEP_THREADS;
        }
    }
    sync_threads();

    // Fold each pass's two sub-histograms and add them to the device histogram.
    {
        let s = &*smem;
        let gh = gpu::sync::Atomic::new(global_hist);
        let mut i = tid;
        while i < RADIX {
            unroll! {
                for p in 0..4 {
                    let base = (p as u32) * GH_PASS_WORDS + i;
                    let total = s[base as usize] + s[(base + RADIX) as usize];
                    gh.index(((p as u32) * RADIX + i) as usize).atomic_addi(total);
                }
            }
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

    let smem = smem_alloc.alloc::<u32>(RADIX as usize, 0u32);

    {
        let scanned =
            inclusive_warp_scan_circular_shift(global_hist[(tid + pass * RADIX) as usize]);
        let mut w = smem.chunk_mut(MapLinear::new(1));
        w[0] = scanned;
    }
    sync_threads();

    let groups = RADIX >> LANE_LOG; // 8
    let v = if tid < groups {
        *smem[(tid << LANE_LOG) as usize]
    } else {
        0u32
    };
    {
        let mut w = smem.chunk_mut(MapLinear::new(1usize << LANE_LOG));
        if tid < 32 {
            let s = exclusive_warp_scan(v);
            if tid < groups {
                w[0] = s;
            }
        }
    }
    sync_threads();

    {
        let mine = *smem[tid as usize];
        let prev = if tid > 0 {
            *smem[(tid - 1) as usize]
        } else {
            0u32
        };
        let (group_base, _) = gpu::shuffle!(idx, prev, 1u32, 32);
        let base = if lane != 0 { mine + group_base } else { mine };

        // Tile 0's slot of this pass. Stride between passes is (tiles + 1) * RADIX.
        let mut ph = chunk_mut(
            pass_hist,
            reshape_map!([1u32] | [(RADIX, (tiles + 1) * RADIX), grid_dim::<DimX>()] => layout: [t0, t1, i0]),
        );
        ph[0u32] = (base << 2) | FLAG_INCLUSIVE;
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
#[cfg_attr(feature = "launch_bound", gpu::attr(nvvm_launch_bound(512, 1, 1, 3)))]
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

    let smem = smem_alloc.alloc::<u32>(BIN_SMEM_WORDS as usize, 0u32);
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
    let mut scanned = 0u32;
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

        scanned = inclusive_warp_scan_circular_shift(running);
    }
    {
        let mut w = smem.chunk_mut(MapLinear::new(1));
        if tid < RADIX {
            w[0] = scanned;
        }
    }
    sync_threads();

    let groups = RADIX >> LANE_LOG; // 8
    let v = if tid < groups {
        *smem[(tid << LANE_LOG) as usize]
    } else {
        0u32
    };
    {
        let mut w = smem.chunk_mut(MapLinear::new(1usize << LANE_LOG));
        if tid < 32 {
            let s = exclusive_warp_scan(v);
            if tid < groups {
                w[0] = s;
            }
        }
    }
    sync_threads();

    let mut total = 0u32;
    if tid < RADIX {
        let mine = *smem[tid as usize];
        let prev = if tid > 0 {
            *smem[(tid - 1) as usize]
        } else {
            0u32
        };
        let (group_base, _) = gpu::shuffle!(idx, prev, 1u32, 32);
        total = if lane != 0 { mine + group_base } else { mine };
    }
    {
        let mut w = smem.chunk_mut(MapLinear::new(1));
        if tid < RADIX {
            w[0] = total;
        }
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
    let local = if tid < RADIX {
        *smem[tid as usize]
    } else {
        0u32
    };
    {
        let mut w = smem.chunk_mut(
            reshape_map!([1] | [DOWNSWEEP_THREADS] => layout: [t0, i0], offset: BASE_SLOT),
        );
        if tid < RADIX {
            w[0] = local;
        }
    }
    sync_threads();

    // ---- 4. Stage the tile in shared memory ----------------------------------------
    // `offsets[i]` is a rank, so the destinations are injective but only by a
    // counting argument -- the ballot multi-split gives each key a distinct rank
    // within its warp+digit group, and the two prefix sums lift that to a distinct
    // rank in the tile. That is a theorem about the preceding sections, not a
    // property of the index expression, so `chunk_mut` can only accept it through
    // `MapExplicit`, whose obligation is discharged by hand.
    //
    // Measured: the `MapExplicit` form does turn all eight stores into plain
    // `st.shared.u32` (`atom.shared.exch` 18 -> 10), but it is worth *nothing* --
    // 23.556 ms against 23.555 ms at 256 Mi. Uncontended `atom.shared.exch.b32`
    // already runs at store throughput, so the scatter stays on the safe API and
    // the port keeps a single `unsafe` line, in section 6, where the same change
    // to the *global* scatter is worth 25%.
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
    let mut reduction = 0u32;
    if tid < RADIX {
        let mut k = if do_lookback != 0 { part } else { 0 };
        loop {
            let flag = ph
                .index((pass_base + k * RADIX + tid) as usize)
                .atomic_ori(0u32);
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
    }

    let base_local = if tid < RADIX {
        *smem[(BASE_SLOT + tid) as usize]
    } else {
        0u32
    };
    {
        let mut w = smem.chunk_mut(
            reshape_map!([1] | [DOWNSWEEP_THREADS] => layout: [t0, i0], offset: BASE_SLOT),
        );
        if tid < RADIX {
            w[0] = reduction - base_local;
        }
    }
    sync_threads();

    // ---- 6. Scatter out to global memory -------------------------------------------
    // Consecutive threads read consecutive shared slots, so keys sharing a digit
    // land in consecutive global addresses: coalesced runs.
    //
    // This is where the cost is. `base + i` is a global rank, injective for the same
    // counting reason as section 4, but as an `Atomic` each store lowers to
    // `atom.global.exch.b32`, which does not coalesce. Routing it through
    // `MapExplicit` restores eight plain `st.global.u32` and takes 256 Mi from
    // 23.555 ms to 17.687 ms -- the whole of the onesweep gap over CUDA.
    #[cfg(not(feature = "safe_only"))]
    {
        let mut dests = [0u32; KPT];
        let mut vals = [0u32; KPT];
        unroll! {
            for k in 0..8 {
                let i = tid + (k as u32) * DOWNSWEEP_THREADS;
                let key = *smem[i as usize];
                let digit = (key >> radix_shift) & RADIX_MASK;
                let base = *smem[(BASE_SLOT + digit) as usize];
                dests[k] = base + i;
                vals[k] = key;
            }
        }
        let len = alt.len() as u32;
        // SAFETY: `base` is the digit's global offset and `i` the key's rank within
        // that digit, so `base + i` is injective over the whole grid.
        let map = unsafe { MapExplicit::<KPT>::new(dests, len) };
        let mut w = chunk_mut(alt, map);
        unroll! {
            for k in 0..8 {
                w[k as usize] = vals[k];
            }
        }
    }
    #[cfg(feature = "safe_only")]
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
