//! GPUSorting radix-sort upsweep in cuda-oxide, for comparison against the
//! SeGuRu port in `../../gpusorting/src/upsweep.rs`.
//!
//! Pass 1 of each radix digit: per-block digit histograms. Same algorithm and
//! same tuning as the SeGuRu kernel (128 threads, 4096 keys per partition, two
//! sub-histograms to halve shared-atomic contention, a warp-scan of the digit
//! groups, then a grid-wide accumulate), so the two ports differ only in the
//! toolchain and the safety model.
//!
//! # What could and could not be written safely
//!
//! Everything that touches *global* memory here is safe. `pass_hist` is written
//! at `digit * padded_thread_blocks + block`, an index that has nothing to do
//! with the writing thread's own coordinates, and `global_hist` is accumulated
//! across the whole grid; cuda-oxide expresses both with `DeviceAtomicU32`,
//! whose methods are safe. The warp scans are safe. `sync_threads` is safe.
//!
//! The **shared-memory histogram is not expressible safely**. It has to be a
//! `static mut SharedArray`, so every one of the accesses below is `unsafe`,
//! and the block-scoped atomic increments need `BlockAtomicU32::from_ptr`,
//! which is an `unsafe fn` taking a raw pointer. `crates/cuda-device/src/
//! shared.rs:41-46` gives the reason: the memory is uninitialised at kernel
//! entry and the barriers that order it are invisible to the type system.
//!
//! The SeGuRu kernel this is compared against does the same thing, including
//! the shared-memory atomics, with no `unsafe` at all.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::{
    SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};
use std::time::Instant;

const RADIX: u32 = 256;
const RADIX_MASK: u32 = 255;
const RADIX_PASSES: u32 = 4;
const UPSWEEP_THREADS: u32 = 128;
const PART_SIZE: u32 = 4096;
const SCAN_THREADS: u32 = 128;
/// Two independent sub-histograms halve shared-atomic contention.
const SUB_HISTS: u32 = 2;
const HIST_WORDS: usize = (RADIX * SUB_HISTS) as usize;
/// `PART_SIZE` keys as 128-bit quads.
const VEC_PART_SIZE: u32 = PART_SIZE / 4;
const LANE_LOG: u32 = 5;
const LANE_MASK: u32 = 31;

/// Four keys in one 128-bit element, so the histogram loop issues
/// `ld.global.v4.u32` exactly like the SeGuRu port's `U32_4`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct U32x4([u32; 4]);

// SAFETY: a plain POD aggregate of four `u32` with no pointers or padding, so
// it is sound to memcpy between host and device.
unsafe impl cuda_core::DeviceCopy for U32x4 {}

#[cuda_module]
mod kernels {
    use super::*;

    /// Inclusive prefix sum across the 32 lanes of a warp.
    #[cuda_device::device]
    #[inline(always)]
    fn inclusive_warp_scan(val: u32) -> u32 {
        let mut x = val;
        let lane = warp::lane_id();
        let mut k = 0u32;
        while k < 5 {
            let delta = 1u32 << k;
            let t = warp::shuffle_up(x, delta);
            if lane >= delta {
                x += t;
            }
            k += 1;
        }
        x
    }

    /// Exclusive prefix sum across the 32 lanes of a warp.
    #[cuda_device::device]
    #[inline(always)]
    fn exclusive_warp_scan(val: u32) -> u32 {
        let inc = inclusive_warp_scan(val);
        let t = warp::shuffle_up(inc, 1);
        if warp::lane_id() != 0 { t } else { 0 }
    }

    /// Inclusive scan rotated left by one lane: lane `k` receives the inclusive
    /// sum of lanes `0..k`, and lane 0 receives the whole warp's total.
    #[cuda_device::device]
    #[inline(always)]
    fn inclusive_warp_scan_circular_shift(val: u32) -> u32 {
        let inc = inclusive_warp_scan(val);
        let src = (warp::lane_id() + LANE_MASK) & LANE_MASK;
        warp::shuffle(inc, src)
    }

    /// Per-block digit histograms, plus the grid-wide exclusive digit prefix.
    ///
    /// The host pads the key array to a whole number of `PART_SIZE` partitions
    /// with `u32::MAX`, so this kernel has no ragged-tail branch.
    #[kernel(launch_context = lc)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
    pub fn radix_upsweep(
        sort: &[super::U32x4],
        global_hist: &[DeviceAtomicU32],
        pass_hist: &[DeviceAtomicU32],
        radix_shift: u32,
        padded_thread_blocks: u32,
    ) {
        static mut HIST: SharedArray<u32, HIST_WORDS> = SharedArray::UNINIT;

        let _ = lc;
        let tid = thread::threadIdx_x();
        let bid = thread::blockIdx_x();
        let lane = warp::lane_id();

        // Shared memory is uninitialised at kernel entry, so the histogram has
        // to be zeroed by hand; SeGuRu's `smem_alloc.alloc(.., 0u32)` does this
        // as part of the allocation.
        let mut z = tid as usize;
        while z < HIST_WORDS {
            // SAFETY: `z < HIST_WORDS`, and the stride is the block width, so
            // each slot is written by exactly one thread.
            unsafe {
                HIST[z] = 0;
            }
            z += UPSWEEP_THREADS as usize;
        }
        thread::sync_threads();

        // Histogram this block's partition. Threads 0..63 accumulate into
        // sub-histogram 0 and threads 64..127 into sub-histogram 1.
        let wave = (tid / 64) * RADIX;
        let start = bid * VEC_PART_SIZE;
        let mut i = start + tid;
        let end = start + VEC_PART_SIZE;
        while i < end {
            let quad = if (i as usize) < sort.len() {
                sort[i as usize]
            } else {
                super::U32x4([u32::MAX; 4])
            };
            let mut j = 0usize;
            while j < 4 {
                let d = (quad.0[j] >> radix_shift) & RADIX_MASK;
                let slot = (wave + d) as usize;
                // SAFETY: `wave + d < RADIX * SUB_HISTS`. The increment must be
                // atomic because threads of the same half-block collide on a
                // digit; `BlockAtomicU32::from_ptr` is the only way to get a
                // block-scoped atomic over shared memory, and it is an
                // `unsafe fn`.
                let cell = unsafe {
                    BlockAtomicU32::from_ptr(SharedArray::as_raw_mut_ptr(&raw mut HIST).add(slot))
                };
                cell.fetch_add(1, AtomicOrdering::Relaxed);
                j += 1;
            }
            i += UPSWEEP_THREADS;
        }
        thread::sync_threads();

        // Fold the two sub-histograms together, publish the per-block counts,
        // and start the digit-wise scan that `global_hist` needs.
        let mut k = 0u32;
        while k < 2 {
            let d = tid + k * UPSWEEP_THREADS;
            // SAFETY: `d < RADIX` and `d + RADIX < RADIX * SUB_HISTS`.
            let total = unsafe { HIST[d as usize] + HIST[(d + RADIX) as usize] };

            // `digit * padded_thread_blocks + block` is unrelated to this
            // thread's own index, so no index witness can describe it. A
            // relaxed atomic store is the safe way to write it.
            let ph = (d * padded_thread_blocks + bid) as usize;
            if ph < pass_hist.len() {
                pass_hist[ph].store(total, AtomicOrdering::Relaxed);
            }

            let scanned = inclusive_warp_scan_circular_shift(total);
            // SAFETY: `d < RADIX`; one writer per slot.
            unsafe {
                HIST[d as usize] = scanned;
            }
            k += 1;
        }
        thread::sync_threads();

        // Exclusive scan of the eight 32-digit group totals, which the circular
        // shift above parked in lane 0 of each group. The guard is `tid < 32`,
        // i.e. exactly warp 0, so the shuffles still see a full warp.
        if tid < 32 {
            let groups = RADIX >> LANE_LOG;
            // SAFETY: `tid << LANE_LOG < RADIX` when `tid < groups`.
            let v = if tid < groups {
                unsafe { HIST[(tid << LANE_LOG) as usize] }
            } else {
                0
            };
            let s = exclusive_warp_scan(v);
            if tid < groups {
                // SAFETY: as above; lane `tid` is the only writer of its slot.
                unsafe {
                    HIST[(tid << LANE_LOG) as usize] = s;
                }
            }
        }
        thread::sync_threads();

        // Accumulate this block's contribution into the grid-wide digit offsets.
        let base = radix_shift << 5;
        let mut k = 0u32;
        while k < 2 {
            let d = tid + k * UPSWEEP_THREADS;
            // SAFETY: `d < RADIX`, and `d - 1 < RADIX` when `d > 0`.
            let (mine, prev) = unsafe {
                (
                    HIST[d as usize],
                    if d > 0 { HIST[(d - 1) as usize] } else { 0 },
                )
            };
            // Lane 0 of each 32-digit group already holds the group's exclusive
            // prefix; every other lane needs it broadcast from lane 1's read of
            // its predecessor.
            let group_base = warp::shuffle(prev, 1);
            let val = if lane != 0 { mine + group_base } else { mine };
            let gi = (base + d) as usize;
            if gi < global_hist.len() {
                global_hist[gi].fetch_add(val, AtomicOrdering::Relaxed);
            }
            k += 1;
        }
    }

}

fn thread_blocks(n: usize) -> u32 {
    ((n as u32).div_ceil(PART_SIZE)).max(1)
}

fn padded_thread_blocks(n: usize) -> u32 {
    thread_blocks(n).div_ceil(SCAN_THREADS) * SCAN_THREADS
}

/// Deterministic pseudo-random keys, the generator the SeGuRu sort tests use.
fn keys(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

/// CPU reference for the two outputs, mirroring the kernel's layouts.
fn upsweep_cpu(padded: &[u32], radix_shift: u32, tb: u32, ptb: u32) -> (Vec<u32>, Vec<u32>) {
    let mut pass_hist = vec![0u32; (RADIX * ptb) as usize];
    let mut digit_totals = vec![0u32; RADIX as usize];
    for b in 0..tb {
        let start = (b * PART_SIZE) as usize;
        let mut local = vec![0u32; RADIX as usize];
        for &key in &padded[start..start + PART_SIZE as usize] {
            local[((key >> radix_shift) & RADIX_MASK) as usize] += 1;
        }
        for d in 0..RADIX as usize {
            pass_hist[d * ptb as usize + b as usize] = local[d];
            digit_totals[d] += local[d];
        }
    }
    // The kernel accumulates the grid-wide *exclusive* prefix over digits.
    let mut global = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    let base = (radix_shift << 5) as usize;
    let mut running = 0u32;
    for d in 0..RADIX as usize {
        global[base + d] = running;
        running += digit_totals[d];
    }
    (global, pass_hist)
}

const WARMUP: u32 = 5;

fn bench(n: usize, verify: bool) -> (f64, bool) {
    let radix_shift = 0u32;
    let tb = thread_blocks(n);
    let ptb = padded_thread_blocks(n);
    let padded_len = (tb * PART_SIZE) as usize;

    let mut host_keys = keys(n, 7);
    host_keys.resize(padded_len, u32::MAX);

    let gh_len = (RADIX * RADIX_PASSES) as usize;
    let ph_len = (RADIX * ptb) as usize;

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let packed: Vec<U32x4> = host_keys
        .chunks_exact(4)
        .map(|c| U32x4([c[0], c[1], c[2], c[3]]))
        .collect();
    let d_keys = DeviceBuffer::from_host(&stream, &packed).unwrap();
    // `DeviceAtomicU32` is not `DeviceCopy`, so the host cannot allocate or
    // fill one directly. cuda-oxide's intended mapping is to allocate a plain
    // buffer and reinterpret its element type; `cast_elem` is safe and only
    // changes the type the kernel sees.
    let mut d_gh = DeviceBuffer::<u32>::zeroed(&stream, gh_len)
        .unwrap()
        .cast_elem::<DeviceAtomicU32>();
    let mut d_ph = DeviceBuffer::<u32>::zeroed(&stream, ph_len)
        .unwrap()
        .cast_elem::<DeviceAtomicU32>();

    // SAFETY: cuda-oxide makes module loading unsafe unconditionally.
    let module = unsafe { kernels::load(&ctx) }.unwrap();

    let cfg = LaunchConfig1D::new(tb, UPSWEEP_THREADS, 0);
    let prep = module.prepare_radix_upsweep(cfg).unwrap();

    macro_rules! once {
        () => {
            module
                .radix_upsweep(&stream, &prep, &d_keys, &d_gh, &d_ph, radix_shift, ptb)
                .unwrap()
        };
    }

    for _ in 0..WARMUP {
        once!();
    }
    stream.synchronize().unwrap();

    let iters = (2.0e9 / (n as f64 * 4.0)).clamp(20.0, 500.0) as u32;
    let t = Instant::now();
    for _ in 0..iters {
        once!();
    }
    stream.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let ok = if verify {
        // A clean run from zeroed accumulators, since the timed loop summed
        // `iters + WARMUP` runs into them.
        let zeros_gh = vec![0u32; gh_len];
        let zeros_ph = vec![0u32; ph_len];
        let mut plain_gh: DeviceBuffer<u32> = d_gh.cast_elem();
        let mut plain_ph: DeviceBuffer<u32> = d_ph.cast_elem();
        plain_gh.copy_from_host(&stream, &zeros_gh).unwrap();
        plain_ph.copy_from_host(&stream, &zeros_ph).unwrap();
        d_gh = plain_gh.cast_elem();
        d_ph = plain_ph.cast_elem();

        once!();
        stream.synchronize().unwrap();

        let plain_gh: DeviceBuffer<u32> = d_gh.cast_elem();
        let plain_ph: DeviceBuffer<u32> = d_ph.cast_elem();
        let got_gh = plain_gh.to_host_vec(&stream).unwrap();
        let got_ph = plain_ph.to_host_vec(&stream).unwrap();
        let (want_gh, want_ph) = upsweep_cpu(&host_keys, radix_shift, tb, ptb);
        got_gh == want_gh && got_ph == want_ph
    } else {
        true
    };

    (us, ok)
}

fn main() {
    println!("kernel,impl,size,us,ok");
    for &bits in &[20usize, 22, 24, 26] {
        let n = 1usize << bits;
        let (us, ok) = bench(n, true);
        println!("radix_upsweep,oxide,2^{bits},{us:.3},{ok}");
    }
}
