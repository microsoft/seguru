//! GPU radix sort (LSD, 8 bits per pass) written in safe Rust with SeGuRu.
//!
//! Algorithm follows Thomas Smith's `GPUSorting` CUDA reference
//! (`DeviceRadixSort`): four passes of upsweep → scan → downsweep over 32-bit keys,
//! with warp-level multi-split ranking in the downsweep.
//!
//! Differences from a direct transcription of the CUDA code, all of them made to
//! suit SeGuRu's safety model or to go faster:
//!
//! * **Padded partitions.** The host rounds the key array up to a whole number of
//!   partitions and fills the slack with `u32::MAX`. Sorting is a permutation of a
//!   multiset, so the padding sorts to the end and is simply dropped. This removes
//!   every ragged-tail predicate from all three kernels and lets the upsweep load
//!   exclusively through `U32_4`.
//! * **Register-resident key arrays.** All per-thread loops are unrolled with
//!   `crunchy::unroll!` so the `[u32; 15]` key and offset arrays get promoted to
//!   registers by `mem2reg`; with a runtime loop index they would be spilled.
//! * **Hardware `ffs`.** Match-group leaders are found with `u32::trailing_zeros()`.
//! * **Structured scan output.** `radix_scan` writes each thread's own slot via
//!   `chunk_mut` rather than the CUDA original's circular-lane-shifted scatter.
//! * **Warp-boundary guards.** Inter-warp scans are guarded by `tid < 32` (or
//!   `tid < RADIX`) rather than `tid < 8`, so shuffles always see a complete warp.
//!
//! There is no `unsafe` anywhere in this crate.

pub mod clear;
pub mod downsweep;
pub mod driver;
pub mod onesweep;
pub mod onesweep_driver;
pub mod onesweep_probe;
pub mod scan;
pub mod upsweep;
pub mod utils;

#[cfg(feature = "bench")]
pub mod cuda_ffi;

#[cfg(test)]
mod tests;

pub use driver::{radix_sort, radix_sort_timed};
pub use onesweep_driver::{onesweep_sort, onesweep_sort_timed};
pub use gpu::vector::VecTypeTrait;
pub use gpu::U32_4;

pub const RADIX: u32 = 256;
pub const RADIX_LOG: u32 = 8;
pub const RADIX_MASK: u32 = 255;
pub const RADIX_PASSES: u32 = 4;

pub const UPSWEEP_THREADS: u32 = 128;
pub const PART_SIZE: u32 = 4096;

pub const SCAN_THREADS: u32 = 128;

pub const DOWNSWEEP_THREADS: u32 = 512;
pub const BIN_PART_SIZE: u32 = 4096;
pub const BIN_WARPS: u32 = 16;
pub const BIN_HISTS_SIZE: u32 = BIN_WARPS * RADIX;
pub const BIN_SUB_PART_SIZE: u32 = BIN_PART_SIZE / BIN_WARPS;
pub const BIN_KEYS_PER_THREAD: u32 = BIN_PART_SIZE / DOWNSWEEP_THREADS;

/// Words of shared memory the downsweep needs.
///
/// The same buffer is used first for the per-warp histograms (`BIN_HISTS_SIZE`
/// words) and then reused as the local scatter tile (`BIN_PART_SIZE` words), so it
/// must be large enough for whichever is bigger, plus `RADIX` words for the digit
/// base offsets. At the default tile size the two happen to be equal; with a
/// smaller keys-per-thread the histogram becomes the binding term.
pub const SMEM_WORDS: u32 = if BIN_PART_SIZE > BIN_HISTS_SIZE {
    BIN_PART_SIZE
} else {
    BIN_HISTS_SIZE
} + RADIX;



/// Number of partitions needed for `n` keys.
pub fn thread_blocks(n: usize) -> u32 {
    (((n as u32) + PART_SIZE - 1) / PART_SIZE).max(1)
}

/// `thread_blocks` rounded up to a multiple of `SCAN_THREADS`, which is the stride
/// used by the `pass_hist` layout so that `radix_scan` can chunk it evenly.
pub fn padded_thread_blocks(n: usize) -> u32 {
    let tb = thread_blocks(n);
    tb.div_ceil(SCAN_THREADS) * SCAN_THREADS
}

/// Pack `keys` into `U32_4` lanes, padding to a whole number of partitions with
/// `u32::MAX`. Done without `unsafe`; the padding sorts to the tail and is dropped.
pub fn pack_padded(keys: &[u32]) -> Vec<U32_4> {
    let padded = (thread_blocks(keys.len()) * PART_SIZE) as usize;
    let mut out = Vec::with_capacity(padded / 4);
    let mut i = 0usize;
    while i < padded {
        let g = |k: usize| -> u32 {
            if i + k < keys.len() {
                keys[i + k]
            } else {
                u32::MAX
            }
        };
        out.push(U32_4::new([g(0), g(1), g(2), g(3)]));
        i += 4;
    }
    out
}

/// Unpack the first `n` keys out of a `U32_4` buffer.
pub fn unpack(packed: &[U32_4], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for v in packed {
        let d = v.data();
        for k in 0..4 {
            if out.len() == n {
                return out;
            }
            out.push(d[k]);
        }
    }
    out
}
