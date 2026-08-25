//! PolyBench/GPU ported to SeGuRu.
//!
//! Every benchmark lives in its own module and exposes three things: the
//! SeGuRu kernel(s), a host driver that pads the data, launches and unpads,
//! and a scalar CPU reference used as the correctness oracle.

use crunchy::unroll;

pub mod common;

/// CUDA C++ reference bindings, used only by the `polybench-bench` binary.
#[cfg(feature = "bench")]
pub mod cuda_ffi;

pub mod atax;
pub mod bicg;
pub mod conv2d;
pub mod conv3d;
pub mod corr;
pub mod covar;
pub mod doitgen;
pub mod fdtd2d;
pub mod gemm;
pub mod gesummv;
pub mod gramschm;
pub mod jacobi1d;
pub mod jacobi2d;
pub mod lu;
pub mod mvt;
pub mod syr2k;
pub mod syrk;
pub mod threemm;
pub mod twomm;

/// Index a read-only kernel parameter without paying an unelidable bounds
/// check.
///
/// SeGuRu emits, for every `&[T]` index in a kernel, a `setp.gt.u64` plus two
/// `selp.b64` that LLVM cannot fold away, because nothing bounds a
/// thread-derived index. The kernels here use this two-step idiom to hand LLVM
/// the missing range fact, in safe Rust and for arbitrary sizes:
///
/// ```ignore
/// let total = ni * nj;                                   // u32, exact extent
/// if total == 0 || a.len() < total as usize { return; }
/// let a = &a[..total as usize];                          // len == zext(total)
/// let last = total - 1;
/// ...
/// a[ix(i * nj + j, last)]
/// ```
///
/// The sub-slice is what makes it work in 32 bits: `a.len()` is then literally
/// `zext(total)`, so LLVM can compare it against `zext(umin(idx, total - 1))`
/// by comparing the `u32` operands, and the check folds. Clamping against
/// `a.len() - 1` directly is also provable but only in 64 bits, which costs a
/// `min.u64` per access and recovers only about a third as much (see
/// `README.md`, experiment A).
///
/// The clamp is a no-op on the data: every index the kernels produce is
/// already `< total`, so `min` never fires and the results are unchanged. It
/// replaces a *panic* on an out-of-range index with a read of the last
/// element, which is why it is confined to reads.
#[gpu::device]
#[inline(always)]
pub(crate) fn ix(idx: u32, last: u32) -> usize {
    idx.min(last) as usize
}

/// Full-warp sum reduction via butterfly shuffles.
///
/// Every lane ends up with the total, so callers pick one lane to store.
#[gpu::device]
#[inline(always)]
pub(crate) fn warp_sum(v: f32) -> f32 {
    let mut v = v;
    unroll! {
        for s in 0..5 {
            let (peer, _) = gpu::shuffle!(xor, v, 1u32 << s, 32u32);
            v += peer;
        }
    }
    v
}
