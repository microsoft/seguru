//! Probes for the three constructs a OneSweep port needs, tested one at a time.
//!
//! OneSweep replaces the reduce-then-scan sequence (upsweep, scan, downsweep per
//! digit) with a single fused pass per digit that obtains its tile's global
//! offset by *decoupled look-back*: each tile publishes its own digit counts,
//! then walks backwards over its predecessors' published flags until it finds an
//! already-inclusive prefix. It reads the keys once per digit instead of twice.
//!
//! Three things in the CUDA original looked like they might be inexpressible in
//! safe SeGuRu. This file tests each in isolation rather than guessing:
//!
//! * `lookback_probe` — an unbounded spin loop whose trip count and exit depend
//!   on values another *block* wrote to global memory.
//! * `atomic_read_probe` — reading a global location atomically. SeGuRu exposes
//!   no atomic load, but `atomic_ori(0)` is an RMW that returns the old value
//!   and lowers to `atom.global.or.b32`, which (like the `volatile` load the
//!   CUDA uses) bypasses the non-coherent L1.
//! * `dynamic_tile_probe` — the CUDA acquires its partition index with
//!   `atomicAdd(&index[pass], 1)` rather than using `blockIdx`.
//!
//! Build one at a time: the GPU backend aborts at the first analysis error, so a
//! probe that shares a crate with a failing kernel is never analysed at all and
//! "passing" means nothing.

use gpu::prelude::*;

use crate::{RADIX, RADIX_MASK};

const FLAG_NOT_READY: u32 = 0;
const FLAG_REDUCTION: u32 = 1;
const FLAG_INCLUSIVE: u32 = 2;
const FLAG_MASK: u32 = 3;

/// Probe 1: the decoupled look-back loop.
///
/// CUDA:
/// ```text
/// if (threadIdx.x < RADIX) {
///     uint32_t reduction = 0;
///     for (uint32_t k = partitionIndex; k >= 0; ) {
///         const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];
///         if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
///             reduction += flagPayload >> 2;
///             atomicAdd(&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
///                       1 | (reduction << 2));
///             s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
///             break;
///         }
///         if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
///             reduction += flagPayload >> 2;
///             k--;
///         }
///     }
/// }
/// ```
///
/// Every thread spins a different number of times, on a value a different block
/// publishes. There is no `sync_threads` inside the loop, which is what makes it
/// legal at all -- a barrier under thread-dependent control flow would deadlock.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn lookback_probe(pass_hist: &mut [u32], out: &mut [u32]) {
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();

    if tid < RADIX {
        let ph = gpu::sync::Atomic::new(pass_hist);
        let mut reduction = 0u32;
        let mut k = bid;
        loop {
            let flag = ph.index((tid + k * RADIX) as usize).atomic_ori(0u32);
            let kind = flag & FLAG_MASK;
            if kind == FLAG_INCLUSIVE {
                reduction += flag >> 2;
                break;
            }
            if kind == FLAG_REDUCTION {
                reduction += flag >> 2;
                if k == 0 {
                    break;
                }
                k -= 1;
            }
            // kind == FLAG_NOT_READY: spin, re-reading the same slot.
        }
        let o = gpu::sync::Atomic::new(out);
        o.index((tid + bid * RADIX) as usize).atomic_assign(reduction);
    }
}

/// Probe 2: an atomic *read* of global memory, and an atomic publish.
///
/// SeGuRu has no atomic load, but `atomic_ori(0)` and `atomic_addi(0)` are RMWs
/// that return the old value without changing it.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn atomic_read_probe(pass_hist: &mut [u32], out: &mut [u32]) {
    let tid = thread_id::<DimX>();
    let bid = block_id::<DimX>();

    let ph = gpu::sync::Atomic::new(pass_hist);
    let slot = (tid + bid * RADIX) as usize;
    let seen = ph.index(slot).atomic_ori(0u32);
    ph.index(slot).atomic_addi(FLAG_REDUCTION | (seen << 2));

    let o = gpu::sync::Atomic::new(out);
    o.index(slot).atomic_assign(seen & RADIX_MASK);
}

/// Probe 3: acquiring the partition index dynamically instead of using `blockIdx`.
///
/// CUDA:
/// ```text
/// if (threadIdx.x == 0)
///     s_warpHistograms[BIN_PART_SIZE - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
/// __syncthreads();
/// const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];
/// ```
///
/// The tile index is then used to address global memory. If SeGuRu rejects this,
/// the fallback is to use `blockIdx` directly, which costs the guarantee that a
/// tile's predecessors were launched before it.
#[gpu::cuda_kernel(dynamic_shared)]
pub fn dynamic_tile_probe(index: &mut [u32], sort: &[u32], out: &mut [u32]) {
    let tid = thread_id::<DimX>();
    let smem = smem_alloc.alloc::<u32>(1usize);

    if tid == 0 {
        let idx = gpu::sync::Atomic::new(index);
        let acquired = idx.index(0usize).atomic_addi(1u32);
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index(0usize).atomic_assign(acquired);
    }
    sync_threads();

    let part = *smem[0usize];
    let key = sort[(part * RADIX + tid) as usize];
    let o = gpu::sync::Atomic::new(out);
    o.index((part * RADIX + tid) as usize).atomic_assign(key);
}
