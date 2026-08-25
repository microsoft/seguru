//! Pass 2 of each radix digit: exclusive scan of `pass_hist` along the block axis.
//!
//! One CTA per digit. `pass_hist[digit * padded_thread_blocks + b]` is replaced by
//! the sum of all `b' < b`, so that a downsweep block can compute where its keys of
//! a given digit start.
//!
//! Mirrors `DeviceRadixSort::Scan`, except that the CUDA original writes results
//! back through a circular-lane-shifted (data-dependent) address in order to turn an
//! inclusive scan into an exclusive one. SeGuRu proves write disjointness
//! structurally instead: we keep every thread writing its *own* slot and derive the
//! exclusive value locally with one extra `shfl.up`. Same result, no scatter, and no
//! atomics on the output.

use gpu::prelude::*;

use crate::utils::{inclusive_warp_scan, LANE_LOG};
use crate::SCAN_THREADS;

#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_scan(pass_hist: &mut [u32], padded_thread_blocks: u32) {
    assert!(Config::BDIM_X == SCAN_THREADS);
    let tid = thread_id::<DimX>();
    let lane = lane_id();
    let warp = tid >> LANE_LOG;
    const WARPS: u32 = SCAN_THREADS >> LANE_LOG;

    let smem = smem_alloc.alloc::<u32>(SCAN_THREADS as usize);

    // chunk[k] -> pass_hist[bid * padded_thread_blocks + k * SCAN_THREADS + tid]
    let local = padded_thread_blocks / SCAN_THREADS;
    let mut ph = chunk_mut(
        pass_hist,
        reshape_map!([local] | [SCAN_THREADS, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
    );

    let mut carry = 0u32;
    let mut part = 0u32;
    while part < local {
        let mine = ph[part];
        let inc = inclusive_warp_scan(mine);
        {
            let w = gpu::sync::SharedAtomic::new(&mut *smem);
            w.index(tid as usize).atomic_assign(inc);
        }
        sync_threads();

        // Inclusive scan over the four warp totals, run on all of warp 0 so the
        // shuffles keep a full warp (unused lanes contribute 0).
        if tid < 32 {
            let v = if tid < WARPS {
                *smem[((((tid + 1) << LANE_LOG) - 1)) as usize]
            } else {
                0u32
            };
            let s = inclusive_warp_scan(v);
            if tid < WARPS {
                let w = gpu::sync::SharedAtomic::new(&mut *smem);
                w.index((((tid + 1) << LANE_LOG) - 1) as usize)
                    .atomic_assign(s);
            }
        }
        sync_threads();

        // Exclusive value for this thread = (predecessor's inclusive within warp)
        //                                 + (total of all preceding warps)
        //                                 + (carry from preceding partitions).
        let (prev, _) = gpu::shuffle!(up, inc, 1u32, 32);
        let within = if lane > 0 { prev } else { 0u32 };
        let across = if warp > 0 {
            *smem[((warp << LANE_LOG) - 1) as usize]
        } else {
            0u32
        };
        let block_total = *smem[(SCAN_THREADS - 1) as usize];

        ph[part] = within + across + carry;
        carry += block_total;
        sync_threads();

        part += 1;
    }
}
