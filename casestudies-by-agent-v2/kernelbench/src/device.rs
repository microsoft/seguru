//! Cooperative-group reduction helpers shared by the row kernels.
//!
//! The pattern is the one used throughout `examples/llm-rs-gpu`: reduce inside
//! each warp with shuffles (`ThreadWarpTile::redux`), stage one value per warp
//! in shared memory, then reduce those with a second warp-level shuffle. It is
//! two shuffle reductions and one 32-element shared array, instead of the
//! `log2(blockDim)` shared-memory round trips a tree reduction needs.
//!
//! Every helper opens with `sync_threads()` so it is safe to call several of
//! them in sequence even if the compiler happens to place their staging arrays
//! at the same shared address. All of them must be called from non-divergent
//! control flow (`sync_threads` and `GpuShared::zero` are `sync_data` APIs).

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ReduxMin, ThreadWarpTile, WarpReduceOp};
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::*;

/// Sum `val` across the whole thread block; every thread receives the result.
#[gpu::device]
#[gpu_codegen::sync_data]
#[inline(always)]
pub fn block_reduce_sum(val: f32) -> f32 {
    let warp = ThreadWarpTile::<32>;
    let mut smem = GpuShared::<[f32; 32]>::init(0.0f32);
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let warp_val = warp.redux(ReduxAdd, val);
    sync_threads();
    {
        let mut s = smem
            .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
            .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
        if lane == 0 {
            s[0] = warp_val;
        }
    }
    sync_threads();
    let v = if lane < num_warps { smem[lane as usize] } else { 0.0 };
    warp.redux(ReduxAdd, v)
}

/// Maximum of `val` across the whole thread block.
#[gpu::device]
#[gpu_codegen::sync_data]
#[inline(always)]
pub fn block_reduce_max(val: f32) -> f32 {
    let warp = ThreadWarpTile::<32>;
    let mut smem = GpuShared::<[f32; 32]>::init(0.0f32);
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let warp_val = warp.redux(ReduxMax, val);
    sync_threads();
    {
        let mut s = smem
            .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
            .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
        if lane == 0 {
            s[0] = warp_val;
        }
    }
    sync_threads();
    let v = if lane < num_warps { smem[lane as usize] } else { f32::MIN };
    warp.redux(ReduxMax, v)
}

/// Minimum of `val` across the whole thread block, used by `argmax` to pick the
/// smallest index among the lanes holding the row maximum.
///
/// The warp stage is the hardware `redux.sync` instruction, which became usable
/// at 32 lanes once the `BASE_THREAD_MASK` overflow in `crates/gpu/src/cg.rs`
/// was fixed. Measured ~8% faster end to end than an equivalent
/// `shuffle!(xor, ...)` butterfly (12.8 us vs 13.9 us for argmax at 4096x1024).
#[gpu::device]
#[gpu_codegen::sync_data]
#[inline(always)]
pub fn block_reduce_min_i32(val: i32) -> i32 {
    let warp = ThreadWarpTile::<32>;
    let mut smem = GpuShared::<[i32; 32]>::init(0i32);
    let lane = warp.thread_rank();
    let num_warps = warp.meta_group_size();
    let warp_val = warp.redux(ReduxMin, val);
    sync_threads();
    {
        let mut s = smem
            .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
            .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
        if lane == 0 {
            s[0] = warp_val;
        }
    }
    sync_threads();
    let v = if lane < num_warps { smem[lane as usize] } else { i32::MAX };
    warp.redux(ReduxMin, v)
}

/// Inclusive block-wide scan of `val`; returns the sum of all *preceding*
/// threads (the exclusive prefix) and the total over the block.
#[gpu::device]
#[gpu_codegen::sync_data]
#[inline(always)]
pub fn block_exclusive_scan(val: f32) -> (f32, f32) {
    let warp = ThreadWarpTile::<32>;
    let mut smem = GpuShared::<[f32; 32]>::init(0.0f32);
    let lane = warp.thread_rank();
    let warp_id = warp.subgroup_id();
    let num_warps = warp.meta_group_size();

    // Hillis-Steele inclusive scan inside the warp via shuffle-up.
    let mut acc = val;
    let mut offset = 1u32;
    while offset < 32 {
        let (peer, _) = gpu::shuffle!(up, acc, offset, 32);
        if lane >= offset {
            acc += peer;
        }
        offset *= 2;
    }
    // In the staging chunk below only lane 0 of warp `w` addresses `smem[w]`;
    // every other lane silently addresses `smem[0]` (see `src/bin/probe.rs`).
    // So take the warp total from a butterfly reduction, which every lane sees,
    // rather than publishing lane 31's inclusive-scan result.
    let warp_total = warp.redux(ReduxAdd, val);

    sync_threads();
    {
        let mut s = smem
            .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
            .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
        if lane == 0 {
            s[0] = warp_total;
        }
    }
    sync_threads();

    // Every thread sums the totals of the warps in front of it; with at most 32
    // warps per block this is cheaper than a second scan.
    let mut warp_prefix = 0.0f32;
    let mut block_total = 0.0f32;
    let mut w = 0u32;
    while w < num_warps {
        let t = smem[w as usize];
        if w < warp_id {
            warp_prefix += t;
        }
        block_total += t;
        w += 1;
    }
    (warp_prefix + acc - val, block_total)
}
