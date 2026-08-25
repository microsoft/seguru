//! **Standing reproducer for two open SeGuRu compiler bugs. This is not a
//! benchmark and it is not part of the operator library** — it exists so the
//! bugs stay reproducible after this port is finished. Run it with
//! `cargo run --release -p kernelbench-gpu --bin probe`.
//!
//! Bug 1 — *scope-chunk lane addressing*: in the standard "stage one value per
//! warp in shared memory" idiom, only lane 0 of warp `w` addresses `smem[w]`;
//! every other lane silently addresses `smem[0]`, i.e. warp 0's slot. Writes
//! from any other lane are lost and corrupt a neighbour, with no compile error,
//! no bounds check and no runtime diagnostic.
//!
//! Bug 2 — *`GpuShared::zero()` does not zero*: `GpuShared::<[f32; 32]>::zero()`
//! leaves the array holding whatever the previous kernel launch left in that
//! shared address, despite the constructor name promising zeroed memory. A
//! kernel that reads a slot before writing it therefore observes another
//! kernel's data. Visible in the `only lane 31` row below, where `smem[1..8]`
//! still holds the values written by the *previous* launch.
//!
//! Idiom under test (the "stage one value per warp in shared memory" step of a
//! block reduction, as written in `examples/llm-rs-gpu` and in this crate's
//! `src/device.rs`):
//!
//! ```ignore
//! let mut s = smem
//!     .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
//!     .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
//! s[0] = warp_value;
//! ```
//!
//! Expectation: `s` is warp `w`'s private one-element slot, i.e. every lane of
//! warp `w` addresses `smem[w]`, so it should not matter which lane performs
//! the write (publishing an inclusive-scan total from lane 31 is the natural
//! thing to do).
//!
//! Actual behaviour: only **lane 0** addresses its own warp's slot. Every other
//! lane silently addresses `smem[0]` instead - warp 0's slot - so a write from
//! any lane other than 0 both loses its own value and corrupts warp 0's. No
//! bounds check, no compile error, no runtime diagnostic.
//!
//! Run with: `cargo run --release -p kernelbench-gpu --bin probe`
//!
//! Observed on A100 / CUDA 13.3, 256 threads = 8 warps, value = `100*warp + lane`:
//!
//! ```text
//! all lanes write: [701, 100, 200, 300, 400, 500, 600, 700, 0, ...]
//! only lane 0    : [  0, 100, 200, 300, 400, 500, 600, 700, 0, ...]   <- correct
//! only lane 31   : [731, 100, 200, 300, 400, 500, 600, 700, 0, ...]
//! ```
//!
//! In `all lanes write`, lane 0 of each warp produced the correct `smem[w] =
//! 100*w`, while all 248 other lanes raced on `smem[0]` (winner 701 = warp 7,
//! lane 1). In `only lane 31`, the eight warp totals were never published at
//! all: `smem[0]` got 731 and `smem[1..8]` are *stale values left over from the
//! previous kernel launch* - a second observation worth noting, since
//! `GpuShared::<[f32; 32]>::zero()` does not zero the array at run time.
//!
use gpu::chunk_scope::{Block, Thread, build_chunk_scope};
use gpu::prelude::*;
use gpu::cg::{CGOperations, ThreadWarpTile};
use gpu_host::gpu_config;

const BLOCK: u32 = 256;

/// `mode`: 0 = every lane writes, 1 = only lane 0 writes, 2 = only lane 31.
#[gpu::cuda_kernel]
pub fn stage_probe_kernel(out: &mut [f32], mode: u32) {
    let warp = ThreadWarpTile::<32>;
    let lane = warp.thread_rank();
    let warp_id = warp.subgroup_id();
    let mut smem = GpuShared::<[f32; 32]>::zero();

    let value = 100.0 * warp_id as f32 + lane as f32;
    sync_threads();
    {
        let mut s = smem
            .chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
            .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1));
        let write = (mode == 0) || (mode == 1 && lane == 0) || (mode == 2 && lane == 31);
        if write {
            s[0] = value;
        }
    }
    sync_threads();

    let mut o = chunk_mut(out, reshape_map!([1] | [BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]));
    o[0] = smem[lane as usize];
}

fn main() {
    let host = vec![-1.0f32; BLOCK as usize];
    gpu_host::cuda_ctx(0, |ctx, m| {
        for mode in 0..3u32 {
            let mut d_out = ctx.new_tensor_view::<[f32]>(&host).unwrap();
            let cfg = gpu_config!(1, 1, 1, @const BLOCK, 1, 1, 0);
            stage_probe_kernel::launch(cfg, ctx, m, &mut d_out, mode).unwrap();
            let mut got = vec![0.0f32; BLOCK as usize];
            d_out.copy_to_host(&mut got).unwrap();
            let label = match mode {
                0 => "all lanes write",
                1 => "only lane 0   ",
                _ => "only lane 31  ",
            };
            println!("{label}: smem = {:?}", &got[..32]);
        }
    });
}
