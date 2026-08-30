//! Launches the OneSweep probes so they are actually analysed and codegenned.
//!
//! A `#[gpu::cuda_kernel]` function is generic over its launch config and is only
//! instantiated where `::launch` is called, so a probe with no host caller is
//! never seen by the GPU backend at all. Declaring the module is not enough.
//!
//! Run with `cargo run --release -p gpusorting-gpu --bin onesweep-probe`.
//! Comment out the probes you are not testing: the backend aborts at the first
//! analysis error, so a later probe in the same crate is never reached.

use gpu_host::gpu_config;
use gpusorting_gpu::onesweep_probe::{atomic_read_probe, dynamic_tile_probe, lookback_probe};
use gpusorting_gpu::RADIX;

fn main() {
    const BLOCKS: u32 = 4;
    const THREADS: u32 = 256;
    let ph_len = (RADIX * (BLOCKS + 1)) as usize;
    let out_len = (RADIX * BLOCKS) as usize;

    let out: Vec<u32> = gpu_host::cuda_ctx(0, |ctx, m| {
        // Every predecessor tile already publishes an INCLUSIVE flag, so the
        // look-back terminates on its first read and the probe cannot hang.
        let seed: Vec<u32> = (0..ph_len).map(|_| 2u32 | (1u32 << 2)).collect();
        let zeros = vec![0u32; out_len];
        let idx_seed = vec![0u32; 1];
        let keys: Vec<u32> = (0..out_len as u32).collect();

        let mut d_ph = ctx.new_tensor_view::<[u32]>(&seed).unwrap();
        let mut d_out = ctx.new_tensor_view::<[u32]>(&zeros).unwrap();
        let mut d_idx = ctx.new_tensor_view::<[u32]>(&idx_seed).unwrap();
        let d_keys = ctx.new_tensor_view::<[u32]>(&keys).unwrap();

        macro_rules! cfg {
            () => {
                gpu_config!(BLOCKS, 1, 1, @const THREADS, 1, 1, 4096u32)
            };
        }

        lookback_probe::launch(cfg!(), ctx, m, &mut d_ph, &mut d_out).unwrap();
        println!("probe 1 (decoupled look-back): launched");

        atomic_read_probe::launch(cfg!(), ctx, m, &mut d_ph, &mut d_out).unwrap();
        println!("probe 2 (atomic read via ori(0)): launched");

        dynamic_tile_probe::launch(cfg!(), ctx, m, &mut d_idx, &d_keys, &mut d_out).unwrap();
        println!("probe 3 (dynamic tile acquisition): launched");

        ctx.sync().unwrap();
        let mut host_out = vec![0u32; out_len];
        d_out.copy_to_host(&mut host_out).unwrap();
        host_out
    });
    println!("out[0..8] = {:?}", &out[..8]);
}
