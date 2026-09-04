//! Times `radix_upsweep` on its own, so it can be compared kernel-for-kernel
//! against the cuda-oxide port in `../../../oxide/sort-upsweep/`.
//!
//! The full sort benchmark in `bench.rs` measures four passes of upsweep, scan
//! and downsweep together, which cannot be compared against a port of one
//! kernel. This binary isolates pass 1 with the same methodology the rest of
//! the suite uses: allocation and host transfers happen once, outside the timed
//! region, `warmup` untimed launches precede `iters` timed ones, and a single
//! `ctx.sync()` brackets the measurement.
//!
//! `cargo run --release -p gpusorting-gpu --bin upsweep-bench`

use gpu_host::gpu_config;
use gpusorting_gpu::{
    PART_SIZE, RADIX, RADIX_MASK, RADIX_PASSES, UPSWEEP_THREADS, pack_padded, padded_thread_blocks,
    thread_blocks, upsweep::radix_upsweep,
};
use std::time::Instant;

const WARMUP: usize = 5;

/// The key generator used by the cuda-oxide port, so both sides histogram
/// exactly the same data.
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

/// CPU reference for the kernel's two outputs, in the kernel's own layouts.
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
    let mut global = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    let base = (radix_shift << 5) as usize;
    let mut running = 0u32;
    for d in 0..RADIX as usize {
        global[base + d] = running;
        running += digit_totals[d];
    }
    (global, pass_hist)
}

fn bench(n: usize) -> (f64, bool) {
    let radix_shift = 0u32;
    let tb = thread_blocks(n);
    let ptb = padded_thread_blocks(n);

    let host_keys = keys(n, 7);
    let packed = pack_padded(&host_keys);
    // `pack_padded` fills the slack with `u32::MAX`; rebuild the same padded
    // key stream on the host for the reference.
    let mut padded = host_keys.clone();
    padded.resize((tb * PART_SIZE) as usize, u32::MAX);

    let gh_len = (RADIX * RADIX_PASSES) as usize;
    let ph_len = (RADIX * ptb) as usize;
    let iters = (2.0e9 / (n as f64 * 4.0)).clamp(20.0, 500.0) as usize;

    let mut elapsed_us = 0.0f64;
    let ok = gpu_host::cuda_ctx(0, |ctx, m| {
        let zeros_gh = vec![0u32; gh_len];
        let zeros_ph = vec![0u32; ph_len];
        let d_keys = ctx.new_tensor_view(packed.as_slice()).unwrap();
        let mut d_gh = ctx.new_tensor_view(zeros_gh.as_slice()).unwrap();
        let mut d_ph = ctx.new_tensor_view(zeros_ph.as_slice()).unwrap();

        macro_rules! once {
            () => {{
                let cfg =
                    gpu_config!(tb, 1, 1, @const UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
                radix_upsweep::launch(cfg, ctx, m, &d_keys, &mut d_gh, &mut d_ph, radix_shift, ptb)
                    .unwrap();
            }};
        }

        for _ in 0..WARMUP {
            once!();
        }
        ctx.sync().unwrap();

        let t = Instant::now();
        for _ in 0..iters {
            once!();
        }
        ctx.sync().unwrap();
        elapsed_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // A clean run from zeroed accumulators: the timed loop summed every
        // launch into the same buffers.
        d_gh.copy_from_host(zeros_gh.as_slice()).unwrap();
        d_ph.copy_from_host(zeros_ph.as_slice()).unwrap();
        once!();
        ctx.sync().unwrap();

        let mut got_gh = vec![0u32; gh_len];
        let mut got_ph = vec![0u32; ph_len];
        d_gh.copy_to_host(&mut got_gh).unwrap();
        d_ph.copy_to_host(&mut got_ph).unwrap();

        let (want_gh, want_ph) = upsweep_cpu(&padded, radix_shift, tb, ptb);
        got_gh == want_gh && got_ph == want_ph
    });

    (elapsed_us, ok)
}

fn main() {
    println!("kernel,impl,size,us,ok");
    for &bits in &[20usize, 22, 24, 26] {
        let n = 1usize << bits;
        let (us, ok) = bench(n);
        println!("radix_upsweep,seguru,2^{bits},{us:.3},{ok}");
    }
}
