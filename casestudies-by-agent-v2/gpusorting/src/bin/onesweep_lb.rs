//! Prices the decoupled look-back spin.
//!
//! Runs the identical kernel twice: once normally, once with every tile's
//! backwards walk starting at slot 0, which `onesweep_scan` always seeds
//! INCLUSIVE. The second run publishes and reads exactly as much as the first but
//! never waits on a predecessor, so it sorts incorrectly and the difference is the
//! cost of the waiting alone.

use gpusorting_gpu::onesweep_driver::onesweep_sort_inner;

fn lcg(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        })
        .collect()
}

fn main() {
    println!("| {:>7} | {:>12} | {:>14} | {:>9} |", "keys", "onesweep ms", "no wait ms", "wait");
    println!("|{:->9}|{:->14}|{:->16}|{:->11}|", "", "", "", "");
    for (label, n) in [("4 Mi", 1usize << 22), ("16 Mi", 1 << 24), ("64 Mi", 1 << 26), ("256 Mi", 1 << 28)] {
        let keys = lcg(0x5EED, n);
        let iters = if n <= 1 << 24 { 50 } else { 20 };
        let (_, full) = onesweep_sort_inner(&keys, 5, iters, false);
        let (_, nolb) = onesweep_sort_inner(&keys, 5, iters, true);
        println!(
            "| {label:>7} | {full:>12.3} | {nolb:>14.3} | {:>8.1}% |",
            (full - nolb) / full * 100.0
        );
    }
}
