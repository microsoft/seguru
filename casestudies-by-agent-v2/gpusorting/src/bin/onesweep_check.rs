//! Correctness check for the OneSweep port.
//!
//! `cargo run --release -p gpusorting-gpu --bin onesweep_check`

use gpusorting_gpu::onesweep_sort;

fn lcg(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        })
        .collect()
}

fn check(name: &str, keys: Vec<u32>) {
    let mut expected = keys.clone();
    expected.sort_unstable();
    let got = onesweep_sort(&keys);
    if got == expected {
        println!("  {name:<34} n = {:>9}  OK", keys.len());
    } else {
        let bad = got
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        println!(
            "  {name:<34} n = {:>9}  MISMATCH at {bad}: got {:?} want {:?}",
            keys.len(),
            &got[bad..(bad + 4).min(got.len())],
            &expected[bad..(bad + 4).min(expected.len())]
        );
        std::process::exit(1);
    }
}

fn main() {
    check("single partition", lcg(1, 4096));
    check("ragged, not a multiple of tile", lcg(2, 12345));
    check("multi partition", lcg(3, 1 << 20));
    check("already sorted", (0..1u32 << 16).collect());
    check("reverse sorted", (0..1u32 << 16).rev().collect());
    check("all equal", vec![7u32; 1 << 16]);
    check(
        "extremes",
        vec![0, u32::MAX, 0, u32::MAX, 1, u32::MAX - 1]
            .into_iter()
            .cycle()
            .take(1 << 16)
            .collect(),
    );
    check("high bits only", lcg(5, 1 << 18).iter().map(|k| k & 0xFF00_0000).collect());
    check("large", lcg(6, 1 << 24));
    println!("all OK");
}
