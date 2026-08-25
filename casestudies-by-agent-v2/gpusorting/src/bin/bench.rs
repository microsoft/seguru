//! Radix-sort benchmark: SeGuRu vs CUB vs Thrust vs a single-core CPU sort.
//!
//! All GPU timings are kernel-only; allocation and host transfers happen once,
//! outside the timed loop. Radix sort is data oblivious, so every implementation
//! does exactly the same amount of work regardless of the input distribution.

use gpusorting_gpu::cuda_ffi::{CudaSort, CudaSorter};
use gpusorting_gpu::radix_sort_timed;
use std::time::Instant;

const WARMUP: usize = 5;

struct Row {
    label: &'static str,
    n: usize,
    sg: f64,
    cub: f64,
    thrust: f64,
    cpu: Option<f64>,
}

fn gkeys(n: usize, ms: f64) -> f64 {
    n as f64 / (ms * 1e-3) / 1e9
}

fn iters_for(n: usize) -> usize {
    if n <= 1 << 20 {
        200
    } else if n <= 1 << 24 {
        50
    } else {
        20
    }
}

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
    let sizes: &[(&str, usize)] = &[
        ("64 Ki", 1 << 16),
        ("1 Mi", 1 << 20),
        ("4 Mi", 1 << 22),
        ("16 Mi", 1 << 24),
        ("64 Mi", 1 << 26),
        ("256 Mi", 1 << 28),
    ];

    let mut rows = Vec::new();

    for &(label, n) in sizes {
        let keys = lcg(0x5EED, n);
        let iters = iters_for(n);

        let mut expected = keys.clone();
        let cpu = if n <= 1 << 24 {
            let t = Instant::now();
            expected.sort_unstable();
            Some(t.elapsed().as_secs_f64() * 1e3)
        } else {
            expected.sort_unstable();
            None
        };

        let (sg_sorted, sg_ms) = radix_sort_timed(&keys, WARMUP, iters);
        assert!(sg_sorted == expected, "SeGuRu sort is wrong at n = {n}");

        let cuda = CudaSorter::new(&keys);
        assert!(
            cuda.sorted(CudaSort::Cub) == expected,
            "CUB baseline is wrong at n = {n}"
        );
        let cub_ms = cuda.bench(CudaSort::Cub, WARMUP as u32, iters as u32);
        let thrust_ms = cuda.bench(CudaSort::Thrust, WARMUP as u32, iters as u32);

        println!("  n = {label:>7} ({n:>10}) done");
        rows.push(Row {
            label,
            n,
            sg: sg_ms,
            cub: cub_ms,
            thrust: thrust_ms,
            cpu,
        });
    }

    println!("\n32-bit key sort, A100. Times are milliseconds for one full sort.\n");
    println!(
        "| {:>7} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>7} |",
        "keys", "SeGuRu ms", "CUB ms", "Thrust ms", "CPU ms", "SG Gkeys/s", "SG / CUB", "vs CPU"
    );
    println!(
        "|{:->9}|{:->12}|{:->12}|{:->12}|{:->12}|{:->12}|{:->10}|{:->9}|",
        "", "", "", "", "", "", "", ""
    );
    for r in &rows {
        let cpu_s = r.cpu.map(|c| format!("{c:.2}")).unwrap_or("-".into());
        let vs_cpu = r
            .cpu
            .map(|c| format!("{:.0}x", c / r.sg))
            .unwrap_or("-".into());
        println!(
            "| {:>7} | {:>10.3} | {:>10.3} | {:>10.3} | {:>10} | {:>10.2} | {:>8.2} | {:>7} |",
            r.label,
            r.sg,
            r.cub,
            r.thrust,
            cpu_s,
            gkeys(r.n, r.sg),
            r.sg / r.cub,
            vs_cpu
        );
    }
    println!("\nSG / CUB < 1 means SeGuRu is faster than cub::DeviceRadixSort.");
}
