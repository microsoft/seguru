//! Radix-sort benchmark: SeGuRu against three CUDA baselines and a CPU sort.
//!
//! The column that actually measures SeGuRu is **DRS-ours**: the upstream
//! reduce-then-scan `DeviceRadixSort.cu` that our kernels transliterate,
//! recompiled at our port's tile size. Same algorithm, same tuning, same launch
//! geometry — the only difference is the compiler and the safety checks.
//!
//! **DRS-up** is the same CUDA kernels at upstream's own tuning (7680 keys per
//! tile against our 4096). The DRS-up / DRS-ours gap is a tuning cost that has
//! nothing to do with SeGuRu.
//!
//! **CUB** is kept for context but is *not* like-for-like: on CUDA 13.3 / sm_80
//! it dispatches `DeviceRadixSortOnesweepKernel`, a different algorithm that
//! reads the keys once per digit instead of twice.
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
    drs_up: f64,
    drs_ours: f64,
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
        assert!(
            cuda.sorted(CudaSort::DrsUpstreamTuning) == expected,
            "same-algorithm baseline (upstream tuning) is wrong at n = {n}"
        );
        assert!(
            cuda.sorted(CudaSort::DrsOurTuning) == expected,
            "same-algorithm baseline (our tuning) is wrong at n = {n}"
        );
        let cub_ms = cuda.bench(CudaSort::Cub, WARMUP as u32, iters as u32);
        let thrust_ms = cuda.bench(CudaSort::Thrust, WARMUP as u32, iters as u32);
        let drs_up_ms = cuda.bench(CudaSort::DrsUpstreamTuning, WARMUP as u32, iters as u32);
        let drs_ours_ms = cuda.bench(CudaSort::DrsOurTuning, WARMUP as u32, iters as u32);

        println!("  n = {label:>7} ({n:>10}) done");
        rows.push(Row {
            label,
            n,
            sg: sg_ms,
            cub: cub_ms,
            thrust: thrust_ms,
            drs_up: drs_up_ms,
            drs_ours: drs_ours_ms,
            cpu,
        });
    }

    println!("\n32-bit key sort, A100. Times are milliseconds for one full sort.");
    println!("DRS-ours is the same algorithm AND same tuning as SeGuRu, in CUDA C++.\n");
    println!(
        "| {:>7} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>8} | {:>7} |",
        "keys",
        "SeGuRu ms",
        "DRS-ours",
        "DRS-up",
        "CUB ms",
        "Thrust ms",
        "CPU ms",
        "SG/DRS-o",
        "SG / CUB",
        "vs CPU"
    );
    println!(
        "|{:->9}|{:->11}|{:->11}|{:->11}|{:->11}|{:->11}|{:->11}|{:->11}|{:->10}|{:->9}|",
        "", "", "", "", "", "", "", "", "", ""
    );
    for r in &rows {
        let cpu_s = r.cpu.map(|c| format!("{c:.2}")).unwrap_or("-".into());
        let vs_cpu = r
            .cpu
            .map(|c| format!("{:.0}x", c / r.sg))
            .unwrap_or("-".into());
        println!(
            "| {:>7} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9} | {:>9.2} | {:>8.2} | {:>7} |",
            r.label,
            r.sg,
            r.drs_ours,
            r.drs_up,
            r.cub,
            r.thrust,
            cpu_s,
            r.sg / r.drs_ours,
            r.sg / r.cub,
            vs_cpu
        );
    }
    println!("\nSG/DRS-o is the cost of SeGuRu: same algorithm, same tuning, safe Rust");
    println!("against CUDA C++. SG/CUB additionally includes the cost of not using");
    println!("onesweep, and DRS-up/DRS-ours is the cost of our smaller tile size.");
    println!(
        "\nGkeys/s (SeGuRu): {}",
        rows.iter()
            .map(|r| format!("{:.2}", gkeys(r.n, r.sg)))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
