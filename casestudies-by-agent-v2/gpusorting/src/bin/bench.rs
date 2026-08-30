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
//! **SG-OS** is the SeGuRu onesweep port. Its like-for-like baseline is
//! **OS-ours** — upstream's `OneSweep.cu`, the file it transliterates, rebuilt
//! at our tile size. CUB is *also* onesweep on CUDA 13.3 / sm_80, but it is a
//! heavily tuned production implementation, so `SG-OS / CUB` measures tuning
//! effort as much as it measures SeGuRu.
//!
//! **SG-RS** is the SeGuRu reduce-then-scan port, which reads the keys twice per
//! digit; compare it against DRS-ours, not against CUB.
//!
//! All GPU timings are kernel-only; allocation and host transfers happen once,
//! outside the timed loop. Radix sort is data oblivious, so every implementation
//! does exactly the same amount of work regardless of the input distribution.

use gpusorting_gpu::cuda_ffi::{CudaSort, CudaSorter};
use gpusorting_gpu::{onesweep_sort_timed, radix_sort_timed};
use std::time::Instant;

const WARMUP: usize = 5;

struct Row {
    label: &'static str,
    n: usize,
    sg: f64,
    os: f64,
    os_ours: f64,
    os_up: f64,
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

        let (os_sorted, os_ms) = onesweep_sort_timed(&keys, WARMUP, iters);
        assert!(os_sorted == expected, "SeGuRu onesweep is wrong at n = {n}");

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
        assert!(
            cuda.sorted(CudaSort::OneSweepOurTuning) == expected,
            "onesweep baseline (our tuning) is wrong at n = {n}"
        );
        assert!(
            cuda.sorted(CudaSort::OneSweepUpstreamTuning) == expected,
            "onesweep baseline (upstream tuning) is wrong at n = {n}"
        );
        let os_ours_ms = cuda.bench(CudaSort::OneSweepOurTuning, WARMUP as u32, iters as u32);
        let os_up_ms = cuda.bench(CudaSort::OneSweepUpstreamTuning, WARMUP as u32, iters as u32);

        println!("  n = {label:>7} ({n:>10}) done");
        rows.push(Row {
            label,
            n,
            sg: sg_ms,
            os: os_ms,
            os_ours: os_ours_ms,
            os_up: os_up_ms,
            cub: cub_ms,
            thrust: thrust_ms,
            drs_up: drs_up_ms,
            drs_ours: drs_ours_ms,
            cpu,
        });
    }

    println!("\n32-bit key sort, A100. Times are milliseconds for one full sort.");
    println!("Each SeGuRu column sits next to CUDA C++ running the SAME algorithm");
    println!("at the SAME tile size, so both ratios isolate the cost of SeGuRu.\n");
    println!(
        "| {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
        "keys",
        "SG-RS",
        "DRS-ours",
        "RS ratio",
        "SG-OS",
        "OS-ours",
        "OS ratio",
        "OS-up",
        "CUB",
        "Thrust"
    );
    println!(
        "|{:->9}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|{:->10}|",
        "", "", "", "", "", "", "", "", "", ""
    );
    for r in &rows {
        println!(
            "| {:>7} | {:>8.3} | {:>8.3} | {:>8.2} | {:>8.3} | {:>8.3} | {:>8.2} | {:>8.3} | {:>8.3} | {:>8.3} |",
            r.label,
            r.sg,
            r.drs_ours,
            r.sg / r.drs_ours,
            r.os,
            r.os_ours,
            r.os / r.os_ours,
            r.os_up,
            r.cub,
            r.thrust
        );
    }
    println!("\nRS ratio and OS ratio are the two same-algorithm, same-tuning ratios.");
    println!("OS-up is upstream's own tuning; CUB is production-tuned onesweep, so");
    println!("SG-OS/CUB measures tuning effort as much as it measures SeGuRu.");
    println!(
        "\nGkeys/s (SG onesweep): {}",
        rows.iter()
            .map(|r| format!("{:.2}", gkeys(r.n, r.os)))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let _ = |r: &Row| (r.cpu, r.drs_up);
}
