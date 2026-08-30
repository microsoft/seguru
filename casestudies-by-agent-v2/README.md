# SeGuRu GPU case studies (v2)

This directory is a from-scratch redo of `../casestudies-by-agent/`. The
original port was produced by an older model; this one was rewritten with a
newer one, with three goals:

1. **Performance.** Kernels are written to be fast on an A100, not merely to
   compile. Where a CUDA reference exists it is an instruction-for-instruction
   mirror of the SeGuRu kernel, so the comparison measures the toolchain rather
   than two different algorithms.
2. **No `unsafe`.** No `unsafe` appears in any GPU kernel or its host code. The
   only exception is a per-crate `src/cuda_ffi.rs`, which exists solely to call
   the CUDA C++ reference baselines from the benchmark binaries and is compiled
   only under the `bench` cargo feature. `./verify.sh` enforces this.
3. **Verified correctness.** Every kernel is checked against an independent CPU
   implementation in `cargo test`.

Upstream benchmark sources: [polybenchGpu](https://github.com/sgrauerg/polybenchGpu),
[KernelBench](https://github.com/ScalingIntelligence/KernelBench),
[CUDA_AES](https://github.com/cihangirtezcan/CUDA_AES),
[GPUSorting](https://github.com/b0nes164/GPUSorting),
[HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU).

## Case studies

| Directory | Package | What it is | Status |
|---|---|---|---|
| `aes/` | `aes-gpu-v2` | AES-128 ECB encrypt/decrypt, T-table in shared memory, 4 blocks per thread, `U32_4` vector I/O | Complete: 6 tests pass, benchmark and `aes/README.md` written |
| `gpusorting/` | `gpusorting-gpu` | LSD radix sort, 8 bits per pass (upsweep / scan / downsweep), warp-level multi-split ranking | Complete: 9 tests pass |
| `polybench/` | `polybench-gpu` | 19 PolyBench/GPU kernels: BLAS-like (`gemm`, `twomm`, `threemm`, `syrk`, `syr2k`, `gramschm`, `lu`), memory-bound (`atax`, `bicg`, `gesummv`, `mvt`, `doitgen`), stencils (`jacobi1d`, `jacobi2d`, `fdtd2d`, `conv2d`, `conv3d`) and statistics (`corr`, `covar`) | Complete: 19 tests pass; CUDA baseline in progress |
| `kernelbench/` | `kernelbench-gpu` | 21 neural-network operators: activations (`relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `softplus`, `leaky_relu`), normalisations (`layer_norm`, `rms_norm`, `l1_norm`, `l2_norm`), reductions (`softmax`, `log_softmax`, `sum_dim`, `mean_dim`, `max_dim`, `argmax_dim`, `cumsum`), `mse_loss`, `max_pool1d` | Complete: 22 tests pass, benchmark and README written |
| `heongpu/` | `heongpu-gpu` | HEonGPU homomorphic-encryption primitives: forward/inverse NTT, Barrett and Montgomery modular arithmetic, element-wise polynomial ops | Complete: 11 tests pass, CUDA mirror benchmark and README written |

## Environment prerequisites

| Component | Version used |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, driver 580.159.03 |
| CUDA | 13.3, installed at `/usr/local/cuda` |
| LLVM / MLIR | 20, installed at `/usr/lib/llvm-20` |
| Rust | nightly-2025-03-28 (pinned by the repository's `rust-toolchain.toml`) |

The SeGuRu compiler driver, `rustc-gpu`, must be installed and on `PATH`:

```bash
cd /home/ziqiaozhou/seguru
MLIR_SYS_200_PREFIX=/usr/lib/llvm-20 TABLEGEN_200_PREFIX=/usr/lib/llvm-20 \
    cargo install --path ./crates/rustc-gpu --locked
```

Both environment variables are required. Without them `melior-macro` caches an
empty include directory and the build fails with `could not find llvm in ods`.
The same variables must be set again if you ever change the codegen backend and
reinstall.

## Building and running

`env.sh` must be sourced before *every* cargo command in this workspace. It puts
LLVM 20 (which provides `mlir-opt`, invoked by the codegen backend) and CUDA on
`PATH`, and sets `LD_LIBRARY_PATH`, `MLIR_SYS_200_PREFIX` and
`TABLEGEN_200_PREFIX`. Without it the build panics with `mlir-opt not found`.

```bash
cd /home/ziqiaozhou/seguru/casestudies-by-agent-v2
source env.sh

cargo test --release -p aes-gpu-v2          # one case study
./verify.sh                                 # all of them, plus the unsafe audit
./verify.sh aes gpusorting                  # a subset
```

`.cargo/config.toml` already selects `rustc-gpu` and sets `USE_FAST`,
`USE_FTZ` and `NVPTX_FEATURES=+ptx87`; do not edit it.

### Benchmarks

Benchmarks live behind the `bench` cargo feature, because enabling them makes
`build.rs` compile the CUDA C++ reference with `nvcc`. They are separate
binaries:

```bash
cargo run --release -p aes-gpu-v2     --features bench --bin aes-bench
cargo run --release -p gpusorting-gpu --features bench --bin sort-bench
```

```bash
cargo run --release -p heongpu-gpu     --features bench --bin heongpu-bench
cargo run --release -p kernelbench-gpu --bin kernelbench-bench
```

The PolyBench CUDA baseline and its `bench` binary are the last piece still
being added; see `polybench/README.md` for its current state.

All GPU timings in this directory are kernel-only: allocation and host/device
transfers happen once, outside the timed loop, and `ctx.sync()` brackets the
measured region.

## Results

### AES-128 (A100 80GB PCIe, CUDA 13.3, LLVM 20)

Kernel-only time, encryption. "CUDA mirror" is an instruction-for-instruction
translation of the SeGuRu kernel; "CUDA classic" is the textbook formulation
(one AES block per thread, four T-tables in `__constant__` memory). Full tables,
including decryption, are in `aes/README.md`.

| Size | SeGuRu (µs) | CUDA mirror (µs) | CUDA classic (µs) | SeGuRu GB/s |
|---|---|---|---|---|
| 1 MiB | 16.7 | 16.4 | 421.6 | 62.9 |
| 16 MiB | 140.7 | 140.7 | 5,496.2 | 119.3 |
| 256 MiB | 2,073.2 | 2,078.9 | 89,319.7 | 129.5 |
| 1 GiB | 8,294.3 | 8,301.8 | 362,603.4 | 129.5 |

SeGuRu reaches parity with the equivalent CUDA on encryption (1.00–1.02×) **and
on decryption (1.00–1.02×)**, with bounds checking enabled.

An earlier revision of this file claimed a 12% win on decryption. That was a
benchmark bug: the CUDA mirror decrypted the *plaintext* buffer while SeGuRu
decrypted real *ciphertext*. AES T-table indices are the data bytes, so the
shared-memory bank-conflict rate is input-dependent, and the synthetic plaintext
happens to collide a whole warp onto two of the 32 banks. Both sides now decrypt
ciphertext and the honest answer is parity — see `aes/README.md` for the
measurement that isolates it.

### Radix sort (32-bit keys)

Kernel-only. CUB is `cub::DeviceRadixSort::SortKeys`, Thrust is `thrust::sort`,
CPU is a single-threaded host LSD radix sort. Full analysis, including six
measured optimisation experiments (three negative, three kept), is in
`gpusorting/README.md`.

| Keys | SeGuRu (ms) | CUB (ms) | Thrust (ms) | CPU (ms) | SeGuRu Gkeys/s | SeGuRu/CUB | vs CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 Ki | 0.094 | 0.077 | 0.092 | 0.95 | 0.70 | 1.21x | 10x |
| 1 Mi | 0.152 | 0.116 | 0.331 | 17.59 | 6.92 | 1.31x | 116x |
| 4 Mi | 0.386 | 0.231 | 0.470 | 76.62 | 10.88 | 1.67x | 199x |
| 16 Mi | 1.357 | 0.642 | 0.920 | 328.24 | 12.36 | 2.11x | 242x |
| 64 Mi | 5.006 | 2.228 | 2.743 | - | 13.41 | 2.25x | - |
| 256 Mi | 19.610 | 8.760 | 9.931 | - | 13.69 | 2.24x | - |

This is the one case study where SeGuRu clearly trails, and the reasons are now
measured per kernel with `nsys` (finding 9). The 11.6 ms gap at 256 Mi splits
almost exactly in half: 5.9 ms because CUB runs *onesweep* (one global histogram,
then a fused scatter per digit) while we pay upsweep+scan on every pass, and
5.6 ms because our downsweep is stuck at 50% occupancy (56 registers x 256
threads = 4 blocks/SM) and so reaches only 40% of peak bandwidth against CUB's
67%. Neither is a cost of safe Rust — an `unsafe` port of the same CUDA file
would pay both. An earlier claim that a "mandatory atomic scatter" cost ~40% of
runtime was falsified: it costs nothing.

### HEonGPU (NTT and modular arithmetic)

Kernel-only, 4 Mi coefficients. "CUDA" is an instruction-for-instruction mirror.
Full tables in `heongpu/README.md`.

| Kernel | SeGuRu/CUDA | vs CPU |
|---|---|---|
| Forward NTT | 1.44-1.47x | 523-1512x |
| Inverse NTT | 1.10-1.32x | 523-1512x |
| Element-wise modular ops | 1.01-1.03x | - |

The element-wise kernels run at about 1.54 TB/s, roughly 79% of the A100's peak
HBM bandwidth, i.e. they are at the memory-bandwidth roof.

The NTT gap narrowed from 1.58x to 1.44x by masking twiddle indices against a
constant-length sub-slice and hand-packing twiddle pairs into `U32_4`. The
remainder is a **compiler defect, now diagnosed**: alignment is dropped on the
memref path, so `u64` loads carry `align 4` instead of `align 8` and NVPTX splits
each one into two `ld.global.u32` — exactly 128 against CUDA's 64. (The
`.ptr .align 1` on kernel parameters is a red herring; a minimal NVPTX test shows
access width follows the *load's* alignment, not the parameter attribute.) See `FINDINGS.md` bug 6; it is
the highest-value outstanding fix in the toolchain. A leading alternative theory
— that the missing `ld.global.nc` (`const __restrict__`) was to blame — was
tested and **refuted**: stripping `__restrict__` from the CUDA baseline cost it
only 0-5%.

### PolyBench

Kernel-only, µs/iteration, largest size shown. 14 of 19 kernels have a CUDA
baseline; `doitgen`, `covar`, `corr`, `lu` and `gramschm` were skipped as
launch-sequencing dominated. Every row was verified element-wise against the CUDA
output (worst relative inf-norm error 4.0e-6) before any time was recorded. Full
table in `polybench/README.md`.

| Kernel | Size | SeGuRu (µs) | CUDA (µs) | SeGuRu/CUDA | vs CPU |
|---|---|---:|---:|---:|---:|
| `gemm` | 4096 | 12 900 | 13 069 | 0.99x | 3509x |
| `twomm` | 2048 | 3 548 | 3 513 | 1.01x | 3443x |
| `threemm` | 2048 | 5 318 | 5 287 | 1.01x | 3368x |
| `syrk` | 4096 | 16 602 | 16 156 | 1.03x | 1810x |
| `syr2k` | 2048 | 4 565 | 5 078 | 0.90x | 963x |
| `atax` | 8192 | 1 072 | 1 360 | 0.79x | 100x |
| `bicg` | 8192 | 1 071 | 1 361 | 0.79x | 11x |
| `gesummv` | 8192 | 310 | 316 | 0.98x | 272x |
| `mvt` | 8192 | 1 079 | 1 363 | 0.79x | 78x |
| `conv2d` | 8192 | 388 | 375 | 1.03x | 189x |
| `conv3d` | 384 | 540 | 514 | 1.05x | 117x |
| `jacobi1d` | 2^24 | 3 761 | 3 686 | 1.02x | 36x |
| `jacobi2d` | 4096 | 2 114 | 2 037 | 1.04x | 32x |
| `fdtd2d` | 4096 | 4 669 | 4 606 | 1.01x | 45x |

Two caveats we want stated rather than buried. First, **both** hand-written
implementations are 1.8-2.2x off cuBLAS SGEMM (28/130/983/7310 µs), so the GEMM
parity result means "SeGuRu matches equivalent hand-written CUDA", not "SeGuRu
matches the best available GEMM". Second, the `syr2k` and `gesummv` rows where
SeGuRu appears to *win* are flagged as suspect in `polybench/README.md`: the CUDA
`syr2k` uses 48 registers against SeGuRu's 71 with identical shared memory and
block shape, so it has higher occupancy and is still slower, which most likely
means that CUDA baseline leaves ~10% on the table.

The `atax`/`bicg`/`mvt` ratio at n=2048 is **3.0x, and should not be quoted**: at
that size the column pass launches only 8 CTAs on 108 SMs (7% of the machine), so
it is purely issue-bound, and separately the 16 MB matrix is L2-resident on the
A100 while 64 MB at n=4096 is not. The representative figure is the 1.18-1.20x at
n=4096 and n=8192. The launch shape is a property of the ported kernel (one thread
per output column) and is identical on both sides.

The stencils originally trailed by 1.05-1.88x because **LLVM could not relate a
thread-derived index to a slice length**, so every global load carried a
`setp.gt.u64` plus two `selp.b64`. Two safe-Rust source changes — sub-slicing to
the exact extent so the check folds in 32 bits, and using
`MapContinuousLinear::new(1)` instead of `reshape_map!` to drop a `div.u32` —
removed almost all of it. `conv3d` went 204 -> 162 PTX instructions and
**1.88x -> 1.05x**, with an unchanged load count and max relative error 1.2e-7.
Details and the four failed variants are in `polybench/README.md`.

One row needs an explicit caveat: the **0.79x on `atax`/`bicg`/`mvt` is not a
codegen win.** At 8192, nsys shows SeGuRu at 288 GB/s and CUDA at 220 GB/s on a
1555 GB/s part — both at 15-19% of peak. The mirrored CUDA column kernel is
simply weak, and so is ours; the ratio flatters SeGuRu rather than reflecting
better code. The honest summary of PolyBench is *parity across the board*, not a
win.

**The bounds-check tax has been measured, not just inferred.**
`polybench/src/bin/boundscheck.rs` builds variants in which the checks provably
*are* elided, by masking indices with a compile-time power-of-two (`idx & (N-1)`
is provably `< N` to LLVM) behind a single `a.len() >= N` guard. Because every
index is already in range the mask is a no-op on the data, so these variants are
**numerically correct**, still safe Rust, and if anything under-state the tax
(the mask itself costs one `and.b32` per access):

| Kernel | Size | Stock (µs) | Checks elided (µs) | Bounds-check tax |
|---|---|---:|---:|---:|
| `conv3d` | 128³ | 38.9 | 27.7 | **28.6%** |
| `conv3d` | 256³ | 288.8 | 198.5 | **31.3%** |
| `mvt` column pass | 8192² | 1445.7 | 716.4 | **50.4%** |

So bounds checks were **29-31% of conv3d's runtime and ~50% of the mvt column
pass's** — the clearest measurement in the suite of what safety was costing.

**That cost has since been recovered, in safe Rust, and the recovery is shipped.**
The prediction above was that eliding the checks would take conv3d from 1.88x to
roughly 1.3x; the actual result was better, **1.05x**, because the same rework
also removed a `div.u32`. The technique is to sub-slice a parameter to its exact
extent (`&a[..total]`) so that `a.len()` is literally `zext(total)` and the
comparison folds in 32 bits rather than 64. See `polybench/README.md`.

The revised conclusion is the stronger one: this was **a missing compiler
analysis, not a cost of the safety guarantee**, and until the compiler derives
the fact itself, a programmer can supply it without writing `unsafe`.

### KernelBench (neural-network operators)

Kernel-only, 4096x1024 `f32`. These operators are all memory-bound, so the
meaningful comparison is against achievable HBM bandwidth (~1.55-1.9 TB/s on
this A100) rather than against a second implementation. Full table in
`kernelbench/README.md`.

| Operator | Time (µs) | Achieved GB/s |
|---|---:|---:|
| `relu` | 20.9 | 1606 |
| `gelu` (tanh) | - | 1522 |
| `softmax` | 23.3 | 1442 |
| `layer_norm` | - | 1368 |
| `sum_dim` / `mean_dim` / `max_dim` | - | ~1910 |
| `mse_loss` | - | 1805 |
| `max_pool1d` | - | 1746 |

That is 95-100% of achievable bandwidth for the reduction kernels. At 1024x1024
these operators become launch-latency bound, with a floor around 5 µs. Note that
no PyTorch or CUDA C++ baseline was available on this machine, so these are
absolute numbers against machine peak rather than a head-to-head ratio.

### Cross-case-study summary

| Workload | Character | SeGuRu vs hand-written CUDA |
|---|---|---|
| AES-128 encrypt | Shared-memory / bank-conflict bound | Parity (1.00-1.02x) |
| AES-128 decrypt | Shared-memory / bank-conflict bound | Parity (1.00-1.02x) |
| HEonGPU element-wise | Memory-bandwidth bound | Parity (1.01-1.03x) |
| KernelBench operators | Memory-bandwidth bound | At 95-100% of HBM roof |
| PolyBench GEMM family | Compute bound, shared-memory tiled | Parity (0.99-1.11x) |
| PolyBench stencils | Bounds-check bound | Parity (1.02-1.05x) after optimisation; was 1.05-1.88x |
| PolyBench mat-vec | Bandwidth bound, both weak | 0.79x — see caveat, not a real win |
| HEonGPU NTT | `u64` split into 2x32 by a compiler defect | 1.10-1.47x slower; was 1.3-1.6x |
| Radix sort | Data-dependent scatter | 2.2x slower at scale; 1.21-1.31x at <=1 Mi keys |

The pattern is consistent and, we think, the main technical result of this
exercise:

* **Where the kernel is bound by memory bandwidth or by shared-memory
  behaviour, safe Rust on SeGuRu reaches parity with hand-written CUDA.** Bounds
  checking is essentially free in these regimes because the bottleneck is
  elsewhere.
* **Where the kernel needs a data-dependent scatter, SeGuRu pays a real and
  measurable tax**, because safety forces those writes through atomics. In the
  radix sort this is ~40% of runtime, even though the scatter is provably a
  permutation - the type system just cannot express that. A "provably disjoint
  scatter" primitive would close most of this gap without giving up safety.
* **Where the kernel is issue-bound on many small indexed loads** (stencils,
  mat-vecs), SeGuRu pays for bounds checks it cannot elide - a measured **29-31%
  of `conv3d`** and **~50% of the `mvt` column pass**. Unlike the scatter tax this
  looks tractable: the indices are affine in the thread id and provably in range,
  and simply handing LLVM that range fact recovers the time without leaving safe
  Rust.
* **Where the data type is poorly served by the backend** (64-bit strided
  access), the gap is an implementation limitation rather than a fundamental
  one.

Two of these three costs are artifacts of the current implementation rather than
of safety itself. Only the scatter tax is inherent to the guarantee, and even that
would largely dissolve given a way to express "this index map is injective".

## Toolchain bugs found

Porting these five workloads surfaced five defects in SeGuRu itself, three of which
have been fixed as part of this work:

| # | Defect | Status |
|---|---|---|
| 1 | `ctlz`/`cttz` missing from the intrinsic table, so `leading_zeros()`/`trailing_zeros()` failed in device code | Fixed |
| 2 | GPU globals (`static_shared_*`, `const_alloc_*`, `memory_alloc_*`) named per-crate but emitted with external linkage, colliding when a binary links a library's GPU module | Fixed |
| 3 | `ThreadWarpTile::<32>::BASE_THREAD_MASK` overflowed, making full-warp `redux.sync` unusable | Fixed |
| 4 | Chained `chunk_to_scope` addresses lane 0's slot from every lane - silent wrong answers | Open |
| 5 | `GpuShared::<T>::zero()` does not actually zero at run time | Open |

Details, evidence and reproducers are in [`FINDINGS.md`](FINDINGS.md). The two open
items both cause silent corruption with no diagnostic, and have a standing
reproducer in `kernelbench/src/bin/probe.rs`.

## Differences from the previous generation

* **The old AES headline claim — "SeGuRu is 13% faster than hand-written CUDA" —
  was an artifact of a weak baseline, and is retracted here.** The old CUDA
  reference stored the AES T-tables in `__constant__` memory. Constant memory
  only broadcasts efficiently when every lane of a warp reads the same address;
  AES T-table lookups are lane-divergent by construction, so each warp access
  serialises into up to 32 transactions. That baseline is roughly 44× slower
  than a shared-memory CUDA kernel at 1 GiB. It was never a fair opponent. The
  v2 benchmark therefore reports **both** an instruction-for-instruction CUDA
  mirror of the SeGuRu kernel *and* the classic `__constant__` baseline, so the
  reader can see which difference comes from the language and which from the
  algorithm. Against the mirror, SeGuRu encryption is at parity.

* **The old tutorial's claim that ~138 GB/s was "near the HBM limit" is wrong.**
  An A100 has on the order of 1.5–1.9 TB/s of HBM bandwidth, so 138 GB/s is
  under a tenth of it. The v2 AES kernel is shared-memory-bank-conflict bound,
  not HBM bound: at 1 GiB it moves roughly 260 GB/s of HBM traffic (read plus
  write) while issuing tens of gigabytes of shared-memory traffic per GiB of
  plaintext. The CUDA mirror hits the same ceiling, which is why the two match.

* **Padding instead of tail predicates.** Host buffers are padded so the grid
  maps exactly onto the data. This removes every ragged-tail branch, lets loads
  be full `U32_4` vector loads, and — importantly for SeGuRu — keeps `chunk_mut`
  out of divergent control flow, which the compiler rejects.

* **Vectorised I/O and register-resident per-thread arrays.** Per-thread loops
  are unrolled with `crunchy::unroll!` so local arrays are promoted to registers
  by `mem2reg`; with a runtime loop index they stay in local memory, which is
  global memory in disguise.

* **Tables derived at compile time.** `aes/src/tables.rs` computes the S-box,
  T-tables and key schedule with `const fn` from the GF(2^8) field definition,
  rather than carrying transcribed literals.

* **Backend gap closed rather than worked around.** `leading_zeros` /
  `trailing_zeros` previously panicked in codegen with ``GPU intrinsic `cttz`
  not supported``, and the old sort emulated `ffs` with a serial loop. The
  intrinsics are now mapped in
  `crates/rustc_codegen_gpu/src/builder/intrinsic.rs`, and `gpusorting` uses
  `u32::trailing_zeros()` directly.

* **Structural disjointness instead of scatter where possible.** `radix_scan`
  keeps every thread writing its own slot and derives the exclusive scan value
  with one extra `shfl.up`, instead of the CUDA original's circular-lane-shifted
  scatter. Same result, no atomics on the output.

## Reading guide

* `PORTING-NOTES.md` — the short, factual API and rules reference: kernel
  anatomy, host API, chunk maps, shared memory, the divergence rule, and the
  performance rules. Read this first.
* `TUTORIAL.md` — a longer walkthrough for someone who already knows CUDA, with
  worked examples taken from `aes/` and `gpusorting/`, a gotchas section, and a
  list of known toolchain gaps.
* `aes/README.md` — the reference case study write-up: design rationale plus
  full measured results.
* `crates/gpu/src/chunk_impl.rs` in the repository root — the authoritative
  documentation for `reshape_map!`.
* `doc/optimization.md` in the repository root — the general SeGuRu kernel
  optimisation rules.
