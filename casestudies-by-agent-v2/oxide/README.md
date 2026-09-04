# SeGuRu vs. cuda-oxide

An apples-to-apples comparison against [cuda-oxide][oxide], NVIDIA's
experimental Rust-to-CUDA compiler, on three kernels ported from the
case studies in this directory:

| benchmark | ported from | why this one |
|---|---|---|
| `polybench-gemm` | `polybench/src/gemm.rs` | shared-memory tiling, the standard case for a blocked kernel |
| `polybench-atax` | `polybench/src/atax.rs` | warp-level reduction, no shared memory |
| `sort-upsweep` | `gpusorting/src/upsweep.rs` | shared-memory histogram with atomics |

Both sides use the same input generator, the same problem sizes, the same
iteration counts and the same kernel-only timing loop, so the rows line up
directly. Correctness is checked on every run.

## The finding

**cuda-oxide has no safe shared-memory API.** Its own documentation states
that "all shared memory access requires `unsafe`" because shared memory is
uninitialized at kernel start and concurrently accessed
(`crates/cuda-device/src/shared.rs:41-46`); a `SharedArray` lives in a
`static mut`, so every read and write is an `unsafe` block. This is not an
oversight in our ports: cuda-oxide's own `gemm_views` example, whose entire
subject is proof-carrying safe views over global memory, still writes its
shared tiles inside `unsafe`.

Everything else in cuda-oxide can be safe, and we used the safe path
throughout: global reads and writes through `DisjointSlice` with a
hardware-minted index witness, `DeviceAtomicU32`, the warp primitives,
`sync_threads`, and even kernel launch via `#[launch_contract]`. Only
`kernels::load` is unconditionally `unsafe` on the host.

The consequence is visible in the numbers:

- **gemm** — the *safe* cuda-oxide kernel cannot stage tiles in shared
  memory, and is **2.2-2.5x slower** than SeGuRu. Adding shared memory makes
  it **1.1-1.2x faster** than SeGuRu, but costs two `unsafe` blocks.
- **radix_upsweep** — a shared-memory histogram **cannot be written safely at
  all** in cuda-oxide. The port needs seven `unsafe` blocks to reach the
  performance SeGuRu gets with none.
- **atax** — needs no shared memory, so both sides are fully safe and the
  performance is comparable. This is the control.

SeGuRu reaches shared-memory performance with **zero `unsafe`** in every
kernel.

## Results

A100 80GB PCIe, kernel-only microseconds, all runs verified against a CPU
reference. `cuda` is the hand-written CUDA baseline already present in the
polybench case study.

### gemm

| size | SeGuRu | cuda | oxide (safe) | oxide (shared mem) |
|---|---|---|---|---|
| 512^3 | 63.2 | 57.5 | 159.4 | 53.8 |
| 1024^3 | 289.0 | 278.7 | 593.6 | 253.4 |
| 2048^3 | 1775.5 | 1755.3 | 3477.6 | 1562.8 |
| 4096^3 | 12971.9 | 13066.5 | 26233.0 | 11438.1 |

### atax

| size | SeGuRu | cuda | oxide (safe) |
|---|---|---|---|
| 2048^2 | 220.5 | 127.3 | 154.2 |
| 4096^2 | 495.6 | 638.4 | 688.5 |
| 8192^2 | 1071.3 | 1359.6 | 1468.2 |

### radix_upsweep

| size | SeGuRu | oxide (needs `unsafe`) |
|---|---|---|
| 2^20 | 9.10 | 9.38 |
| 2^22 | 23.87 | 24.18 |
| 2^24 | 91.54 | 90.73 |
| 2^26 | 331.71 | 322.46 |

### Device code size and `unsafe`

Device code only, blank and comment lines excluded (see `metrics.py`).

| kernel | SeGuRu LoC | SeGuRu `unsafe` | oxide LoC | oxide `unsafe` |
|---|---|---|---|---|
| gemm | 73 | 0 | 153 | 2 |
| atax | 59 | 0 | 67 | 0 |
| radix_upsweep | 120 | 0 | 128 | 7 |

## Fairness

The comparison is only meaningful if the two sides do the same work through
the same memory path, so:

- **Vector loads.** SeGuRu's kernels use 128-bit `Float4` / `U32_4` loads.
  The cuda-oxide ports use `#[repr(C, align(16))]` POD quad types to get the
  same `ld.global.v4`. Reading scalars instead made the upsweep port 2.6-3.7x
  slower, which would have been an artifact of the port rather than a
  property of cuda-oxide.
- **One allocation per matrix.** `atax` reads A as quads in one kernel and as
  scalars in the other. SeGuRu aliases both views over a single tensor, so the
  port uses `DeviceBuffer::cast_chunks` to re-view the same allocation rather
  than uploading a second copy. Uploading twice doubled the working set past
  the 40MB L2 and made 2048^2 look 2.1x slower than it is.
- **Same timing methodology.** `WARMUP` untimed launches, then `iters` timed
  launches and a single synchronize, with `iters` from the same
  `iters_for_flops` / `iters_for_bytes` formulas as the SeGuRu benchmarks.
- **Same inputs.** Both sides use the identical xorshift generator, so the
  input bytes match exactly.

Numbers were taken on an otherwise idle GPU; the A100 in this machine is
shared, so check `nvidia-smi --query-compute-apps` before trusting a ratio.

## Running it

Requires a local [cuda-oxide][oxide] checkout (developed against commit
`97f8b2b7`, which needs the `nightly-2026-08-28` toolchain). Point
`CUDA_OXIDE` at it and `env.sh` will link it as `.cuda-oxide`, which is the
path each crate's dependencies name:

```sh
export CUDA_OXIDE=/path/to/cuda-oxide
./bench-compare.sh          # both sides + metrics, into results/
./bench-oxide.sh            # just cuda-oxide, CSV on stdout
./bench-oxide.sh gemm atax  # a subset
./metrics.py                # LoC and unsafe counts
```

A single benchmark, with the build output visible:

```sh
source env.sh
cd polybench-gemm && cargo oxide run
```

`results/` is generated and git-ignored.

### Why the extra config files

- `env.sh` sets `CUDA_OXIDE_BACKEND` to the prebuilt
  `librustc_codegen_cuda.so`. Without it, a standalone crate re-clones and
  rebuilds the whole backend from git.
- `.cargo/config.toml` resets `rustc` to the stock driver. The parent
  `casestudies-by-agent-v2/.cargo/config.toml` sets `rustc = "rustc-gpu"`,
  which makes every cuda-oxide build die with `couldn't load codegen backend:
  cannot allocate memory in static TLS block`.
- Each crate carries its own `rust-toolchain.toml`, because cuda-oxide and
  SeGuRu pin different nightlies.

[oxide]: https://github.com/NVlabs/cuda-oxide
