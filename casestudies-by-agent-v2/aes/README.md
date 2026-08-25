# AES-128 ECB on SeGuRu

A from-scratch SeGuRu port of GPU AES-128 in ECB mode, with a CUDA C++
reference for comparison.

## What is here

| File | Purpose |
|---|---|
| `src/tables.rs` | S-box, T-tables and key schedule, all derived at compile time from the GF(2^8) field definition |
| `src/lib.rs` | The two SeGuRu kernels (`aes128_encrypt`, `aes128_decrypt`) and their host helpers |
| `src/cpu.rs` | Scalar CPU AES-128 used as the correctness oracle and CPU baseline |
| `src/tests.rs` | FIPS-197 vectors plus GPU-vs-CPU cross-checks across sizes |
| `cuda/aes_ref.cu` | CUDA C++ reference: a mirror of the SeGuRu kernel plus a textbook baseline |
| `src/cuda_ffi.rs` | Safe wrapper over the CUDA reference (the only `unsafe` in the crate) |
| `src/bin/bench.rs` | Benchmark driver |

## Kernel design

* **Vectorised I/O.** Plaintext and ciphertext are typed `[U32_4]`, so each
  16-byte AES block moves in one 128-bit access instead of four 32-bit ones.
* **Four AES blocks per thread.** The unroller interleaves the four independent
  block computations, hiding shared-memory latency without extra occupancy.
* **One T-table, not four.** `TE1..TE3` are byte rotations of `TE0`, so only
  1 KiB of shared memory is staged and the prologue is a single load per
  thread. The rotations are a shift/or pair, which is free next to the
  shared-memory traffic.
* **Round keys in shared memory**, staged once per CTA instead of being
  re-read from global memory every round. The host pads the schedule to
  `BLOCK_DIM` words so the staging load is uniform across the CTA — the SeGuRu
  divergence analysis rejects a `chunk_mut` inside a divergent branch.
* **Branch-free tail.** The host pads device buffers to a whole number of
  `grid * BLOCK_DIM * BLOCKS_PER_THREAD` AES blocks, so the kernel needs no
  tail predicate and the `reshape_map!` chunk proves per-thread disjointness of
  all writes.

No `unsafe` appears in the kernels or their host code. The only `unsafe` in the
crate is in `src/cuda_ffi.rs`, which exists purely to call the CUDA C++
reference from the benchmark.

## Running

```bash
source ../env.sh          # LLVM 20 + CUDA on PATH
cargo test  --release -p aes-gpu-v2 --lib
cargo run   --release -p aes-gpu-v2 --features bench --bin aes-bench
```

## Measured results (A100 80GB PCIe, CUDA 13.3, LLVM 20)

Kernel-only time; host/device transfers happen once outside the timed loop.
"CUDA mirror" is `cuda/aes_ref.cu`'s `aes128_encrypt_opt`, an
instruction-for-instruction translation of the SeGuRu kernel. "CUDA classic"
is the textbook formulation used by most CUDA AES implementations: one AES
block per thread with the four T-tables in `__constant__` memory.

### Encryption

| Size | SeGuRu (µs) | CUDA mirror (µs) | CUDA classic (µs) | SG/CUDA | SeGuRu GB/s | CPU (µs) | GPU speedup |
|---|---|---|---|---|---|---|---|
| 16 KiB | 16.3 | 16.1 | 104.3 | 1.02× | 1.0 | 213 | 13× |
| 1 MiB | 16.7 | 16.4 | 421.6 | 1.01× | 62.9 | 12,336 | 740× |
| 16 MiB | 140.7 | 140.7 | 5,496.2 | 1.00× | 119.3 | — | — |
| 256 MiB | 2,073.2 | 2,078.9 | 89,319.7 | 1.00× | 129.5 | — | — |
| 1 GiB | 8,294.3 | 8,301.8 | 362,603.4 | 1.00× | 129.5 | — | — |

### Decryption

| Size | SeGuRu (µs) | CUDA mirror (µs) | SG/CUDA | SeGuRu GB/s |
|---|---|---|---|---|
| 16 KiB | 14.8 | 14.5 | 1.02× | 1.1 |
| 1 MiB | 15.1 | 14.8 | 1.02× | 69.2 |
| 16 MiB | 123.7 | 123.6 | 1.00× | 135.6 |
| 256 MiB | 1,818.0 | 1,816.5 | 1.00× | 147.7 |
| 1 GiB | 7,263.4 | 7,243.7 | 1.00× | 147.8 |

### Reading the numbers

* **SeGuRu reaches parity with equivalent CUDA** on both encryption (1.00–1.02×)
  and decryption (1.00–1.02×), with bounds checking enabled.

  > **Correction.** An earlier revision of this file claimed SeGuRu was *12%
  > faster on decryption*. That was a benchmark bug, not a result. The CUDA
  > mirror was decrypting the **plaintext** buffer while SeGuRu decrypted real
  > **ciphertext**, and AES is data-dependent: the T-table index *is* the data
  > byte, so the shared-memory bank-conflict rate depends on the input
  > distribution. The synthetic plaintext is `byte[i] = i*31 ^ (i>>8)`, so
  > indices of consecutive lanes differ by `16*31 = 240 (mod 256)`; because
  > `240 % 32 == 16`, an entire warp collides onto **two** of the 32 banks in the
  > first round, against ~uniform spread for pseudorandom ciphertext. One
  > heavily-conflicted round in ten accounts for the ~12%.
  >
  > Verified by decrypting both buffers with the *same* SeGuRu kernel:
  >
  > | Size | dec(ciphertext) | dec(plaintext) | old CUDA number |
  > |---|---:|---:|---:|
  > | 16 MiB | 123.9 µs | 141.1 µs | 141.0 µs |
  > | 1 GiB | 7 266.7 µs | 8 309.9 µs | 8 309.8 µs |
  >
  > `dec(plaintext)` reproduces the old CUDA figure exactly. `aes_ref.cu` now
  > encrypts once and swaps buffers so both sides decrypt real ciphertext.
  > The corrected answer is **parity**.

  The general lesson: when a safe-language port appears to *beat* hand-written
  CUDA, suspect the harness first. This is the second such artifact found in this
  suite (see also the `atax`/`bicg`/`mvt` launch-shape note in
  `polybench/README.md`).
* **The algorithm matters far more than the language.** The textbook CUDA
  kernel is 44× slower at 1 GiB, because `__constant__` memory only broadcasts
  when every lane in a warp reads the same address; AES T-table lookups are
  lane-divergent by construction, so each warp access serialises into 32
  transactions. Staging the table in shared memory is the fix, and SeGuRu's
  `GpuShared` expresses it safely.
* **This kernel is shared-memory bound, not HBM bound.** At 1 GiB the kernel
  moves ~260 GB/s of HBM traffic (read + write) against an A100's ~1.9 TB/s,
  while issuing ~43 GB of shared-memory traffic per GiB of plaintext. The
  ceiling is bank conflicts on the T-table, which affects the CUDA reference
  identically.
