# GPUSorting — LSD radix sort in safe Rust on SeGuRu

A 32-bit unsigned-integer LSD radix sort, written from scratch in safe Rust and
compiled to PTX with SeGuRu. No `unsafe` appears anywhere except `src/cuda_ffi.rs`,
which exists only to call the CUDA reference implementation for benchmarking and is
gated behind the `bench` feature.

Ported from the algorithm in
[b0nes164/GPUSorting](https://github.com/b0nes164/GPUSorting) (`DeviceRadixSort`).

## Algorithm

Four passes over the keys, one per 8-bit digit. Each pass is three kernels:

| kernel | file | grid | what it does |
| --- | --- | --- | --- |
| `radix_upsweep` | `src/upsweep.rs` | one block per tile | histogram the tile's digits into shared memory, then atomically accumulate into a global `[RADIX][num_tiles]` table, transposed so the scan is coalesced |
| `radix_scan` | `src/scan.rs` | one block per digit | exclusive scan of that digit's row, giving each tile its global offset for that digit |
| `radix_downsweep` | `src/downsweep.rs` | one block per tile | re-rank the tile's keys locally, then scatter them to their final global positions |

`src/clear.rs` zeroes the histogram between passes; `src/driver.rs` is the host-side
loop that chains the four passes and ping-pongs the two key buffers.

The downsweep is where all the work is. Each of 512 threads holds 15 keys in
registers. Ranking uses a warp-level multi-split: for each of the 8 bits of the
digit, `ballot_sync` produces the set of lanes whose bit matches, and intersecting
those eight ballots yields, for each key, the mask of lanes in the warp holding the
same digit. `lowest_set_bit` on that mask identifies the warp-local leader, and
`popcount` of the mask below the lane gives the rank within the digit. Warp
subtotals are then combined by a full-width shuffle scan (`src/utils.rs`), keys are
placed into a shared-memory tile in sorted order so the final global write is
coalesced, and the tile is written out.

## Results

NVIDIA A100 80GB PCIe, CUDA 13.3, kernel-only timing, random `u32` keys, median of
several iterations after warm-up. CUB is `cub::DeviceRadixSort::SortKeys`, Thrust is
`thrust::sort`, CPU is a single-threaded LSD radix sort on the host.

| keys | SeGuRu (ms) | CUB (ms) | Thrust (ms) | CPU (ms) | SeGuRu Gkeys/s | SeGuRu/CUB | vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 Ki | 0.094 | 0.077 | 0.092 | 0.95 | 0.70 | 1.21x | 10x |
| 1 Mi | 0.152 | 0.116 | 0.331 | 17.59 | 6.92 | 1.31x | 116x |
| 4 Mi | 0.386 | 0.231 | 0.470 | 76.62 | 10.88 | 1.67x | 199x |
| 16 Mi | 1.357 | 0.642 | 0.920 | 328.24 | 12.36 | 2.11x | 242x |
| 64 Mi | 5.006 | 2.228 | 2.743 | — | 13.41 | 2.25x | — |
| 256 Mi | 19.610 | 8.760 | 9.931 | — | 13.69 | 2.24x | — |

Reproduce with:

```bash
source ../env.sh
cargo test  --release -p gpusorting-gpu --lib -- --test-threads=1
cargo run   --release -p gpusorting-gpu --features bench --bin bench
```

## Optimisation log

`nsys` attributes **77%** of SeGuRu's sort time to the downsweep (upsweep 17%, scan
5%), so every experiment below targets that kernel. Experiments A–C were done by
timing; D–F were driven by diffing the generated PTX in
`target/release/deps/gpu/*.ptx` against an instruction histogram.

Two facts from that PTX shape everything below.

**SeGuRu's bounds check is branch-free.** Rather than a compare-and-branch to a
panic, an indexed load lowers to a six-instruction clamp-to-null:

```
cvt.u64.u32   %rd, %r_idx;
setp.gt.u64   %p,  %rd_len, %rd;      // in range?
mul.wide.u32  %rd_off, %r_idx, 4;
selp.b64      %rd_base, %rd_base, 0, %p;   // out of range -> null
selp.b64      %rd_off,  %rd_off,  0, %p;
add.s64       %rd_addr, %rd_base, %rd_off;
```

That is why an out-of-range access surfaces as `CUDA_ERROR_ILLEGAL_ADDRESS`
instead of a panic, and it means the cost of a bounds check is a few ALU ops on a
non-critical path, not a branch.

**Predicate traffic in the downsweep is mostly not bounds checks.** `setp` and
`selp` together are 28% of the kernel's 2 017 instructions, which looks damning
until you attribute them: only 32 `setp`s belong to bounds checks. The bulk —
140 `setp.eq.s32` and 121 `setp.ne.s32` against immediates — come from the warp
multi-split ballot loop, which is algorithmic and would be emitted by CUDA too.
Chasing "bounds-check overhead" here would have been chasing the wrong 4%.
Experiments D and E are the record of doing exactly that.

### A. Scatter straight from registers, skipping the shared-memory tile — rejected

Writing each key to its final global address directly from the register that holds
it removes an entire shared-memory round trip. It made the sort **2.4x slower**.
The shared tile exists to make the global writes coalesced; within a warp the 32
keys being written are contiguous after ranking, so one transaction serves the
whole warp. Scattering from registers gives each lane an unrelated address and
serialises the write. Coalescing dominates. Reverted.

### B. Replace the atomic global scatter with a plain structured store — measurement only

SeGuRu requires data-dependent writes to go through `gpu::sync::Atomic`, which
lowers to `atom.global.exch.b32`. To find out what that costs, the scatter was
temporarily rewritten as a `chunk_mut` store with a static map. The result is
*wrong* — the map does not express the permutation — but it is a valid timing of
the same memory traffic without the atomics:

| 256 Mi keys | time |
| --- | ---: |
| atomic scatter (correct) | 24.5 ms |
| non-atomic store (incorrect, timing only) | 17.4 ms |

**The mandatory atomic is about 40% of total sort time.** This is the single
largest, and the most interesting, cost of SeGuRu's safety guarantee in this
benchmark, because in this algorithm the scatter is provably a permutation: every
destination index is written exactly once. The type system simply has no way to say
so. A "provably disjoint scatter" primitive — an API that consumes a proof or a
witness that the index map is injective and then emits an ordinary `st.global` —
would recover most of the gap to CUB without giving up memory safety. Reverted;
the shipped code is the correct atomic version.

### C. Fully unroll the global scatter loop — kept

The final loop writes 15 keys per thread. Replacing the counted loop with a fully
unrolled `crunchy::unroll!` body lets the backend issue all 15 addresses
independently instead of serialising on the induction variable:

| 256 Mi keys | time | SeGuRu/CUB |
| --- | ---: | ---: |
| before | 24.5 ms | 2.79x |
| after | 19.5 ms | 2.23x |

A 20% improvement for a local change, and it is in the shipped code. Note that
`crunchy::unroll!` requires **literal** bounds — `unroll! { for i in 0..15 { ... } }`
— a `const` will not do.

### D. Pad shared memory to a power of two and mask every shared index — no effect

In the PolyBench port, guarding a buffer once with `a.len() >= N` and then
indexing with `idx & (N - 1)` lets LLVM prove every subsequent access in range and
delete the check (see `polybench/src/bin/boundscheck.rs`). The same trick was
applied here: the shared tile was padded to 8 192 words and every shared index
masked with `& (8192 - 1)`.

| 256 Mi keys | time | `setp` count |
| --- | ---: | ---: |
| before | 19.491 ms | 335 |
| after | 19.578 ms | 320 |

**It does not transfer to dynamic shared memory.** The PolyBench version works
because the buffer is a `&[f32]` parameter whose length is a runtime value LLVM
can reason about relative to the mask. A dynamic shared allocation comes back from
`smem_alloc.alloc::<u32>(n)` with an opaque length, so masking narrows the index
without ever giving the optimiser the corresponding fact about the extent, and the
comparison survives. Fifteen `setp`s went away and the time did not move. Reverted.

### E. Make the radix shift a const generic — rejected, 9% slower

The ballot loop's per-bit sequence is `and %r, bit, 31` / `shl.b32 1, %r` /
`and key, mask` / two complementary `setp` / `ballot` / `selp` / `xor`. The first
two exist only because `radix_shift` is a runtime argument. Turning it into a const
generic should fold them into immediates:

```rust
pub fn radix_downsweep<const SHIFT: u32>(/* ... */) { /* ... */ }
// host side:
radix_downsweep::launch::<SHIFT, _, _>(&config, /* ... */)?;
```

(The generated `launch` is `<UserConst, Config, CN>`, hence the two `_`s. Worked
example in `examples/syntax/`.) The four passes were then unrolled so each gets its
own monomorphisation. Tests passed, and it was **9% slower**:

| 256 Mi keys | time | instructions in downsweep |
| --- | ---: | ---: |
| runtime shift | 19.49 ms | 2 018 |
| const-generic shift | 21.32 ms | 2 143 |

The instruction count went *up*. Four copies of a 2 000-instruction kernel cost
more in instruction-cache pressure than the folded shifts save, and each
monomorphisation is scheduled independently, perturbing register allocation. D and
E were isolated from one another to confirm the mask was neutral and the const
generic alone caused the regression. Both reverted.

**Generalisation:** on a large kernel, const-generic specialisation is a pessimism
unless the constant unlocks a structural simplification, not just an immediate.

### F. Cut the tile from 15 to 8 keys per thread — kept

The PTX work in D and E was a dead end, but `ptxas -v` on the same files gave the
real answer. The downsweep uses **93 registers with zero spills**. At 512
threads/block that is 93 × 512 = 47 616 registers, well over the 32 768 available
per SM, so **only one block is resident per SM — 25% occupancy**. The earlier note
in this file that shared memory was the limit was wrong; registers are.

Registers scale with keys per thread, because the tile is held live in registers
across the ranking phase. Shrinking `BIN_PART_SIZE` from 7 680 to 4 096:

| keys/thread | tile | registers | blocks/SM |
| ---: | ---: | ---: | ---: |
| 15 | 7 680 | 93 | 1 |
| 8 | 4 096 | 56 | 2 |
| 4 | 2 048 | 38 | 3 |

Two constants must move together. `PART_SIZE` (the upsweep's tile) and
`BIN_PART_SIZE` (the downsweep's) index the *same* per-tile histogram array, so
they must be equal; changing only `BIN_PART_SIZE` produces silent corruption.

The 4-keys variant additionally exposed a latent assumption. The downsweep reuses
one shared allocation first for the per-warp histograms (`BIN_HISTS_SIZE` = 4 096
words) and then as the scatter tile (`BIN_PART_SIZE` words), but sized it as
`BIN_PART_SIZE + RADIX` — correct only while the tile is the larger of the two. At
2 048 keys the histogram overran the allocation. The shipped code now sizes it as
`SMEM_WORDS = max(BIN_PART_SIZE, BIN_HISTS_SIZE) + RADIX`, which makes the tile
size a genuinely free parameter.

Results, all with 9/9 tests passing:

| keys | 15/thread | **8/thread** | 4/thread |
| ---: | ---: | ---: | ---: |
| 64 Ki | 1.38x | **1.21x** | 1.19x |
| 1 Mi | 1.45x | **1.31x** | 1.45x |
| 4 Mi | 1.75x | **1.67x** | 2.13x |
| 16 Mi | 2.18x | **2.11x** | 2.79x |
| 64 Mi | 2.26x | **2.25x** | 3.16x |
| 256 Mi | 2.22x | **2.24x** | 3.25x |

(ratios vs CUB, lower is better)

Eight keys per thread is the optimum and is shipped. The gain is real but
size-dependent: **12% at 1 Mi keys and 12% at 64 Ki**, tapering to nothing by
64 Mi. That shape is the tell. Small sorts are latency-bound, so the second
resident block fills issue slots that were previously idle. Large sorts are bound
by DRAM bandwidth and by the atomic scatter, and extra occupancy cannot create
bandwidth. Four keys per thread buys a third block but doubles the number of
tiles, which doubles upsweep and scan work and the global atomic traffic — 45%
slower at 256 Mi.

**Generalisation:** occupancy tuning pays off exactly where the kernel is
latency-bound. Check `ptxas -v` for the register count before assuming shared
memory is the occupancy limit — and note that neither D nor E would have been
attempted had `ptxas -v` been read before the PTX instruction histogram.

## Why the remaining 2.2x gap to CUB

Two independent reasons, in rough order of size:

1. **Algorithm.** CUB uses OneSweep: decoupled look-back lets it make a single pass
   over the data per digit, whereas upsweep+downsweep reads the keys twice per
   digit. That alone is close to a 2x difference in memory traffic. Whether
   OneSweep is expressible in safe SeGuRu is an open question — it needs
   acquire-semantics atomic loads and a spin-wait on a neighbour's flag, and the
   spin-wait is inside divergent control flow, which is exactly what SeGuRu's
   `sync_data` analysis rejects.

   `nsys` kernel counts make the structural difference concrete: for the same
   workload CUB launched its histogram kernel **1 146** times against our
   **2 280** upsweeps. CUB computes all four digit histograms in a single pass —
   legal because a global histogram is order-invariant — and then needs no
   per-tile histogram pass at all. We cannot copy just that half: our `pass_hist`
   is *per tile*, and tile membership changes after every permutation, so passes
   2–4 genuinely cannot be precomputed. Closing this requires decoupled look-back,
   not a fused histogram.
2. **The atomic scatter tax** quantified in experiment B above, about 40% of the
   remaining time.

Neither is a code-quality problem in this port; both are properties of the
programming model. That is the useful finding.

## SeGuRu notes specific to this case study

* `lowest_set_bit` is implemented with `u32::trailing_zeros()`. That did not
  previously work — `cttz` was missing from the intrinsic table in
  `crates/rustc_codegen_gpu/src/builder/intrinsic.rs`, and the earlier generation of
  this benchmark worked around it with a 32-iteration serial loop run 15 times per
  thread. The intrinsic is now wired up; the loop is gone.
* Warp shuffle scans work fine inside conditionals **provided the guard is widened
  to a warp boundary**. The inter-warp scan uses `if tid < 32` and feeds identity
  values to the lanes that have no real data, rather than `if tid < 8`. A narrower
  guard trips `Invalid use of diversed data in GPU code`. `src/utils.rs` documents
  this contract at the top of the file.
* The `reshape_map!` stride-override tuple `(D, TD)` only contributes a stride to
  the dimensions that come **after** it in the `layout:` permutation. On the last
  dimension it merely bounds the index, and an out-of-range id yields an invalid
  pointer — a runtime `CUDA_ERROR_ILLEGAL_ADDRESS`, not a panic. `src/upsweep.rs`
  and `src/scan.rs` are worked examples with and without a meaningful override.
