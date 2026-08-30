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

### B. Replace the atomic scatter with a plain structured store — SUPERSEDED, the atomic is free

SeGuRu requires data-dependent writes to go through `gpu::sync::Atomic`. An early
version of this section reported that removing it took 256 Mi from 24.5 ms to
17.4 ms and concluded "the mandatory atomic is about 40% of total sort time".
**That conclusion is wrong and has been withdrawn.**

It was retested properly once the toolchain bug that blocked the safe non-atomic
scatter was fixed (`SharedAtomic::new` was missing `ret_sync_data(0)`, which
tainted the `&mut GpuShared` handle permanently). Two checks, both negative:

* **PTX diff.** The generated PTX is *identical* with and without the shared
  `Atomic` — 29 `atom`/`red` instructions and 16 `st.shared.u32` either way.
  `atomic_assign` on a shared-memory location already lowers to a plain
  `st.shared.u32`; it never emitted an atomic instruction to begin with.
* **Wall clock.** 256 Mi keys: **19.606 ms with the atomic, 19.603 ms without.**

The original 24.5 -> 17.4 measurement is not reproducible and was most likely
taken under GPU contention (see the caveat in `FINDINGS.md`) or against a variant
that dropped work as well as the atomic. The shipped code keeps the atomic
version, which is both correct and free. The real gap to CUB is diagnosed in
"Why the remaining 2.2x gap to CUB" below, and the atomic is no part of it.


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

Measured, not estimated: `nsys profile -t cuda --stats=true` on `sort-bench`.
The per-kernel maxima below are the 256 Mi run.

| kernel | ms | traffic | achieved BW | % of 1549 GB/s peak |
| --- | --- | --- | --- | --- |
| our `radix_upsweep` | 1.18 | 1.07 GB read | 913 GB/s | 59% |
| our `radix_scan` | 0.50 | 0.13 GB | — | — |
| our `radix_downsweep` | 3.47 | 2.15 GB read+write | 618 GB/s | 40% |
| CUB `DeviceRadixSortHistogramKernel` | 0.78 | 1.07 GB read | 1376 GB/s | 89% |
| CUB `DeviceRadixSortOnesweepKernel` | 2.06 | 2.15 GB read+write | 1040 GB/s | 67% |

These reconstruct both sorts to within 5%: ours is `(1.18 + 0.50 + 3.47) x 4 =
20.6 ms` against 19.61 ms measured, CUB is `0.78 once + 2.06 x 4 = 9.0 ms`
against 8.74 ms. The 11.6 ms gap splits almost exactly in half.

1. **5.92 ms (51%) is algorithmic.** CUB dispatches
   `DeviceRadixSortOnesweepKernel` — it is running *onesweep*, not the algorithm
   we ported. It pays one global histogram (0.78 ms, legal because a global
   histogram is order-invariant) and then fuses scatter with decoupled look-back.
   We pay `upsweep + scan` on every pass: 1.68 ms x 4 = 6.7 ms. That is also why
   we move 12.9 GB of key traffic against CUB's 9.7 GB (1.33x) — the upsweep
   re-reads the whole key array each pass to build a histogram the downsweep then
   rebuilds locally anyway.

   We cannot copy just the fused-histogram half: our `pass_hist` is *per tile*,
   and tile membership changes after every permutation, so passes 2-4 genuinely
   cannot be precomputed. Closing this needs decoupled look-back. An earlier
   version of this file called that **an open question**, guessing it needed an
   acquire-ordered atomic load of the shape `sync_data` rejects. It has since
   been written (`src/onesweep.rs`) and SeGuRu accepts it — see "Onesweep" below.

2. **5.63 ms (49%) is our downsweep running 1.69x slower than CUB's scatter**
   (618 vs 1040 GB/s) while issuing *identical* traffic. `ptxas -arch=sm_80 -O3
   -v` names the cause: 56 registers at `DOWNSWEEP_THREADS = 256` is 14 336
   registers per block, so only `65536 / 14336 = 4` blocks fit per SM — 1024 of
   2048 threads, **50% occupancy**, with no spills. Shared memory is not the
   binding constraint (`SMEM_WORDS = 4352` u32 = 17 408 B/block would allow 9
   blocks in A100's 164 KB). 51 registers would buy a 5th block (62.5%), 42 a
   sixth (75%).

Neither half is a cost of safe Rust. The algorithmic half is inherited from the
CUDA file we ported and an `unsafe` port of the same file would pay it too; the
occupancy half is a register-allocation outcome, and the inner loop's PTX
contains no bounds checks or safety instrumentation to blame it on.

Note this supersedes the earlier claim that an "atomic scatter tax" was worth
about 40% of the time. It is worth nothing at all — see experiment B above.

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


## Onesweep

`src/onesweep.rs` is a safe-Rust port of upstream's `OneSweep.cu`: a global
histogram, a seeding scan, and four fused binning passes with decoupled
look-back. It passed all nine cases in `bin/onesweep_check.rs` on its first run.
Run it under `timeout` — a wrong look-back hangs the device rather than failing.

`cuda/upstream/OneSweep.cu` is vendored (MIT) and `cuda/os_variant.cu` compiles
it twice, at upstream's 7680-key tile and at our 4096-key tile, so `bin/bench.rs`
can print a same-algorithm, same-tuning ratio (`OS ratio`) alongside `RS ratio`.
**`CUB ratio` is not a like-for-like number and should not be quoted as one.**

At 256 Mi keys the port costs 1.75x the same CUDA onesweep, against a flat 1.15x
for reduce-then-scan; below 1 Mi keys it is 1.7-2.6x *faster*. `bin/onesweep_lb.rs`
localises the gap: restart each tile's backwards walk at slot 0 (seeded
`FLAG_INCLUSIVE` by the scan, so it terminates immediately) and the publishes and
reads are unchanged but no tile waits. That removes 45% of the runtime and puts
the safe port at 12.76 ms against the CUDA baseline's 13.31 ms.

So the whole gap is the spin, and the cause is one missing primitive.
`crates/gpu/src/sync.rs` has no atomic load and no CAS, so the poll is written
`atomic_ori(0)` — an RMW returning the old value — which lowers to
`atom.global.or.b32`. CUDA's `volatile` load lowers to `ld.volatile.global.u32`.
An RMW must take the L2 sector exclusively, so pollers serialise against each
other and against the publisher. An `atomic_load` lowering to
`ld.relaxed.gpu.global.u32` would be race-free and should recover most of it.

Two traps worth knowing, both of which cost a debugging session:

* Pre-seeding the flag array `FLAG_INCLUSIVE` to short-circuit the spin **hangs
  the GPU**. The tile's own `atomic_addi` publish then lands on an already-tagged
  slot, the flag field wraps to 3, and successors match neither branch.
* Upstream stashes its acquired partition index in
  `s_warpHistograms[BIN_PART_SIZE - 1]`, which aliases warp 15 / bin 255 as soon
  as `BIN_PART_SIZE <= BIN_HISTS_SIZE`. It is correct at 7680 and wild at 4096.
  Given a dedicated `__shared__` slot here, marked `LOCAL MODIFICATION`.
