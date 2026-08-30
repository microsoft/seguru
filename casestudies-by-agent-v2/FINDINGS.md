# Toolchain bugs and limitations found by this exercise

Rewriting all five case studies from scratch, and then optimising the three
slowest against their CUDA originals, surfaced six defects in SeGuRu itself.
Three are fixed here; three are reported with standing reproducers. This list is
arguably a more useful output of the exercise than the benchmark numbers, because
each item was found by a *port attempt* or a *PTX diff* rather than by reading
the code.

Bugs 1-3 were found while porting. Bug 6 — the highest-value one — was found only
by diffing generated PTX against the CUDA baseline's, which is a good argument
for making that diff a routine step rather than a debugging tactic.

## Fixed

### 1. `ctlz` / `cttz` missing from the intrinsic table

**Symptom.** `u32::leading_zeros()` and `u32::trailing_zeros()` in device code
failed with `GPU intrinsic 'cttz' not supported`.

**Cause.** `crates/rustc_codegen_gpu/src/builder/intrinsic.rs` mapped `sym::ctpop`
to `melior_math::ctpop`, but had no entry for the count-leading/trailing-zeros
intrinsics, even though the corresponding MLIR ops exist.

**Fix.** Added four entries next to `ctpop`:

```rust
sym::ctlz         => melior_math::ctlz, 1,
sym::ctlz_nonzero => melior_math::ctlz, 1,
sym::cttz         => melior_math::cttz, 1,
sym::cttz_nonzero => melior_math::cttz, 1,
```

**Why it mattered.** The previous generation of this benchmark suite worked around
the gap with a 32-iteration serial loop, executed 15 times per thread in the radix
sort's hottest kernel. `gpusorting/src/utils.rs::lowest_set_bit` is now a single
instruction.

### 2. `static_shared_N` symbol collision across crates

Found by the KernelBench port.

**Symptom.** A binary that launches a library kernel using static shared memory
failed to link: `static_shared_0: symbol multiply defined`. Confusingly, the AES
crate did the same thing and linked fine.

**Cause.** `crates/rustc_codegen_gpu/src/context/const_static.rs::define_static_shared_mem`
named the global `static_shared_{idx}` from a *per-module* counter and gave it
external linkage. An executable links its own GPU module together with the GPU
module of every extern crate, so both defined `static_shared_0`. `llvm-link`
tolerates a duplicate only when the two definitions have identical type — AES
linked purely by luck, because its library and binary shared buffers were both
`[1024 x i8]`. KernelBench's were `[128 x i8]` and `[1024 x i8]`. Confirmed with
`llvm-dis` on the emitted `.gpu.bc`.

**Fix.** Mangle the local crate's stable id into the symbol, so each crate gets its
own namespace. `const_static.rs` now has:

```rust
pub(crate) fn crate_unique_suffix(&self) -> String {
    format!("{:016x}", self.tcx.stable_crate_id(rustc_hir::def_id::LOCAL_CRATE).as_u64())
}
```

Verified by reverting KernelBench's wrapper workaround: the binary now links and
runs with the natural layout, and all 22 tests still pass.

**The same bug existed on two more code paths.** After the shared-memory fix, the
PolyBench benchmark port hit the identical failure for constant allocations:
`Linking globals named 'const_alloc_0': symbol multiply defined!`. All three GPU
global names — `static_shared_*`, `const_alloc_*` and `memory_alloc_*` — were
generated from per-crate counters or per-crate ids, and all three now carry the
crate-unique suffix. Two independent ports hitting the same class of bug on
different paths suggests the invariant "every emitted GPU global must be unique
across the whole link, not just within its module" is worth asserting centrally.

### 3. `ThreadWarpTile::<32>::BASE_THREAD_MASK` overflows

Found by the KernelBench port.

**Symptom.** `ThreadWarpTile::<32>` failed const-evaluation, making the integer
`redux.sync` path unusable at the one width that matters most: a full warp.

**Cause.** `crates/gpu/src/cg.rs:101` computed the mask as `(1u32 << SIZE) - 1`,
and `1u32 << 32` overflows.

**Fix.** Shift down from `u32::MAX` instead, which is correct for every supported
SIZE including 32:

```rust
pub const BASE_THREAD_MASK: u32 = { u32::MAX >> (32 - Self::CHECKED_SIZE) };
```

**Measured benefit.** With the full-warp `redux.sync` path unlocked, KernelBench's
`argmax_dim` was switched from a hand-rolled `shuffle!(xor, ...)` butterfly to the
hardware instruction, measured side by side in one process (20 warmup, 200 timed
launches, repeated):

| `argmax_dim` warp stage | 1024x1024 | 4096x1024 |
|---|---:|---:|
| `shuffle!(xor, ...)` butterfly | 6.3 µs | 13.9 µs |
| `redux.sync` (`ReduxMin`) | 6.0 µs | **12.8 µs** |

An 8% improvement at the larger shape. The index-recovery cost that usually makes
`redux.sync` a poor fit for argmax does not apply here: the row maximum is reduced
first, then each lane's *candidate index* is min-reduced, and the minimum over
candidate indices is the argmax — one instruction, no second pass.

## Open

Both of the following cause **silent wrong answers**, with no compile error, no
bounds-check failure and no runtime diagnostic. Standing reproducer:

```bash
cargo run --release -p kernelbench-gpu --bin probe
```

### 4. Chained `chunk_to_scope` addresses lane 0's slot from every lane

**Symptom.** Given a block-to-warp-to-thread staging chunk:

```rust
smem.chunk_to_scope(build_chunk_scope(Block, warp), MapContinuousLinear::new(1))
    .chunk_to_scope(build_chunk_scope(warp, Thread), MapContinuousLinear::new(1))
```

writing `s[0] = warp_value` should place warp `w`'s value in `smem[w]` from any
lane. Instead **only lane 0 addresses `smem[w]`; every other lane addresses
`smem[0]`**, i.e. warp 0's slot. With 256 threads writing `100*warp + lane`:

| who writes | result |
|---|---|
| all lanes | `[701, 100, 200, 300, 400, 500, 600, 700, 0, ...]` |
| only lane 0 | `[0, 100, 200, 300, 400, 500, 600, 700, 0, ...]` (correct) |
| only lane 31 | `[731, 100, 200, 300, 400, 500, 600, 700, 0, ...]` |

**Why it is dangerous.** The natural way to publish a warp's subtotal after an
inclusive scan is to write it from lane 31, which holds the total. Doing so stores
nothing to the intended slot and instead races on warp 0's slot. In KernelBench's
`cumsum` this presented as a small numeric discrepancy that looked like `f32`
rounding, not like memory corruption.

**Suggested diagnostic.** The per-thread view has length 1 while the scope it is
derived from has 32 ranks, and the map is static, so indexing from a thread with
non-zero rank in the parent scope is provably outside the intended slot and could
be rejected at compile time. Failing that, the chunk index should be bounds-checked
against the view length rather than folded into the base address.

### 5. `GpuShared::<T>::zero()` does not zero at run time *(fixed)*

**Symptom.** A shared buffer created with `zero()` can be observed holding stale
values from a previous kernel launch — visible in the `only lane 31` row above,
where slots the kernel never wrote still contain `100..700`.

**Cause.** `crates/gpu/src/shared.rs:29` is an intrinsic stub, lowered by
`define_static_shared_mem` to an MLIR `memref.global` carrying a dense-zero
initializer. CUDA `__shared__` memory has no static-initializer semantics — it is
uninitialized per thread block, and shared memory is reused between launches. The
zero initializer is therefore decorative at run time.

**Why it is dangerous.** The constructor's name is a promise the hardware does not
keep, and the failure is silent and non-deterministic: whether stale data is
observed depends on what ran before.

**Possible resolutions**, in rough order of preference — this is a design call, so
it is left to the maintainers rather than patched here:

1. Rename to `uninit()` and require callers to initialise explicitly. Honest, and
   free, but a breaking API change.
2. Have codegen emit a real cooperative zeroing loop followed by `sync_threads()`.
   Matches the name, but adds a barrier to every kernel that declares shared
   memory, which is a real cost in kernels that fully overwrite the buffer anyway.
3. Keep the name but document it loudly and lint uses that read a shared location
   before writing it.

**Fixed** (branch `ziqiao/gpushared-init`) via option 1 plus a safe wrapper:

* `unsafe fn GpuShared::<T>::uninit()` — the allocation intrinsic, renamed and
  marked `unsafe`, documenting that the caller must write before reading.
* `fn GpuShared::<[E; N]>::init::<PER_THREAD>(&mut self, v)` — safe cooperative
  fill across the block.
* `shared_init!(name: ty, per_thread, init)` — a statement macro bundling
  allocation, `init` and the barrier into one safe line.

The HEonGPU NTT tile buffers were migrated to `shared_init!`; 11/11 tests pass.


### 7. A chunked `GpuShared` handle cannot be returned from a function

**Symptom.** The natural constructor shape is rejected:

```rust
pub fn make<const PER_THREAD: usize>(v: E) -> Self {
    let mut this = unsafe { Self::uninit() };
    { let mut c = this.chunk_mut(MapLinear::new(1));
      for i in 0..PER_THREAD { c[i] = v; } }
    this
}
```

Every later `chunk_mut` in the *caller* then fails with
`error: Invalid use of diversed data in GPU code`. This is why the public API is
`init(&mut self)` plus a macro rather than the more natural
`let mut smem = GpuShared::init(...); sync_threads();`.

**Isolated.** The rejection is caused by returning the chunked handle itself, not
by anything else in the body:

| form | result |
|---|---|
| `fn new() -> Self`, runtime loop bound, barrier inside | `InvalidDiversedData` |
| `fn new() -> Self`, `const PER_THREAD`, barrier inside | `InvalidDiversedData` |
| `fn make() -> Self`, `const PER_THREAD`, barrier at call site | `InvalidDiversedData` |
| `fn make() -> Self` **+ `ret_sync_data(1000)`** | divergence OK; 2x shared alloc |
| `fn init(&mut self)`, `const PER_THREAD`, barrier at call site | **compiles, 11/11 pass** |

So neither the loop bound nor the barrier placement matters; only the move does.

**A second, independent symptom of the same restriction.** Moving the handle out of
a *block expression* (an expression-form macro) compiles, but the backend emits a
**second** shared allocation:

```
ptxas error: Entry function '...ntt_forward_tile...' uses too much shared data
             (0x10000 bytes, 0xc000 max)
```

0x10000 is exactly 2 x 32 KB for one `[u64; 4096]` tile. `uninit()` is
`#[inline(never)]` and its call site *is* the allocation, so binding the result and
moving it again materialises a whole second buffer rather than aliasing the first.

**Cause — two independent blockers, one of which is already solvable.**

*1. The divergence rejection is not a real limitation.* `TaintSourceDetector`
(`mir_analysis.rs:80-110`) conservatively taints the destination of **every**
`#[gpu_codegen::device]` call, because any device fn might call `thread_id`. It
exempts functions that declare their return value non-diversed:

```rust
let trusted_non_diversed = matches!(attr.gpu_item, ...)
    || attr.ret_sync_data.contains(&GpuAttributes::MAX_FN_IN_PARAMS);
```

`MAX_FN_IN_PARAMS` is `1000` and index `1000` denotes the return value
(`attr.rs:15-16, 43`). Annotating the constructor with
`#[gpu_codegen::ret_sync_data(1000)]` **removes the `InvalidDiversedData` error
entirely** — verified. A shared-memory handle really is uniform across the block,
so the annotation is sound rather than a mere silencer.

*2. The remaining blocker is the duplicated allocation.* The forked
`rustc_codegen_ssa` gives **every MIR local of type `GpuShared` its own shared
global**:

```rust
// rust/compiler/rustc_codegen_ssa/src/mir/place.rs:127-133
if Some(def.did()) == bx.cx().tcx().get_diagnostic_item(sym("gpu::GpuShared")) {
    return PlaceValue::alloca_shared(bx, size, layout.align.abi).with_type(layout);
}
```

and `alloca_shared` calls `define_static_shared_mem`, minting a fresh
`static_shared_N` every time (`builder/mod.rs:1956-1966`). A constructor produces
two such locals — the callee's temporary and the caller's binding — so the block
gets two 32 KB tiles. Neither `#[inline(always)]` nor `#[inline(never)]` avoids
this, and `-Zmir-enable-passes=+DestinationPropagation` ICEs the `gpu` crate.

**Fix required for `let mut smem = GpuShared::init(...);` to work.** Make a
`GpuShared` local allocate a shared global only when it is genuinely the result of
the `gpu::new_shared_mem` intrinsic, and make a move between `GpuShared` places
alias the existing global instead of allocating and copying. That is a change to
the vendored `rustc_codegen_ssa`, so it needs its own validation pass across all
five case studies.

**Impact until then.** Constructors that both allocate and initialise shared memory
cannot be ordinary functions. Every such API needs the `uninit()` + `&mut self` +
caller-side barrier shape, or a macro to hide it.

**Also related.** The analysis is intra-procedural for barriers, so a
`sync_threads()` inside a callee is not credited to the caller
(`MissingSyncThreads`) even with `#[inline(always)]` — which is why `init()` cannot
call it and the macro must.


### 6. Alignment is dropped on the memref path, splitting every `u64` global access

**Symptom.** In `ntt_forward_tile` — algorithmically identical to its CUDA
baseline — SeGuRu emits **128 `ld.global.u32` + 16 `st.global.u32` and zero
64-bit global accesses**, where CUDA emits **64 `ld.global.u64` + 8
`st.global.u64`**. Exactly 2x the global memory instructions. Total PTX is 1540
against 954 (1.61x), which tracks the 1.58x measured runtime ratio. Every 64-bit
datatype on the GPU pays this.

The emitted pairs are unmistakable — adjacent halves of one 64-bit value:

```
ld.global.u32 %rd21, [%rd20];
ld.global.u32 %rd22, [%rd20+4];
```

**Root cause: the load carries `align 4`, not the natural `align 8`.** This was
isolated with a minimal NVPTX test (`llc -march=nvptx64 -mcpu=sm_80`) varying only
the alignment on the load instruction:

| LLVM IR | emitted PTX |
| --- | --- |
| `load i64, ptr %p, align 1` | 8x `ld.global.u8` |
| `load i64, ptr %p, align 4` | **2x `ld.global.u32`** <- what SeGuRu produces |
| `load i64, ptr %p, align 8` | 1x `ld.global.u64` |

**Correction to an earlier diagnosis in this file.** The `.param .u64 .ptr
.align 1` seen on kernel parameters is a **red herring**. The same minimal test
shows that a kernel whose pointer parameters are declared `.ptr .align 1` still
emits `ld.global.u64`, because NVPTX derives access width from the *load
instruction's* alignment, not from the parameter attribute. Anyone chasing this
bug should ignore the parameter declaration entirely and look at the load.

**Where it goes wrong.** `Builder::mlir_load`
(`crates/rustc_codegen_gpu/src/builder/mod.rs:1046-1067`) threads the computed
alignment into `LoadStoreOptions::align` on the LLVM-pointer path and then
silently drops it on the memref fallback path:

```rust
// llvm-pointer path: alignment is honoured
melior::dialect::llvm::LoadStoreOptions::new()
    .align(Some(self.align_to_attr(align))),
...
// memref path: `align` is accepted as a parameter and then never used
self.append_op_res(melior::dialect::memref::load(ptr, indices, self.cur_loc()))
```

`store_with_check` (same file, ~line 1305) has the same shape.

**Why the obvious fix does not work.** Attaching `{alignment = 8 : i64}` to
`memref.load` parses, but MLIR 20's `--finalize-memref-to-llvm` **ignores it** —
verified directly:

```
$ mlir-opt --finalize-memref-to-llvm t.mlir
%4 = llvm.load %3 : !llvm.ptr -> i64      # with the attribute
%4 = llvm.load %3 : !llvm.ptr -> i64      # without it — identical
```

So the alignment cannot simply be forwarded; the memref path has to be changed to
produce an LLVM pointer plus an aligned `llvm.load`, or the element type of the
memref has to match the access type so that ABI alignment applies.

**Still open.** The remaining question is *why* the alignment reaching the load is
4 rather than 8. The leading hypothesis is that a `[u64]` device buffer is
represented with 32-bit elements: `type_memref`
(`crates/rustc_codegen_gpu/src/mlir/mod.rs:432-447`) rewrites aggregate element
types to `i8` with the size folded into an extra dimension, and
`crates/cuda_bindings/src/mem.rs:268,325` exposes only a `flatten()` with no
inverse — which is also what prevents working around the bug in user code by
viewing `[u64]` as `[U32_4]`.

**Not fixed here.** It changes core codegen used by all five case studies and
needs its own validation pass. On the evidence above it remains **the
highest-value outstanding fix in the toolchain**: it is worth roughly 1.5x on
every 64-bit-heavy kernel.

Two smaller related gaps found alongside it:

- `crates/gpu/src/vector.rs:24-52` and `ty.rs:430-445` — `U32_4` does not
  vectorise when passed through a `#[gpu::device]` function, so hand-packing wide
  loads only works when written inline (`macro_rules!` rather than a device fn).
- `crates/cuda_bindings/src/mem.rs:268,325` — `flatten()` has no inverse, so a
  `[u64]` device buffer cannot be viewed as `[U32_4]`.

## Not bugs, but the two measured costs of safety

Neither of these is a defect; both are quantified here because "safe GPU code is
slower" is usually asserted rather than measured. Each was obtained by building a
variant with the cost removed and timing it against the shipped kernel.

### Bounds checks: were 29-51%, now mostly recovered in safe Rust

This started as the headline cost of safety and ended as the task's most useful
result, so both halves are recorded.

**The measurement.** Building each kernel twice — once stock, once with a
provable range fact handed to LLVM — isolates the tax:

| Kernel | Size | Stock (µs) | Checks elided (µs) | Tax |
|---|---|---:|---:|---:|
| `conv3d` | 128³ | 38.9 | 27.7 | 28.6% |
| `conv3d` | 256³ | 288.8 | 198.5 | 31.3% |
| `mvt` column pass | 8192² | 1445.7 | 716.4 | 50.4% |

**The fix, in safe Rust, now shipped.** The tax is not inherent; it comes from
LLVM being unable to relate a thread-derived index to a slice length. Two source
changes recover almost all of it without `unsafe`:

1. **Sub-slice to the exact extent, then clamp in 32 bits.**
   ```rust
   if total == 0 || a.len() < total as usize { return; }
   let a = &a[..total as usize];   // a.len() is now literally zext(total)
   let last = total - 1;
   ... a[idx.min(last) as usize]
   ```
   The sub-slice is the load-bearing part: it lets LLVM compare
   `zext(umin(idx, total-1))` against `zext(total)` by comparing the **u32**
   operands. Clamping against `a.len()-1` instead is also provable but only in
   64 bits, costing a `min.u64` per access and recovering a third as much.

2. **`MapContinuousLinear::new(1)` instead of `reshape_map!`** where the kernel
   is already flat — `reshape_map!` un-flattens against a runtime `grid_dim` and
   emits a `div.u32`.

`conv3d` PTX went 204 → 176 → 162 instructions (`setp.gt.u64` 12 → 1,
`selp.b64` 24 → 2) with the load count unchanged, and the kernel went from
**1.88x CUDA to 1.05x**. The clamp is a no-op on the data — every index the
kernels generate is already in range — and it is confined to reads, since
clamping a *write* would silently corrupt rather than fault.

**Revised conclusion: this was a missing optimisation, not a cost of the
guarantee**, and it can be recovered today by the programmer. The remaining
opportunity is for the compiler to derive the same fact automatically, which is
tractable because stencil indices are affine in the thread id.

What did *not* work, and is worth knowing:

- **`assert!`-style length hoisting** — nothing bounds a thread-derived index, so
  there is no fact to hoist.
- **Clamping in `u32` against `(a.len()-1) as u32`** — the checks come back in
  full; LLVM will not see through the `trunc`/`zext` round trip.
- **The same masking trick on dynamic shared memory** (radix sort, experiment D)
  — no effect. A shared allocation's length is opaque, so masking narrows the
  index without supplying the matching extent fact.

### Atomic scatter: ~40% of the radix sort

SeGuRu requires data-dependent writes to go through `gpu::sync::Atomic`, lowering
to `atom.global.exch.b32`. Replacing the radix sort's scatter with a plain
structured store (incorrect, timing only) took 256 Mi keys from 24.5 ms to
17.4 ms. Full write-up in `gpusorting/README.md`.

**Less tractable, but not fundamental.** In this algorithm the scatter is provably
a permutation - every destination is written exactly once - and the type system has
no way to say so. A "provably disjoint scatter" primitive, consuming a witness that
the index map is injective and emitting an ordinary `st.global`, would recover most
of the gap to CUB without weakening safety.

## Finding 8: `ScopeUniqueMap` cannot express a data-dependent permutation (partially unblocked)

The radix-sort scatter writes each key to `offsets[digit] + rank`, a genuine
permutation of the tile (confirmed independently by four agents on four
different models). It currently uses `Atomic` purely because the destination
index is data-dependent, costing ~40% of sort time (24.5 vs 17.4 ms).

`reshape_map!` lowers to a fixed mixed-radix affine formula and cannot express a
runtime lookup, so a hand-written `ScopeUniqueMap` (`MapExplicit`) is required.
Two blockers were found; the first is fixed:

1. **Fixed.** `propogate_fn_call` tainted a call's destination whenever *any*
   argument was tainted, ignoring the callee's `ret_sync_data(MAX_FN_IN_PARAMS)`
   declaration. That declaration is exactly the assertion that the return is
   block-uniform regardless of its inputs, so it must survive tainted arguments.
   Note the constructor must also be `#[inline(never)]`: MIR inlining otherwise
   erases the call and the attribute with it.
2. **Fixed.** `chunk_mut` reported `InvalidDiversedData` on its *receiver*. A MIR
   dump plus an instrumented build of the analysis found the cause:
   `SharedAtomic::new` takes `&mut GpuShared` and every `&mut` argument of a
   device call is tainted unconditionally, so the first atomic view of a buffer
   taints the buffer handle for the rest of the kernel and no later `chunk_mut`
   on it can be accepted. `chunk_mut` already declares `ret_sync_data(0, 1000)`
   for the same receiver; `SharedAtomic::new` declared nothing. Adding
   `ret_sync_data(0)` to it fixes this and is sound: taking an atomic view does
   not diverge the handle, which stays the same block-uniform base pointer, and
   what varies per thread is the index each access supplies.

### And the result: the `Atomic` was free all along

With both fixes the scatter compiles as a safe `chunk_mut`. It buys **nothing**:

| | `Atomic` scatter | `MapExplicit` scatter |
|---|---|---|
| atom/red in PTX | 29 | 29 |
| `st.shared.u32` in PTX | 16 | 16 |
| 256 Mi sort | 19.603 ms | 19.606 ms |

`atomic_assign` is a store, not a read-modify-write, and already lowered to a
plain `st.shared.u32`; the `Atomic` was a checker artefact with no codegen
consequence. **The earlier estimate that it cost ~40% of sort time (24.5 ->
17.4 ms) was wrong**, and the 2.2x gap against CUB is somewhere else entirely.

The map version is therefore not adopted: no speed, and it costs an `unsafe`
block because `MapExplicit::new` carries the uniqueness obligation. The two
toolchain fixes are kept, because both were genuine bugs that would block any
legitimate use of a data-dependent map.

A methodology note, since this is the third time in this project: the 40% figure
came from an experiment that was never checked against generated code. Reading
the PTX first would have cost minutes and saved the whole exercise.

### Finding 5 follow-up: `init` generalised, no `unsafe` left in the case studies

The first `init` took a `PER_THREAD` const generic and ran a fixed loop, so it
only covered `N` an exact multiple of the block. Every buffer smaller than the
block (the `[f32; 32]` warp scratch in 256-thread blocks, and the AES tables)
still needed `unsafe { uninit() }`, leaving 23 `unsafe` sites.

`init` now hands the elements out round-robin and bounds the loop by `N`, so
thread `t` writes `t, t + block_size, ...` while that stays below `N`. One safe
call covers `N` larger than the block, smaller than it, or not a multiple of it,
and the const generic is gone. All 23 sites converted; `verify.sh` passes 67
tests *and* the no-`unsafe` audit.

Cost: buffers that were fully overwritten before being read now take one
redundant store and one extra barrier per block. Measured on an idle GPU, that
costs nothing detectable:

| | before | after |
|---|---|---|
| NTT fwd, N=16384 | 471.4 us | 471.9 us |
| NTT fwd, N=4096 | 219.4 us | 218.8 us |
| element-wise add | 65.0 us (1549 GB/s) | 65.2 us (1545 GB/s) |
| radix sort, 256 Mi | 19.612 ms (2.22x CUB) | 19.603 ms (2.22x) |
| AES encrypt, 1 GiB | 8299.5 us (1.00x) | 8313.3 us (1.00x) |

Every figure is within 0.4%, i.e. inside run-to-run noise. The redundant store is
amortised over a whole block and the barrier is one of many the kernels already
execute.

**Caveat on benchmarking: always confirm the GPU is idle first.** This machine is
shared. A sweep taken while another process held 17% utilisation showed the NTT
apparently regressing 40% and the sort going 2.11x -> 2.60x against CUB, and the
sort contains no shared-memory initialisation at all. The giveaway is that the
*CUDA baseline* columns slowed by 21-38% at the same time, since they contain no
SeGuRu code; the cleanest control was heongpu's element-wise kernel, which uses
no `GpuShared` whatsoever yet appeared 42% slower. Re-running on an idle GPU
reproduced every baseline number exactly.

Note also that under contention SeGuRu degraded consistently *more* than the CUDA
baselines (+40% vs +29% on NTT, +42% vs +21% on element-wise). That asymmetry is
unexplained and is not caused by any code change here; a plausible but unverified
reading is that the SeGuRu kernels carry a heavier register/shared-memory
footprint and so lose more occupancy when sharing SMs.

Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before
recording any ratio.

## Finding 9: the radix sort costs 1.15x against the same algorithm, not 2.2x — the CUB comparison was apples-to-oranges

Every earlier phase quoted "2.2x slower than CUB" as the cost of safety. That
number is real but it is not a safety cost, because **CUB is not running our
algorithm.** `nsys profile -t cuda --stats=true` shows CUB on CUDA 13.3 / sm_80
dispatching `DeviceRadixSortOnesweepKernel` — onesweep with decoupled look-back.
Our kernels are a transliteration of Thomas Smith's *reduce-then-scan*
`DeviceRadixSort.cu`. Comparing them measures the choice of algorithm.

So we built the honest baseline: `gpusorting/cuda/upstream/` vendors that exact
CUDA file (MIT), and `cuda/drs_variant.cu` compiles it **twice** with different
tuning macros — once at upstream's tuning (7680 keys/tile, 15 per thread) and
once at our Rust port's (4096 / 8). Same algorithm, same launch geometry, same
kernel sequence; the only difference against our port is the compiler and the
safety checks. Both are checked against `sort_unstable` in the harness.

256 Mi keys, A100 idle:

| implementation | ms | vs SeGuRu | what the ratio means |
| --- | ---: | ---: | --- |
| SeGuRu (safe Rust) | 19.608 | — | |
| **DRS, our tuning (CUDA C++)** | **17.045** | **1.15x** | **the cost of SeGuRu** |
| DRS, upstream tuning (CUDA C++) | 16.242 | 1.21x | + our smaller tile |
| CUB (onesweep) | 8.779 | 2.23x | + a different algorithm |
| Thrust | 9.928 | 1.90x | |

The old 2.23x factorises cleanly, and the arithmetic closes:

    1.15 (SeGuRu)  x  1.05 (tuning)  x  1.85 (algorithm)  =  2.24

Per-kernel, same run, same 256 Mi maxima — this is the part that matters:

| kernel | SeGuRu | same CUDA, same tuning | ratio |
| --- | ---: | ---: | ---: |
| `radix_upsweep` | 1.177 ms | 1.195 ms | **0.98x** |
| `radix_scan` | 0.500 ms | 0.518 ms | **0.97x** |
| `radix_downsweep` | 3.474 ms | 2.745 ms | **1.27x** |

**Two of the three kernels are at parity or marginally faster than CUDA C++.**
The whole 1.15x is the downsweep, and the downsweep is the one kernel that does a
data-dependent scatter. That is a far more precise localisation of the cost of
safety than anything in phases 1-3, and it is the one number worth attacking.

Three corrections to earlier claims fall out of this:

* The 2.2x was never a safety cost. Roughly 80% of it is the algorithm and the
  tuning.
* An earlier version of this finding compared our downsweep against *CUB's*
  scatter (618 vs 1040 GB/s) and blamed 50% occupancy. The occupancy figure is
  right — `ptxas` reports 56 registers, and at `DOWNSWEEP_THREADS = 512` that is
  28 672 registers per block, so 2 blocks per SM out of a possible 4, i.e. 1024
  of 2048 threads — but it cannot explain the SeGuRu-vs-CUDA gap, because the
  CUDA baseline runs the identical launch geometry. It explains why *both*
  reduce-then-scan implementations trail onesweep.
  (Note the earlier write-up said "56 registers with 256 threads"; the block is
  512 threads. The occupancy conclusion is unchanged, the arithmetic was wrong.)
* Our port is not even tuned like the file it transliterates: `BIN_PART_SIZE` is
  4096 against upstream's 7680 and `BIN_KEYS_PER_THREAD` is 8 against 15. No
  reason is recorded; it dates from the initial rewrite. Raising the Rust
  constants to 7680/15 **compiles and passes the SeGuRu analysis but produces
  wrong results**, so the port has an undocumented dependency on the 4096 tile.
  Worth about 5% and left open.

### Why onesweep is not simply the next step -- all three predictions were wrong

An earlier version of this finding predicted that onesweep could not be written
in safe Rust, for three reasons: dynamic tile acquisition via `atomicAdd`, an
unbounded spin on a neighbour's flag, and a forward-progress assumption. Those
were **predictions from reading the analysis rules, not results.** The port has
since been written (`src/onesweep.rs`), and all three are false. See Finding 10.

### Method notes

`ncu` is unusable here (`ERR_NVGPUCTRPERM`) but `nsys` needs no counter
permission and the kernel-summary table was sufficient. It should have been the
*first* diagnostic rather than the fourth.

The larger lesson is about the baseline, not the tool: four successive claims
about where the sort's time went were wrong (the AES harness, the contended
sweep, the atomic scatter, and the occupancy story above), and the last two were
wrong because they were measured against a *different algorithm* without anyone
checking. Before quoting a ratio, confirm the baseline runs the same algorithm —
`nsys` prints the kernel names, which is all it took.

nvcc 13.3 no longer macro-expands the operand of `#pragma unroll`, so upstream's
`#pragma unroll BIN_KEYS_PER_THREAD` needed a one-line `_Pragma` workaround. It
is marked `LOCAL MODIFICATION` in the vendored file.

---

## Finding 10: onesweep *is* expressible in safe Rust, and its entire cost against CUDA is one missing primitive — an atomic load

Finding 9 predicted that onesweep was out of reach for safe Rust. The port was
written anyway (`gpusorting/src/onesweep.rs`, ~600 lines, three kernels) and
**passed all nine correctness cases on its first run**, up to 16 Mi keys. Every
one of the three predicted blockers is accepted by SeGuRu:

| predicted blocker | reality |
| --- | --- |
| tile index from a runtime `atomicAdd`, not `blockIdx` | accepted. The index is stashed in shared memory and broadcast; nothing indexes global memory with a per-thread diverged value. |
| unbounded spin whose exit condition another block has not written yet | accepted. `mir_thread_sync_check.rs` has **no termination analysis**. The one liveness-adjacent rule is barrier divergence, and the look-back loop contains no barrier, so it is legal. |
| forward-progress assumption | not checked at all. SeGuRu guarantees memory safety and data-race freedom, **not liveness**. It will happily compile a kernel that hangs. |

That last row is the honest statement of scope, and it cuts both ways: SeGuRu
does not reject onesweep on liveness grounds, and it also would not have saved us
if the look-back had been wrong.

### The same-algorithm, same-tuning comparison

`cuda/upstream/OneSweep.cu` is vendored (MIT) and `cuda/os_variant.cu` compiles it
twice, at upstream's 7680-key tile and at our 4096-key tile, exactly as was done
for reduce-then-scan. Milliseconds, A100 80GB idle, mean of 20-50 iterations:

| keys | SG-RS | DRS-ours | RS ratio | SG-OS | OS-ours | **OS ratio** | OS-up | CUB | Thrust |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 Ki | 0.108 | 0.096 | 1.13x | 0.062 | 0.166 | **0.38x** | 0.171 | 0.077 | 0.088 |
| 1 Mi | 0.151 | 0.142 | 1.06x | 0.123 | 0.210 | **0.59x** | 0.217 | 0.116 | 0.331 |
| 4 Mi | 0.383 | 0.323 | 1.19x | 0.405 | 0.346 | 1.17x | 0.371 | 0.233 | 0.478 |
| 16 Mi | 1.357 | 1.186 | 1.14x | 1.549 | 1.088 | 1.42x | 1.131 | 0.643 | 0.949 |
| 64 Mi | 5.009 | 4.372 | 1.15x | 5.925 | 3.542 | 1.67x | 3.701 | 2.228 | 2.776 |
| 256 Mi | 19.608 | 17.053 | **1.15x** | 23.336 | 13.314 | **1.75x** | 13.940 | 8.757 | 10.028 |

Three things stand out:

* **Below ~1 Mi keys the safe-Rust onesweep is 1.7-2.6x *faster* than the CUDA
  one** (0.38x, 0.59x). At those sizes the run is dominated by launch and clear
  overhead in the dispatcher, not by the kernels.
* **The RS ratio is flat at 1.15x; the OS ratio degrades with size**, 1.17x ->
  1.42x -> 1.67x -> 1.75x. A ratio that grows with the problem is a contention
  signature, not a codegen one.
* CUB (8.757) is still 1.52x faster than upstream's *own* onesweep at either
  tuning (13.3 / 13.9), so CUB's remaining advantage is engineering, not
  algorithm. It is not a like-for-like baseline for anything.

### Where the OS ratio actually goes: the look-back spin, and why

`bin/onesweep_lb.rs` runs the identical kernel twice. In the second run every
tile's backwards walk starts at slot 0 instead of at its immediate predecessor;
slot 0 is seeded `FLAG_INCLUSIVE` by `onesweep_scan`, so the walk terminates on
its first read. The publish and the read traffic are unchanged — only the
*waiting* is removed. The result is wrong by construction; the delta is the cost
of waiting:

| keys | onesweep ms | no wait ms | **waiting** |
| ---: | ---: | ---: | ---: |
| 4 Mi | 0.403 | 0.247 | 38.7% |
| 16 Mi | 1.535 | 0.828 | 46.1% |
| 64 Mi | 5.908 | 3.228 | 45.4% |
| 256 Mi | 23.317 | 12.761 | **45.3%** |

**At 256 Mi, removing the waiting takes the safe-Rust onesweep to 12.76 ms —
faster than the CUDA onesweep at the same tuning (13.31 ms).** The compute half
of the port is already at or past parity with CUDA C++. The whole 1.75x is spin.

The mechanism is one instruction, and it is visible in the generated code:

| | look-back read | PTX |
| --- | --- | --- |
| CUDA (`volatile uint32_t* passHistogram`) | a load | `ld.volatile.global.u32` |
| SeGuRu (`Atomic::atomic_ori(0)`) | a read-modify-write | `atom.global.or.b32` |

`crates/gpu/src/sync.rs` exposes only `memref.atomic_rmw` — there is **no atomic
load and no CAS**. The idiomatic read is therefore `atomic_ori(0)`, an RMW that
returns the old value unchanged. It is correct, and it has the L1-bypassing,
device-scope coherence that CUDA gets from `volatile`, which is precisely why it
was used. But an RMW must take exclusive ownership of the L2 sector, so every
spinner on an address serialises against every other spinner *and* against the
publisher, where `ld.volatile.global` spinners merely share a line. With 256
threads per tile polling 256 addresses across thousands of concurrent tiles, that
is the difference between a shared read and a queue.

**This is the single most actionable result in this document.** It is not a
codegen quality problem, not an occupancy problem, and not an algorithm problem:
it is one missing primitive. An `Atomic::<u32>::atomic_load()` lowering to
`ld.relaxed.gpu.global.u32` (an atomic load — still race-free, so it costs
nothing in safety) would be expected to close most of a 1.75x gap on the single
most important GPU sort primitive, and it would benefit every decoupled-look-back
algorithm, which is most modern GPU scans.

### A latent bug in upstream `OneSweep.cu`, found by retuning it

`OneSweep.cu` stashes the acquired partition index in
`s_warpHistograms[BIN_PART_SIZE - 1]`. That is only safe while
`BIN_PART_SIZE > BIN_HISTS_SIZE` (4096). At upstream's 7680-key tile, index 7679
is past the histograms and all is well; at a **4096**-key tile it *is* warp 15 /
bin 255, so the histogram increments overwrite the partition index and the kernel
scatters to a wild address (`compute-sanitizer` reports the wild write at
`OneSweep.cu:352`). Fixed here with a dedicated `__shared__ uint32_t
s_partitionIndexSlot`, marked `LOCAL MODIFICATION`. Worth reporting upstream.

The safe-Rust port never had this bug: it already used a dedicated `PART_SLOT`
because, unlike the CUDA, it had no spare word to alias.

### Method notes

* A `#[gpu::cuda_kernel]` fn is generic over its config and is **only instantiated
  where `::launch` is called**. With no host caller it is never analysed and never
  emitted, and the build still says "Finished". Verify before trusting a build:
  `find target -name '*.ptx' -newermt '-3 minutes' -exec grep -ho 'visible \.entry [a-zA-Z0-9_]*' {} \;`
* Pre-seeding the flag array as `FLAG_INCLUSIVE` to short-circuit the spin **does
  not work** and hangs the GPU: the tile's own `atomic_addi` publish then lands on
  an already-tagged slot, the flag field wraps to 3, and successors match neither
  branch and spin forever. Hence the start-at-slot-0 formulation above.
* Eliding the look-back entirely also gives a wrong answer *and* a wrong
  measurement — it made the kernel 2.8x **slower**, because zero prefixes send the
  scatter to degenerate addresses and destroy coalescing. Any probe that changes
  the addresses being written is measuring the memory system, not the thing you
  removed.
* Packing all four passes' look-back buffers into one allocation needs
  `(tiles + 1) * RADIX` words *per pass*: `Scan` seeds row 0 and tile *i* publishes
  into row *i+1*. Upstream gets away with `tiles` rows because each pass has its
  own `cudaMalloc`, which rounds up.
