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

## Finding 9: the radix sort's 2.2x gap against CUB is half algorithmic, half occupancy — neither is a SeGuRu safety cost

Earlier phases attributed the sort's 2.2x gap to safety overhead, and finding 8
guessed the shared-memory `Atomic` in the scatter was worth "~40% of sort time".
Both were wrong. `nsys profile -t cuda --stats=true` on `sort-bench` gives the
per-kernel breakdown and settles it.

The first thing the profile shows is that **we are not running CUB's algorithm.**
CUB on CUDA 13.3 / sm_80 dispatches `DeviceRadixSortOnesweepKernel` — it is doing
*onesweep* (one global histogram, then one fused scatter-with-decoupled-lookback
per digit). Our port follows the classic `DeviceRadixSort.cu` structure of three
kernels per digit pass: `radix_upsweep`, `radix_scan`, `radix_downsweep`.

Per-kernel maxima, which are the 256 Mi (268 435 456 key) run:

| kernel | ms | traffic | achieved BW | % of 1549 GB/s peak |
| --- | --- | --- | --- | --- |
| our `radix_upsweep` | 1.18 | 1.07 GB read | 913 GB/s | 59% |
| our `radix_scan` | 0.50 | 0.13 GB | — | — |
| our `radix_downsweep` | 3.47 | 2.15 GB read+write | 618 GB/s | 40% |
| CUB `DeviceRadixSortHistogramKernel` | 0.78 | 1.07 GB read | 1376 GB/s | 89% |
| CUB `DeviceRadixSortOnesweepKernel` | 2.06 | 2.15 GB read+write | 1040 GB/s | 67% |

Reconstructing the whole sort from these validates the reading: ours is
`(1.18 + 0.50 + 3.47) x 4 passes = 20.6 ms` against 19.61 ms measured, and CUB is
`0.78 once + 2.06 x 4 = 9.0 ms` against 8.74 ms measured. Both within 5%.

The 11.6 ms gap then splits almost exactly in two:

- **5.92 ms (51%) is algorithmic.** We pay `upsweep + scan` on *every* pass
  (1.68 ms x 4 = 6.7 ms); onesweep pays a single histogram (0.78 ms). This is
  also why we move 12.9 GB of key traffic against CUB's 9.7 GB (1.33x): the
  upsweep re-reads the whole key array each pass purely to build a histogram the
  downsweep then rebuilds locally anyway.
- **5.63 ms (49%) is our downsweep being 1.69x slower than CUB's scatter**
  (618 vs 1040 GB/s) despite issuing identical traffic.

For the second half, `ptxas -arch=sm_80 -O3 -v` names the cause: the downsweep
uses **56 registers** with `DOWNSWEEP_THREADS = 256`, i.e. 14 336 registers per
block, so only `65536 / 14336 = 4` blocks fit per SM — 1024 of 2048 threads,
**50% occupancy**. Shared memory is not the limit (`SMEM_WORDS = 4352` u32 =
17 408 B/block would allow 9 blocks in A100's 164 KB). There are no spills. At
50% occupancy there is not enough concurrency to hide the scatter's latency,
which is exactly the pattern of a memory kernel stuck at 40% of peak. Dropping to
51 registers would buy a 5th block (62.5%) and 42 would buy a 6th (75%).

Neither half is a cost of safe Rust:

- the algorithmic half is a *choice of algorithm* inherited from the CUDA source
  we ported, and would be paid identically by an `unsafe` port of the same file;
- the occupancy half is a register-allocation outcome, and the PTX contains no
  bounds checks or safety instrumentation in the inner loop to blame it on.

Closing the gap therefore means porting onesweep, not removing safety
constructs. That is the one place a real safety obstacle may appear: decoupled
look-back requires a spin-wait on another block's flag with acquire/release
ordering, and SeGuRu's taint analysis currently has no way to express a
device-scope acquire load in a loop. **That is a hypothesis, not a finding — it
has not been attempted.**

Method note: `ncu` is unusable here (`ERR_NVGPUCTRPERM`), but `nsys` needs no
counter permission and the kernel-summary table was sufficient. It should have
been the *first* diagnostic rather than the fourth; three successive estimates of
where sort time went (the AES harness, the contended sweep, the `Atomic`
scatter) were all falsified by direct measurement.
