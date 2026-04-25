# SeGuRu codegen changes: design and safety

This document records the codegen/API work and related benchmark-side
optimizations made while closing the gap between SeGuRu kernels and raw CUDA.
It intentionally focuses on design and safety. Benchmark history and current
parity status live in `docs/cuda-to-seguru-porting-progress.md`.

## Scope

Covered here:

- Core API/codegen support:
  - `MapWithLidOffset` and `MapWithRows` in `crates/gpu/src/chunk.rs`.
  - `GlobalGroupChunk::open_tile`, `OpenedTile`, and `OpenedRow` in
    `crates/gpu/src/tile.rs`.
  - `reshape_map!` generation of split offset methods in
    `crates/gpu-macros/src/reshape_map.rs`.
- Benchmark-side optimizations that depend on or motivate the codegen work:
  - `Float4` global load/store signatures in KernelBench-C GEMM kernels.
  - Checked benchmark-facing view/map rewrites that keep example code free of
    raw casts and manual `ScopeUniqueMap` implementations.
  - K-major shared-memory layout for fused GEMM kernels.
  - `#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]` on hand-written
    KernelBench-C fused GEMM kernels.

Not covered as implementation changes:

- A custom MLIR/LLVM optimization pass. None was added.
- A general loop-strength-reduction pass for fully unrolled address arithmetic.
- Tensor-core or WMMA lowering.

## Problem addressed

The original fused GEMM kernels reached the right algorithmic shape but emitted
more address arithmetic than the equivalent raw CUDA. In particular, unrolled
per-thread tile stores through `reshape_map!` repeatedly recomputed linear
indices:

```text
map(lid, tid) -> u32 linear index -> zext -> byte offset -> pointer
```

For each unrolled store, LLVM could common-subexpression-eliminate some i32
terms, but it did not synthesize the row-pointer plus immediate-offset pattern
that `nvcc` produces after loop-strength reduction. The important observation
was that a `reshape_map!` index is structurally additive:

```text
map(local_index, thread_ids) =
    thread-dependent base(thread_ids) + local-index offset(local_index)
```

The codegen/API changes expose that split explicitly so user code can
materialize a per-thread base pointer once, then index within the thread-owned
tile with smaller local offsets.

## Core design

### `MapWithLidOffset`

`MapWithLidOffset<CS>` is an unsafe companion trait to `ScopeUniqueMap<CS>`.
It splits the existing `map(lid, thread_ids)` calculation into:

- `thread_base(thread_ids)`: the thread-dependent base offset, including the
  map's struct-level offset.
- `map_lid_offset(lid)`: the local-index contribution, with no thread-dependent
  terms.

The required algebra is:

```text
map(lid, tid).1 == thread_base(tid).1 + map_lid_offset(lid).1
```

The validity flag is split the same way:

```text
map(lid, tid).0 == thread_base(tid).0 & map_lid_offset(lid).0
```

This trait is unsafe because a wrong split can produce a pointer to a different
element than `map` would have produced. The macro-generated implementation is
safe to use because it derives all three methods from the same size/layout
inputs.

### `open_tile`

`GlobalGroupChunk::open_tile(self)` consumes a mutable thread chunk and returns
an `OpenedTile`:

```rust
let mut tile = chunk_mut(y, out_map).open_tile();
tile[(col, row)] = value;
```

At open time, SeGuRu computes:

```text
tile_ptr = data.as_mut_ptr() + thread_base(CS::thread_ids())
base_ok = thread_base_valid & map.precondition()
```

Each later `Index` / `IndexMut` only computes the local contribution:

```text
ptr = tile_ptr + map_lid_offset(lid)
assert_ptr(base_ok & lid_ok, ptr)
```

This avoids repeating the thread-base computation across all unrolled accesses.
It also gives the backend a pointer-shaped representation that is closer to the
PTX shape raw CUDA gets from row-pointer hoisting.

### `MapWithRows` and `OpenedRow`

`MapWithRows<CS>` is available for two-dimensional local index maps,
specifically maps whose local index type is `(u32, u32)`. It further splits the
local offset into:

- `row_lid_offset(row)`
- `in_row_lid_offset(col)`

The required algebra is:

```text
map_lid_offset((col, row)).1 ==
    row_lid_offset(row).1 + in_row_lid_offset(col).1
```

and the validity flags must split as:

```text
map_lid_offset((col, row)).0 ==
    row_lid_offset(row).0 & in_row_lid_offset(col).0
```

`OpenedTile::row_mut(row)` borrows one row of the opened tile:

```rust
let mut y_tile = chunk_mut(y, out_map).open_tile();
let mut row = y_tile.row_mut(i as u32);
row[0u32] = o0;
row[1u32] = o1;
```

This is used by the vectorized `gemm_add_relu` epilogue, where each thread
writes two `Float4` values across eight rows. The row view lets the code express
the intended "row base plus in-row column offset" structure directly.

### `reshape_map!` generated implementations

`reshape_map!` now emits the split methods for its private map type:

- `ScopeUniqueMap::map`
- `MapWithLidOffset::map_lid_offset`
- `MapWithLidOffset::thread_base`
- `MapWithRows::{row_lid_offset, in_row_lid_offset}` when the local rank is 2

All methods are generated from the same `old_sizes`, `new_sizes`, `layout`,
`offset`, and cached weights. This avoids hand-maintained duplicate arithmetic.

The generated code follows the same per-dimension arithmetic as `map`, but each
method only visits the dimensions it owns:

- local dimensions for `map_lid_offset`
- thread dimensions for `thread_base`
- local dimension 1 for `row_lid_offset`
- local dimension 0 for `in_row_lid_offset`

The layout parser still requires a permutation of all local/thread dimensions;
the split methods therefore inherit the same structural mapping as the original
full map.

## Safety design

### Ownership and aliasing

Mutable writes remain tied to the existing SeGuRu ownership model.

- `chunk_mut` returns a `GlobalGroupChunk<'a, T, CS, Map>` over an exclusive
  `&'a mut [T]`.
- `IndexMut` for `GlobalGroupChunk` is only available when
  `CS::ToScope = Thread`, preserving the existing "one thread owns this chunk"
  rule.
- `open_tile(self)` consumes the chunk, so safe Rust cannot keep both the
  original chunk and the opened tile alive.
- `OpenedTile` carries `PhantomData<&'a mut [T]>`, preserving the underlying
  mutable borrow lifetime even though the implementation stores a raw pointer.
- `row_mut(&mut self, row)` borrows the opened tile mutably. Safe Rust cannot
  hold two mutable `OpenedRow`s from the same tile at the same time.
- `OpenedRow` carries `PhantomData<&'row mut T>`, tying the row view to the
  mutable borrow of the parent tile.

No public API accepts a raw pointer, byte stride, or user-supplied row stride.
All pointer offsets come from the same generated map arithmetic that already
backs safe `chunk_mut` indexing.

### Bounds and validity checks

`OpenedTile` and `OpenedRow` follow the same GPU backend `assert_ptr`
convention used by normal chunk indexing: the generated address is passed to
`assert_ptr` with a combined validity flag before it is used by the backend's
safety instrumentation.

This is a backend/codegen contract, not a stronger host-Rust bounds mechanism.
The source still forms the raw pointer and reference expression before calling
`assert_ptr`, just like the existing chunk indexing path. The safety requirement
for the split APIs is therefore that they produce the same address and valid
address set as the original `map` method.

The validity flag is accumulated in stages:

```text
base_ok = thread_base_valid & map.precondition()
row_ok  = base_ok & row_lid_offset(row).valid
ptr_ok  = row_ok  & in_row_lid_offset(col).valid
```

For direct tile indexing, the final check is:

```text
base_ok & map_lid_offset(lid).valid
```

This preserves the original out-of-bounds semantics: when bounds checks are
enabled, invalid accesses trap through `assert_ptr`; when
`DISABLE_GPU_BOUND_CHECK=true`, the same code path can optimize away checks as
configured by the backend.

### Trait contracts

The unsafe trait contracts are deliberately algebraic. They do not claim that
the new APIs are safe because a pointer is "probably in range"; they require
that the split offsets reconstruct the same element as the original full map.

For `MapWithLidOffset`:

```text
thread_base(tid) + map_lid_offset(lid) == map(lid, tid)
```

For `MapWithRows`:

```text
row_lid_offset(row) + in_row_lid_offset(col) ==
    map_lid_offset((col, row))
```

Because `reshape_map!` emits these methods mechanically from the same layout
description, the contract is maintained by construction for macro-generated
maps.

Manual implementations of these traits must be treated like manual
`ScopeUniqueMap` implementations: unsafe, review-sensitive, and responsible for
proving uniqueness and offset equivalence.

### Thread uniqueness

`open_tile` does not change the cross-thread uniqueness proof. It depends on
the same `ScopeUniqueMap` guarantee:

```text
thread_ids1 != thread_ids2 ==>
    map(idx1, thread_ids1) != map(idx2, thread_ids2)
```

The split only changes how an address is computed, not which address each
thread owns. If two threads were disjoint under `map`, their opened tile bases
plus local offsets remain disjoint under the split contract.

### Row views do not create aliasing

`row_mut` is intentionally a borrowing API, not a pointer-returning API. The
caller receives an `OpenedRow` tied to `&mut self`; while that row exists, the
parent `OpenedTile` cannot be used to create another row or perform other
mutable tile accesses through safe Rust.

This means row views preserve Rust's ordinary mutable-alias rule even though the
implementation uses raw device pointers internally.

## Benchmark-side optimizations

### `Float4` global loads and stores

Many KernelBench-C GEMM kernels now accept `x: &[Float4]` and `w: &[Float4]`
instead of scalar `&[f32]` for the global input matrices. The host runner uses
the checked `TensorView::try_cast_slice::<Float4>()` /
`TensorViewMut::try_cast_slice_mut::<Float4>()` APIs before launch instead of
benchmark-local raw pointer casts.

Safety requirements:

- The logical K dimension is a multiple of 4. The GEMM runners assert tile
  divisibility such as `k % BK == 0` with `BK = 8`.
- Vectorized bias/output paths, such as the hand-written `gemm_add_relu` arm,
  also require the logical N dimension to be vector divisible; the fused-GEMM
  runners assert `n % BN == 0` with `BN = 128`.
- The original device allocation is contiguous `f32` data.
- `Float4Inner` is generated as
  `#[repr(C, align(16))] { data: [f32; 4] }`. `Float4` is the corresponding
  single-field `VecType<Float4Inner>` wrapper, and `VecType<T>` is explicitly
  `#[repr(transparent)]`.
- `try_cast_slice` validates that the source byte length is divisible by the
  destination element size, validates destination alignment, and rebuilds slice
  metadata with the destination element count.
- Both the source and destination element types must implement
  `TensorViewCastElement`, an unsafe marker trait reserved for plain data types
  with no invalid bit patterns or drop-sensitive ownership. This keeps the safe
  API from reinterpreting GPU bytes as arbitrary Rust types.
- The reinterpretation is only used to change the typed view passed to the
  kernel; ownership and lifetime remain tied to the original `TensorView` or
  `TensorViewMut` borrow.

`gemm_add_relu` also uses `Float4` for bias and output in the hand-written arm.
That path writes two `Float4`s per row through `OpenedRow`, producing vector
stores while keeping the output map safe.

### Benchmark/example map safety

Benchmark and example crates should not define their own `unsafe impl
ScopeUniqueMap` for one-off address formulas. Those implementations are valid
only if the author proves cross-thread uniqueness and all index/validity
algebra by hand, which is the same class of proof as the core map
implementations. Prefer existing reviewed maps (`MapContinuousLinear`, `Map2D`)
or generated `reshape_map!` maps so the unsafe proof stays in the library or in
macro-generated code.

The PolyBench LU trailing-update map is the current concrete example. The old
benchmark-local map addressed:

```text
A[(k + 1 + i_tail) * n + (k + 1 + j_tail)]
```

by taking a mutable slice rooted at `A[(k + 1) * n]` and manually adding
`k + 1 + j_tail` inside `LuTrailingUpdateMap`. The safe replacement roots the
mutable slice at `A[(k + 1) * n + (k + 1)]` and uses `Map2D::new(n)`, so the map
adds only:

```text
i_tail * n + j_tail
```

relative to the slice base. The resulting element address is identical, and the
edge checks stay in the kernel (`i_tail < rem && j_tail < rem`) while the
cross-thread uniqueness proof comes from the reviewed `Map2D` implementation.

A source guard in `benchmarks/check_polybench_transfer_contract.py` now fails if
examples reintroduce manual `unsafe impl ... ScopeUniqueMap ... for ...`
definitions.

### K-major shared-memory layout

The fused GEMM kernels use a shared-memory load map like:

```rust
let load_map = reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0]);
```

This maps four contiguous K lanes loaded by a thread into a K-major shared
layout:

```text
offset = k_lane * 128 + row_or_col
```

The design goal is to make compute-phase reads from shared memory match the
CUDA-style `[k][row]` / `[k][col]` traversal while preserving SeGuRu's disjoint
write proof during the load phase.

Safety requirements:

- Shared-memory writes still go through `chunk_mut(load_map)`.
- The `reshape_map!` map is a `ScopeUniqueMap`, so each thread's four scalar
  writes are disjoint from other threads' writes.
- `sync_threads()` separates the shared-memory load phase from the compute
  phase, preventing read-after-write races.
- Compute-phase reads from `GpuShared` are read-only indexed accesses; they do
  not create mutable aliasing.

### Launch bounds

Several hand-written KernelBench-C fused GEMM kernels use:

```rust
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
```

This emits PTX launch-bound metadata equivalent to a maximum block size of
`16 x 16 x 1` and a minimum of two CTAs per SM. It is a performance hint to
ptxas, not a memory-safety feature.

Safety and correctness implications:

- The attribute does not change indexing, ownership, or aliasing.
- It must match the actual launch shape used by the runner
  (`BDIM_X = 16`, `BDIM_Y = 16`).
- If a runner launched a different block shape, that would be a launch-contract
  bug, not a memory-safety proof change.

## Validation

The codegen/API changes are covered by compile/codegen tests and benchmark
validation:

- `crates/rustc_codegen_gpu/tests/codegen/row_view.rs` compiles a minimal
  `open_tile().row_mut()` use and checks that it lowers to a vectorized store
  form.
- `crates/rustc_codegen_gpu/tests/codegen/nvvm_launch_bound.rs` checks that
  launch-bound attributes lower to `.maxntid` and `.minnctapersm`.
- `crates/rustc_codegen_gpu/tests/codegen/float4.rs` checks vector load/store
  lowering such as `ld.global.v4.f32` and `st.global.v4.f32`.
- `crates/rustc_codegen_gpu/tests/fixtures/gemm_add_relu_ptxas.rs` is a
  focused ptxas fixture for the optimized fused GEMM pattern, including
  `Float4`, K-major shared layout, launch bounds, `open_tile`, and `row_mut`.
- `examples/kernelbench-c/results/reported_comparison.txt` records the
  KernelBench-C refresh results used as validation evidence:
  the safe vector-view/map rerun reports SeGuRu-from-CUDA geomean vs raw CUDA
  at `1.0045x`, with `12/12` correctness.
- `benchmarks/check_polybench_transfer_contract.py` guards benchmark contracts
  and now rejects benchmark/example-local manual `ScopeUniqueMap`
  implementations.
- `examples/polybench/lu` tests the LU `Map2D` trailing-update replacement
  against the CPU reference.

Useful focused commands:

```bash
cd /home/sanghle/work/seguru/examples
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:/usr/lib/llvm-20/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/lib/llvm-20/lib:${LD_LIBRARY_PATH:-}
export DISABLE_GPU_BOUND_CHECK=true
export RUST_MIN_STACK=67108864

cargo build --release -p kernelbench-c
python3.11 kernelbench-c/python/compare.py
cd /home/sanghle/work/seguru
python3 benchmarks/check_polybench_transfer_contract.py
cd /home/sanghle/work/seguru/examples
cargo test -p polybench-lu
```

For codegen tests, run from the workspace that owns
`rustc_codegen_gpu` with the same CUDA/LLVM environment:

```bash
cargo test -p rustc_codegen_gpu test_codegen_backend_output
```

## Known limitations

- `open_tile` and `row_mut` improve address shape for code that can express a
  thread-owned tile or row. They are not a general LLVM loop-strength-reduction
  replacement.
- `MapWithRows` is only generated for two-dimensional local index maps. Higher
  local ranks need separate split traits and proofs.
- The original motivation was reducing address arithmetic and register
  pressure, but the final performance impact of row views alone was small in
  the GEMM canary. KernelBench-C parity was mainly achieved by matching raw
  CUDA algorithms, vector width, shared-memory layout, and convolution
  strategy.
- The host-side `TensorView<[f32]>` to `TensorView<[Float4]>` reinterpretation
  is safe only when performed through the checked tensor-view API and under the
  stated shape/layout conditions. Benchmark/example code should not use raw
  pointer casts for vector views.
- Benchmark/example code should not add one-off manual `ScopeUniqueMap`
  implementations. If an address formula cannot be expressed with `Map2D`,
  `MapContinuousLinear`, or `reshape_map!`, move the abstraction into a reviewed
  core helper instead of localizing `unsafe` in the example.
- `nvvm_launch_bound` can improve occupancy when register pressure is near a
  block-per-SM threshold, but it can also force spills if used without
  reducing register demand. Treat it as a measured tuning knob, not a default.

## Future work

Potential follow-ups:

- Generalize `MapWithRows` to higher-rank local tiles if another kernel needs
  a safe row/plane view.
- Add stronger PTX-shape tests for `open_tile` / `row_mut` that check for
  row-base reuse directly, not just vector store presence.
- Investigate source-level or backend-level reductions in main-loop register
  pressure; Nsight Compute evidence shows occupancy/register pressure remains
  the main codegen risk for large fused GEMM kernels.
