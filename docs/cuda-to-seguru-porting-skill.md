# CUDA to SeGuRu Porting Skill

## Overview

This skill documents battle-tested strategies for porting CUDA C++ GPU kernels to SeGuRu (Safe GPU Rust). These patterns were validated on PolyBenchGPU, KernelBench, and the production llm-rs codebase (GPT-2 training/inference in SeGuRu).

## Scope and empirical status

Raw custom CUDA parity is the primary target. PyTorch eager is context only when
it dispatches to cuBLAS/cuDNN/tensor cores or otherwise changes the comparison
contract.

This document is intentionally limited to reusable porting rules, recipes, and
debugging guidance. Design rationale, implementation history, benchmark tables,
and current parity targets are maintained in
[`docs/cuda-to-seguru-porting-progress.md`](cuda-to-seguru-porting-progress.md).

## Golden Rules

1. **Use `u32` for all GPU-side index variables and size parameters** — GPU has 32-bit ALUs; `usize` (64-bit) doubles arithmetic cost (30-45% overhead)
2. **Use subslice + iteration for row traversals** — `&a[row*n..(row+1)*n]` does ONE bounds check; per-element `a[i*n+k]` checks every access
3. **Use `MapContinuousLinear`** instead of `MapLinear` for 1D `chunk_mut` — uses `u32` index math internally
4. **Kernel function name must differ from crate name** — `histogram::histogram` causes MLIR mangling bug; use `histogram::histogram_kernel`
5. **Tests must use helper functions** — don't put `cuda_ctx` closure directly in `#[test]` fns (causes `{}` in MLIR symbol names)
6. **`chunk_mut` uses LOCAL indices** — write `c[0]`, not `c[global_idx]`
7. **Row reductions need block- or warp-per-row, not 1-thread-per-row** — reusing the elementwise template (`gs=B.div_ceil(bs)`, `if row<B`) for `sum(x,dim=-1)`-style kernels underutilizes the GPU by 10–20×. See "Row-Reduction Strategy".
8. **Host-side readback is NOT automatic** — after a kernel writes to a `TensorViewMut<[T]>` backed by a host vector, you must call `d_out.copy_to_host(&mut h_out).unwrap()` before reading or persisting `h_out`. Dropping the view does not trigger readback. Silent all-zeros output otherwise. See "Host-Side Patterns".
9. **Port the algorithm before the idioms** — fuse multi-pass reductions (max+sum, mean+var, logsumexp) into one pass; pick vector width (Float4) before writing the loop; pick the tile size with register tiling before writing GEMM. Only then apply SeGuRu's chunk_mut / reshape_map / warp-redux patterns. A clean SeGuRu rendering of a bad algorithm will not reach parity.
10. **Optimize against raw custom CUDA parity first** — PyTorch eager often measures library dispatch, not a comparable hand-written kernel. Treat PyTorch numbers as context unless the raw CUDA baseline is also fast.
11. **Keep benchmark/example code safe** — use generated maps (`reshape_map!`, `MapContinuousLinear`, `Map2D`) and checked host helpers such as `try_cast_slice::<Float4>()`; keep `unsafe` inside reviewed library abstractions or unavoidable external FFI boundaries.

## Kernel Signature Translation

### Parameters
```
CUDA:    __global__ void kernel(float *a, const float *b, int n)
SeGuRu:  #[gpu::cuda_kernel]
         pub fn kernel_name(b: &[f32], a: &mut [f32], n: u32)
```

Rules:
- `__global__` → `#[gpu::cuda_kernel]`
- Read-only pointer → `&[f32]`
- Read-write pointer → `&mut [f32]`
- `int`/`unsigned int` → `u32` or `i32` (match the CUDA type's signedness)
- `float` scalar → `f32`
- Dynamic shared memory → add `(dynamic_shared)` to attribute

### Thread Intrinsics
| CUDA | SeGuRu | Return type |
|------|--------|-------------|
| `threadIdx.x` | `thread_id::<DimX>()` | `u32` |
| `blockIdx.x` | `block_id::<DimX>()` | `u32` |
| `blockDim.x` | `block_dim::<DimX>()` | `u32` |
| `gridDim.x` | `grid_dim::<DimX>()` | `u32` |
| `__syncthreads()` | `sync_threads()` | — |

All return `u32` — keep arithmetic in `u32`, cast to `usize` only at array index sites.

## Memory Write Patterns

### Rule: All mutable writes go through `chunk_mut`

SeGuRu's memory safety model requires wrapping mutable arrays:

```rust
// 1D output — one element per thread
let mut out = chunk_mut(out, MapContinuousLinear::new(1));
out[0] = value;  // LOCAL index, not global

// 2D output — one element per thread in 2D grid
let mut out = chunk_mut(out, Map2D::new(width as usize));
out[(0, 0)] = value;  // LOCAL 2D index
```

### Read-write arrays (e.g., `c *= beta; c += ...`)
```rust
let mut c = chunk_mut(c, Map2D::new(n as usize));
let mut val = c[(0, 0)] * beta;  // READ from chunk
// ... compute ...
c[(0, 0)] = val;  // WRITE back to chunk
```

### Grid-stride loops DO NOT work with `chunk_mut`
CUDA grid-stride writes (`for i = idx; i < n; i += stride { c[i] = ... }`) are incompatible. Instead, launch enough threads for 1 element per thread:
```rust
let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
if idx < n {
    out[0] = ...;  // local index
}
```

## Inner Loop Optimization: Subslice Pattern

This is the **single most impactful optimization**. It changed GEMM from 3× overhead to 1.76×.

### Problem: Per-element indexing
```rust
// SLOW — bounds check on every a[] and b[] access
let mut k: u32 = 0;
while k < nk {
    sum += a[(i * nk + k) as usize] * b[(k * nj + j) as usize];
    k += 1;
}
```

### Solution: Subslice for row access
```rust
// FAST — one bounds check for the entire row slice
let a_row: &[f32] = &a[(i * nk) as usize..((i + 1) * nk) as usize];
let mut b_idx = j as usize;
for a_val in a_row {
    sum += a_val * b[b_idx];
    b_idx += nj as usize;
}
```

### When NOT to use subslice
- **One element per row** (column access): Creating a subslice per row (`&data[i*m..(i+1)*m]` → `row[j]`) adds overhead when you only read one element. Keep per-element `data[(i*m+j) as usize]` with `.ldcs()` instead.
- **Iterators with .enumerate()**: Can have ~20% overhead vs `while` loops in some patterns (noted in llm-rs). Benchmark both.

### Warp-strided subslice pattern (from llm-rs layernorm)
When threads in a warp cooperatively process a row with stride:
```rust
let x = &inp[idx_C as usize..(idx_C + C) as usize];  // subslice the row ONCE
let mut local_sum = 0.0f32;
let mut i = lane_id;
while i < C {
    local_sum += x[i as usize];  // index into the subslice, not the full array
    i += warp.size();
}
```
This is used extensively in llm-rs for layernorm, softmax, and attention kernels.

### Examples by kernel type

**Matrix multiply (GEMM):**
```rust
// A is accessed by row (contiguous) → subslice
// B is accessed by column (stride) → per-element
let a_row = &a[(i * nk) as usize..((i + 1) * nk) as usize];
let mut b_idx = j as usize;
for a_val in a_row {
    sum += a_val * b[b_idx];
    b_idx += nj as usize;
}
```

**Matrix-vector (Ax):**
```rust
// A row × x vector — both are contiguous reads
let a_row = &a[(i * ny) as usize..((i + 1) * ny) as usize];
for (j_idx, a_val) in a_row.iter().enumerate() {
    sum += a_val * x[j_idx];
}
```

**Transpose matrix-vector (A^T x) — cannot subslice:**
```rust
// Column access pattern — keep per-element
let mut i: u32 = 0;
while i < nx {
    sum += a[(i * ny + j) as usize] * x[i as usize];
    i += 1;
}
```

**Symmetric rank-k (A·Aᵀ):**
```rust
// Both a_row_i and a_row_j are contiguous rows
let a_row_i = &a[(i * nj) as usize..((i + 1) * nj) as usize];
let a_row_j = &a[(j * nj) as usize..((j + 1) * nj) as usize];
for k in 0..nj as usize {
    val += alpha * a_row_i[k] * a_row_j[k];
}
```

## Row-Reduction Strategy: Pick Parallelism Per Row, Not Per Thread

**The #1 mistake on reduction kernels** (`sum`, `mean`, `norm`, `max`,
`softmax`, …). The "subslice for row" pattern above is correct for the
*inner loop*, but it does NOT answer the question "how many threads should
cooperate on one row?".

Wrong choice here is a 10–20× performance cliff, not a 5% overhead.

### The pitfall: 1 thread per row

For `y[B] = sum(x[B, D], dim=-1)` with `B=128, D=16384`, a naive elementwise
template reuse writes:

```rust
// ❌ WRONG for small-B reductions: only 128 threads run — GPU is ~1% busy
let row = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
if row < B {
    let x_row = &x[(row * D) as usize..((row + 1) * D) as usize];
    let mut sum = 0.0f32;
    for v in x_row { sum += v; }
    out[0] = sum;
}
```

This is functionally correct but ~17× slower than raw CUDA in practice
(measured on `sum_dim` in KernelBench-B). `B=128` threads can't saturate an
A100; every row sequentially scans 16K floats on a single thread.

### Decision table

Pick the strategy based on `B` (row count) and `D` (row width):

| `B` (rows) | `D` (row width) | Strategy                      | Grid / block                     |
|-----------:|----------------:|-------------------------------|----------------------------------|
| ≥ ~10×SM   | small (≤ ~128)  | 1 thread per row              | `gs = B/bs, bs = 256`            |
| moderate   | ≤ ~1024         | 1 warp per row                | `gs = B/warps_per_block`, `bs = 32 * warps_per_block` |
| small (≤ ~1K) | large (≥ ~1K)| **1 block per row**           | `gs = B`, `bs = 256 or 512`      |
| small      | huge (≥ ~64K)   | 1 block per row + float4 loads| `gs = B`, `bs = 512`             |

Rule of thumb: you want ≥ ~4× SM count worth of *resident threads*. An A100
has 108 SMs, so aim for at least ~50K–100K concurrent threads. `B × threads_per_row`
should land in that range.

### Pattern: 1 block per row (the right sum_dim)

```rust
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[f32], y: &mut [f32], B: u32, D: u32) {
    let mut out = chunk_mut(y, MapContinuousLinear::new(1));
    let row = block_id::<DimX>();                         // 1 block = 1 row
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();

    // Each thread strides across its row accumulating a partial sum.
    let x_row = &x[(row * D) as usize..((row + 1) * D) as usize];
    let mut partial = 0.0f32;
    let mut i = tid as usize;
    while i < D as usize {
        partial += x_row[i];
        i += bdim as usize;
    }

    // Warp reduce, then cross-warp via shared memory (see "Warp reductions"
    // and "Tree reduction in shared memory" below).
    // ... block-reduce `partial` into one scalar in thread 0 ...

    if tid == 0 {
        out[0] = /* block_sum */;
    }
}

// Host: launch with gs = B, bs = 256.
let cfg = gpu_host::gpu_config!(bb, 1, 1, 256, 1, 1, 0);
```

### Pattern: 1 warp per row (when D is moderate)

Used in `layer_norm.rs` KernelBench-B. Each block hosts `warps_per_block` warps,
and each warp owns one row. See `examples/kernelbench-b/src/layer_norm.rs`
for a working template using `warp.subgroup_id()` to index within the block.

### Red flags that you chose wrong

- `gs = B.div_ceil(256)` on a reduction where `B < ~10_000` → 1 thread/row, too few threads.
- No shared memory, no `warp.redux`, no tree reduction inside the kernel → you're not reducing cooperatively.
- Only `block_id` is used to identify the output row, no `thread_id`/`lane` cooperation inside → fine only if `D` is tiny.

If you wrote a reduction kernel that compiles and passes correctness but
looks suspiciously similar to your elementwise template, re-read this section.

### Always vectorize when `D % 4 == 0` (Float4 loads)

Reduction kernels are memory-bound. A cooperative row-reduction that reads
scalars via `x_row[i]` in the inner loop emits one `ld.global.f32` per
element. Switching the kernel signature to take `x: &[Float4]` (with `D4 = D/4`)
cuts the instruction count 4× and measurably closes the gap to raw CUDA:

| Kernel     | Scalar load time | Float4 load time | Speedup |
|------------|-----------------:|-----------------:|--------:|
| `sum_dim`  | 225 µs           | 163 µs           | 1.38×   |
| `layer_norm` | 299 µs         | 243 µs           | 1.23×   |
| `l2_norm`  | 299 µs           | 243 µs           | 1.23×   |

(Softmax is an exception: the online max+sum state can't be vectorized
because each lane's `(local_max, local_sum)` pair depends sequentially
on every element it sees.)

**Recipe**:

```rust
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[Float4], y: &mut [f32], D4: u32) {
    // ...
    let x_row = &x[(row * D4 as usize)..((row + 1) * D4 as usize)];
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < D4 {
        let v = x_row[i as usize];
        acc += v.x + v.y + v.z + v.w;
        i += block_dim::<DimX>();
    }
    // ... block reduce as before
}
```

**Host side**: prefer the checked zero-copy TensorView cast when the device
tensor already exists as contiguous `f32` data:

```rust
let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
let d_x4 = d_x.try_cast_slice::<Float4>().expect("x must be Float4-aligned");
sum_dim_kernel::launch(cfg, ctx, md, &d_x4, &mut d_y, D4).unwrap();
```

If constructing data that is naturally grouped as vectors, building a true
`Vec<Float4>` is also safe:

```rust
let h_x4: Vec<Float4> = h_x
    .chunks_exact(4)
    .map(|c| Float4::new([c[0], c[1], c[2], c[3]]))
    .collect();
let d_x = ctx.new_tensor_view(h_x4.as_slice()).unwrap();
```

Preconditions: `D % 4 == 0` and the base pointer must be 16-byte aligned
(CUDA guarantees this for `cudaMalloc`-allocated tensors). `try_cast_slice`
checks byte divisibility and alignment, and rebuilds the vector-view slice
metadata with length `len / 4`.

**When not to use Float4**:
- `D % 4 != 0` — falls back to a scalar tail, rarely worth the extra code.
- Inner loop has nonlinear dependencies between the 4 lanes (online softmax,
  online Welford — but the outer `sum`/`sumsq` accumulators are still
  vectorizable component-wise as `acc += v.x+v.y+v.z+v.w`; it's the
  *exp/update* that can't be).
- Kernel already compute-bound (GEMM inner loop reading smem) — Float4
  helps memory-bound kernels, not compute-bound ones. Caveat: GEMM's
  *global*-memory tile loads benefit from Float4 even though the compute
  is bound by smem/FMA — see the GEMM Recipe "Non-negotiable #4" for the
  measured ~14% win. "Compute-bound" here means the **smem `tile_a[idx]`
  broadcast reads** inside the kk-loop, which `reshape_map!([4]|…)` already
  compiles to `ld.shared.v4.f32`; those should stay scalar-indexed.

### Pitfall: "one scalar per block" output scope

When the output is `y: [B]` (one scalar per row/block), `chunk_mut(y, MapContinuousLinear::new(1))`
is **wrong**. The default scope is `Thread`, which creates
`gridDim * blockDim = B * 256 ≈ 1M` slots — but `y` has only `B` elements.
Result: `CUDA_ERROR_ILLEGAL_ADDRESS` at launch.

**Correct pattern** — chain Grid→Block→Thread scope so every thread sees
the block's single slot:

```rust
let grid2block = build_chunk_scope(Grid, Block);
let block2thread = build_chunk_scope(Block, Thread);
// ...
let mut y_chunk = y
    .chunk_to_scope(grid2block, MapContinuousLinear::new(1))
    .chunk_to_scope(block2thread, MapContinuousLinear::new(1));
if tid == 0 {
    y_chunk[0] = block_sum;
}
```

This is used whenever the kernel writes one value per block — reduction
scalars (`sum`, `max`, `argmax`), row statistics (`row_max`, `row_sum` in
a 2-pass softmax), etc. See `examples/kernelbench-b/src/from_cuda/softmax.rs`
for the canonical example.

## Shared Memory Tiling: The Key to CUDA Parity

**This is the most important pattern for compute-bound kernels.**

Tiled kernels load global memory into shared memory first. Combined with the right chunking pattern, this reduces global-memory bounds-check overhead AND provides data reuse.

### Bounds-check behavior (measured on PTX)

| Access pattern | PTX | Bounds check? |
|---|---|---|
| Global `a[idx]` (raw) | `setp + selp + ld.global.f32` | Yes (always) |
| Global via subslice `&a[row*n..]` + iteration | `ld.global.f32` from local ptr | No (checked once at slice) |
| Shared `*smem[idx]` on `smem_alloc.alloc(n)` | `setp.lt + selp + ld.shared.f32` | Yes |
| Shared `tile[idx]` on `GpuShared<[f32; N]>` | `setp.lt + selp + ld.shared.f32` | Yes (compiler can't prove `ty*16+k < 256`) |
| Shared via `chunk.chunk_mut(reshape_map)[i]` | `st.shared.f32` / `ld.shared.f32` direct | No (chunk map proves slot is in-range) |

### Pattern: reshape_map + chunk_mut for LOADS (eliminates store-side bounds checks)
```rust
let mut tile_a = GpuShared::<[f32; 256]>::zero();
let mut tile_b = GpuShared::<[f32; 256]>::zero();

// Each of 16x16 threads owns ONE disjoint slot (ty*16 + tx) in each tile.
// layout [i0, t0, t1] → memory = t1*16 + t0 = ty*16 + tx.
let load_map = reshape_map!([1] | [16, 16] => layout: [i0, t0, t1]);

for t in 0..num_tiles {
    {
        let mut chunk_a = tile_a.chunk_mut(load_map);
        chunk_a[0] = /* value from global A */;  // NO bounds check on shared write
    }
    {
        let mut chunk_b = tile_b.chunk_mut(load_map);
        chunk_b[0] = /* value from global B */;
    }
    sync_threads();
    // Compute phase ...
}
```

### Honest limitation on COMPUTE phase

The matmul inner loop reads `tile_a[ty*16+k]` (broadcast across tx) and `tile_b[k*16+tx]` (broadcast across ty). These are **broadcast reads** — multiple threads read the same slot. `chunk_mut`'s disjoint-partition model cannot express broadcast reads, so raw indexing `tile_a[idx]` is unavoidable in the compute loop. This means compute reads keep their `setp+selp` bounds check.

**Measured impact (GEMM N=512, A100):**
| Variant | µs | vs CUDA |
|---|---|---|
| Naive global (u32) | 293 | 2.93× |
| Subslice rows | 201 | 2.01× |
| Dynamic smem tiled, raw `*smem[i]` | 195 | 1.95× |
| Static `GpuShared` + reshape_map loads | **182** | **1.82×** |
| CUDA (hand-written) | 100 | 1.00× |

### Closing the remaining gap

The residual 1.82× vs CUDA comes from bounds checks on compute reads. Since these can't be eliminated structurally, the way to close the gap is to **amortize them** via register tiling: each thread computes NxM outputs instead of 1, reusing each shared load across multiple FMAs. This is what `llm-rs::matmul_forward_kernel4` does (8×8 register tile → 64 FMAs per pair of shared loads, achieving CUDA parity despite bounds checks on shared reads).

### When to use shared memory tiling
- **Compute-bound kernels** with inner loops reading global memory (GEMM, convolution)
- Kernels where the same data is read by multiple threads (reuse across threads)
- When the naive kernel shows ≥1.5× overhead vs CUDA

### When NOT to tile
- **Memory-bound kernels** (vector_add, stencils) — already at parity
- Kernels with no data reuse across threads
- Simple element-wise operations

### Static vs dynamic shared memory

Use `GpuShared::<[T; N]>::zero()` when the tile size is compile-time constant — simpler, no `dynamic_shared` attribute, no smem_bytes in launch config. Use `smem_alloc.alloc::<T>(n)` (with `#[gpu::cuda_kernel(dynamic_shared)]`) when size depends on launch-time parameters.

```rust
// Static
let mut smem = GpuShared::<[f32; 256]>::zero();
let mut chunk = smem.chunk_mut(reshape_map!([1] | [16, 16] => layout: [i0, t0, t1]));
chunk[0] = value;

// Dynamic
#[gpu::cuda_kernel(dynamic_shared)]
pub fn kernel(...) {
    let smem = smem_alloc.alloc::<f32>(block_dim::<DimX>() as usize);
    let mut chunk = smem.chunk_mut(MapContinuousLinear::new(1));
    chunk[0] = value;
}
```

### Tree reduction in shared memory
Use the `reshape_map!` pattern from the existing `reduce_per_grid`:
```rust
for order in (0..16).rev() {
    let stride = 1u32 << order;
    if stride >= bdim { continue; }
    let mut chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
    if tid < stride {
        let right = chunk[1];
        let left = chunk[0];
        chunk[0] = left + right;
    }
    sync_threads();
}
```

## Atomics

### Global atomics
```rust
use gpu::sync::Atomic;

let atomic = Atomic::new(data);         // wrap mutable slice once
atomic.index(i).atomic_addi(1u32);      // index then operate
atomic.index(j).atomic_addf(1.0f32);    // float version
```

### Shared memory atomics
```rust
use gpu::sync::SharedAtomic;

let smem_atomic = SharedAtomic::new(smem_slice);  // wrap shared allocation
smem_atomic.index(bin).atomic_addi(1u32);          // index then operate
```

Note: `SharedAtomic::new` takes `&mut GpuShared<[T]>` (from `smem_alloc.alloc`), not per-element references.

## Vectorized Access (Float4)

`Float4` performs coalesced 128-bit loads/stores. For memory-bound kernels with
contiguous stride-1 access and `D % 4 == 0`, switching the kernel signature
from `&[f32]` to `&[Float4]` (and `D` → `D4 = D/4`) cuts the inner-loop
load-instruction count 4×.

```rust
use gpu::Float4;

#[gpu::cuda_kernel]
pub fn kernel(inp: &[Float4], out: &mut [Float4], N4: u32) {
    let mut out = chunk_mut(out, MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    if idx < N4 {
        let v = inp[idx as usize];
        out[0] = Float4::new([v[0] + 1.0, v[1] + 1.0, v[2] + 1.0, v[3] + 1.0]);
    }
}
```

**Lane access**: `v[0]..v[3]` (or `.x / .y / .z / .w` if the wider SeGuRu prelude is in scope).

**Host pattern**: see `### Always vectorize when D % 4 == 0 (Float4 loads)`
inside the Row-Reduction Strategy section above for the checked zero-copy
`try_cast_slice::<Float4>()` path and the safe `Vec<Float4>` construction path.

**When not to use** — see the "When not to use Float4" bullets in Row-Reduction
Strategy. The short version: online algorithms with sequential per-element
state (softmax max+sum, Welford) cannot vectorize the state update (only the
outer accumulators). For compute-bound GEMM, keep shared-memory compute reads
scalar; use `Float4` only for the global tile loads feeding shared memory.

## Cache Hints

For streaming access patterns where data won't be reused (from llm-rs layernorm):
```rust
use gpu::CacheStreamLoadStore;

let val = x[i as usize].ldcs();   // cache-streaming load (bypass L1)
out[idx].stcs(value);              // cache-streaming store
```

**High impact for column-stride access patterns** — reduced corr/covar overhead from 3.09× to 2.27× (26% faster). Use `.ldcs()` for read-only arrays accessed with non-sequential strides (column traversals, strided patterns).

## Warp Operations

### Warp reductions (from llm-rs)
```rust
use gpu::cg::{CGOperations, ReduxAdd, ThreadWarpTile};

let warp = ThreadWarpTile::<32>;
let lane_id = warp.thread_rank();

// Warp-cooperative sum
let mut local_sum = 0.0f32;
let mut i = lane_id;
while i < C {
    local_sum += x[i as usize];
    i += warp.size();
}
let sum: f32 = warp.redux(ReduxAdd, local_sum);
```

### Warp metadata
```rust
let warp = ThreadWarpTile::<32>;
warp.thread_rank()      // lane ID within warp (0-31)
warp.meta_group_size()  // number of warps per block
warp.subgroup_id()      // warp index within block
warp.size()             // warp size (32)
```

## Host-Side Patterns

### Basic launch
```rust
gpu_host::cuda_ctx(0, |ctx, m| {
    let d_inp = ctx.new_tensor_view(h_data.as_slice()).unwrap();
    let mut d_out = ctx.new_tensor_view(h_out.as_mut_slice()).unwrap();
    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
    kernel::launch(config, ctx, m, &d_inp, &mut d_out, n as u32).unwrap();
    d_out.copy_to_host(&mut h_out).unwrap();  // ← REQUIRED, see Golden Rule #8
});
```

### Golden Rule #8 (host-side) — readback is NOT automatic

> `new_tensor_view(h_vec.as_mut_slice())` snapshots the host buffer to device
> at construction. After a kernel writes to that `TensorViewMut`, you **must**
> call `d_out.copy_to_host(&mut h_out).unwrap()` before reading or persisting
> `h_out`. Dropping the `TensorViewMut` does **not** trigger a readback.
>
> Symptom if omitted: `h_out` stays all zeros (or its initial value), and
> your downstream verification silently reports max-abs-err equal to the
> magnitude of the expected output.

### Host recipe — bench + I/O scaffold (copy-paste starting point)

```rust
pub fn run(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    in_dir: &std::path::Path, out_dir: &std::path::Path,
    iters: usize, shape: &[usize],
) -> (f64, f64) {
    use std::time::Instant;
    let n = shape.iter().product::<usize>();
    // 1. Read input(s) from disk into host buffers.
    let h_x = super::read_bin(&in_dir.join("x.bin"), n);
    let mut h_y = vec![0f32; n];                     // (2) allocate output
    // 3. Host → device. These calls copy in.
    let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
    let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
    // 4. Launch config (pick block/grid for YOUR kernel).
    let nn = n as u32;
    let bs: u32 = 256;
    let gs: u32 = nn.div_ceil(bs);
    // 5. Warmup + timed loops. gpu_config! is non-Copy → recreate each iter.
    let wt = Instant::now();
    for _ in 0..5 {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        my_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let warmup_us = wt.elapsed().as_micros() as f64 / 5.0;

    let t = Instant::now();
    for _ in 0..iters {
        let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
        my_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, nn).unwrap();
    }
    ctx.sync().unwrap();
    let kernel_us = t.elapsed().as_micros() as f64 / iters as f64;

    // 6. REQUIRED: device → host before reading h_y. Skipping this silently
    //    produces all-zeros output.
    d_y.copy_to_host(&mut h_y).unwrap();
    // 7. Persist output.
    super::write_bin(&out_dir.join("y.bin"), &h_y);

    (kernel_us, warmup_us)
}
```

### Static block dimensions with `@const`
From llm-rs — compile-time block size enables better optimization:
```rust
const BSIZE: usize = 256;
let config = gpu_host::gpu_config!(grid as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
```

### Tensor sub-indexing
Split tensors into sub-views for multi-buffer operations (from llm-rs attention):
```rust
let (mut q, mut rest) = qkvr.split_at_mut(bsc_len);
let (mut k, mut v) = rest.split_at_mut(bsc_len);
// Now q, k, v are separate TensorViewMut pointing into the same allocation
```

### cuBLAS integration
For large matmul, use cuBLAS directly alongside SeGuRu kernels (from llm-rs):
```rust
unsafe {
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        C as i32, N as i32, OC as i32,
        &one, weight.as_devptr() as _, C as i32,
        dout.as_devptr() as _, OC as i32,
        &zero, dinp.as_devptr() as _, C as i32);
}
```

## Multi-Kernel Benchmarks

SeGuRu supports multiple kernel launches within a single `cuda_ctx`:

```rust
gpu_host::cuda_ctx(0, |ctx, m| {
    // allocate tensors...

    for t in 0..tsteps {
        let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
        kernel1::launch(cfg, ctx, m, &d_a, &mut d_b, n).unwrap();
        let cfg = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
        kernel2::launch(cfg, ctx, m, &mut d_a, &d_b, n).unwrap();
    }

    d_a.copy_to_host(&mut h_a).unwrap();
});
```

Note: `gpu_config!` returns a non-Copy type — recreate it before each `launch` call.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Using `usize` for kernel params | 30-45% slower | Use `u32`, cast `as usize` at index sites |
| Per-element `a[i*n+k]` in inner loop | 2-3× slower than CUDA | Use subslice `&a[i*n..(i+1)*n]` + iterator |
| `MapLinear` instead of `MapContinuousLinear` | Slightly slower (u64 index math) | Use `MapContinuousLinear::new(1)` |
| Kernel name = crate name | MLIR parse error | Rename kernel (e.g., `histogram_kernel`) |
| `cuda_ctx` in `#[test]` body | MLIR symbol `{}` error | Use helper function called from test |
| `chunk_mut` with global index | Runtime bounds panic | Use local index `c[0]` or `c[(0,0)]` |
| Grid-stride write loop | Runtime bounds panic | Launch enough threads, 1 element per thread |
| `gpu_config!` reuse in loop | Compile error (not Copy) | Recreate before each `launch` |
| Aliased read+write same array | Need separate params | Pass as both `a_read: &[f32]` and `a_write: &mut [f32]` |
| Using iterators everywhere | Can be 20% slower (see llm-rs note) | Benchmark both; use `while` loop if iterator is slower |
| 1-thread-per-row reduction | 10–20× slower than raw CUDA on small-B reductions | Use 1-block-per-row (or 1-warp-per-row when D ≤ 1024); see "Row-Reduction Strategy" |
| `chunk_mut(y, MapContinuousLinear::new(1))` for per-block scalar output | `CUDA_ERROR_ILLEGAL_ADDRESS` at launch | Chain Grid→Block→Thread scope via `chunk_to_scope`; see "one scalar per block output scope" pitfall |
| Missing `d_out.copy_to_host(&mut h_out)` | All-zeros output, no warning | Add the call before reading/persisting `h_out`; Golden Rule #8 |

## Performance Expectations (A100, naive kernels)

| Kernel Category | Expected Ratio vs CUDA | Dominant Factor |
|----------------|----------------------|-----------------|
| Large compute, row access (syrk, syr2k) | **1.00×** | Compute-bound, subslice amortizes checks |
| Stencil (fdtd2d, jacobi2d) | **1.0-1.3×** | Memory-bound, few accesses per thread |
| Reductions with fused stats + Float4 (layernorm) | **1.05×** | Algorithm fusion + vector loads |
| GEMM-family with subslice | **1.7-2.0×** | Column-stride `b[]` still per-element |
| Matrix-vector | **1.7-1.8×** | Mix of row (fast) and column (slow) access |
| Column-heavy reductions (corr, covar) | **~2.3×** | Column-stride access; `.ldcs()` helps significantly |
| Iterative with many launches | **1.9×** | Launch overhead (4.7 vs 2.0 µs) accumulates |
| Mechanical 1:1 port of memory-bound kernel | **1.3-1.5×** | See the [LayerNorm case study](cuda-to-seguru-porting-progress.md#case-study-pytorch-layernorm-algorithm--idioms) — port the algorithm, not syntax |

## GEMM / Matmul Recipe

For dense FP32 matmul `y[M, N] = x[M, K] · W[N, K]^T` (or equivalent). The
current KernelBench-C design is documented in
[`docs/kernelbench-c-float4-16x16-design.md`](kernelbench-c-float4-16x16-design.md);
this section keeps only the reusable porting recipe.

**Tile parameters (the defaults that work for `M, N, K` all multiples of 128/128/8):**

```
BM = BN = 128       // output tile per block
BK = 8              // K-dimension chunk per shared-load
TM = TN = 8         // register-tile per thread (8×8 = 64 accumulators)
blockDim = (16, 16) // 256 threads = 16 tiles of size TM × TN along each axis
```

Each thread owns an 8×8 output sub-tile: **64 FMAs per pair of shared-memory
loads**, which amortizes the bounds-checked broadcast reads that SeGuRu's
disjoint-partition model forces in the compute phase (see "Honest limitation
on COMPUTE phase" above).

**Signature and skeleton**:

Kernel takes `x` / `w` as `&[Float4]` (not `&[f32]`). Each thread's 4-wide
K slice — `a_col = (tid & 1) << 2` — is always 16-byte aligned because
`BK=8` and `K` is a multiple of `BK`. Using `&[Float4]` makes the codegen
emit `ld.global.v4.f32` for the tile load (one 128-bit load replaces four
scalar 32-bit loads). See "Non-negotiables" #4 below.

```rust
const BM: u32 = 128; const BN: u32 = 128; const BK: u32 = 8;
const TM: u32 = 8;   const TN: u32 = 8;
const BDIM_X: u32 = 16; const BDIM_Y: u32 = 16;

#[gpu::cuda_kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
pub fn gemm_kernel(x: &[Float4], w: &[Float4], y: &mut [f32], K: u32) {
    let tx = thread_id::<DimX>();  let ty = thread_id::<DimY>();
    let bm = block_id::<DimY>() * BM;
    let bn = block_id::<DimX>() * BN;
    let tid = ty * BDIM_X + tx;                 // 0..256
    let a_row = tid >> 1;                       // 0..128 (2 threads per row)
    let a_col = (tid & 1) << 2;                 // 0 or 4 — 4-wide K slice

    let mut tile_a = GpuShared::<[f32; (BM * BK) as usize]>::zero(); // 128*8=1024
    let mut tile_b = GpuShared::<[f32; (BN * BK) as usize]>::zero();

    // K-major shared-memory tile layout. For lane i0, this stores:
    //   offset = (a_col + i0) * BM + a_row
    // so compute reads use tile[k][row_or_col], matching raw CUDA.
    let load_map = reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0]);

    // Per-thread disjoint 8×8 slot of the output. Decompose linear index
    // fastest→slowest: [j (stride 1), tx, bid_x, i, ty, bid_y].
    let out_map = reshape_map!(
        [8, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let mut y_thread = chunk_mut(y, out_map);

    let mut acc = [[0.0f32; TN as usize]; TM as usize];

    let mut tstep = 0u32;
    while tstep < K / BK {
        // K_f4 = K/4 (row stride in Float4 units), k_base4 = tstep*(BK/4).
        let k_base4 = tstep * (BK >> 2);
        {
            let mut ca = tile_a.chunk_mut(load_map);
            let v: Float4 = x[((bm + a_row) * (K >> 2) + k_base4 + (a_col >> 2)) as usize];
            ca[0] = v[0]; ca[1] = v[1]; ca[2] = v[2]; ca[3] = v[3];
        }
        { /* same pattern for tile_b from `w`, with bn instead of bm */ }
        sync_threads();

        // Compute — fully `unroll!`ed so the 64-wide accumulator stays in regs.
        let row_off = (ty * TM) as usize;
        let col_off = (tx * TN) as usize;
        unroll! { for kk in 0..8 {
            let mut a_reg = [0.0f32; 8];
            let mut b_reg = [0.0f32; 8];
            for ii in 0..8usize { a_reg[ii] = tile_a[kk * BM as usize + row_off + ii]; }
            for jj in 0..8usize { b_reg[jj] = tile_b[kk * BN as usize + col_off + jj]; }
            unroll! { for ii in 0..8 {
                let ai = a_reg[ii];
                unroll! { for jj in 0..8 {
                    acc[ii][jj] += ai * b_reg[jj];
                }}
            }}
        }}
        sync_threads();
        tstep += 1;
    }

    // Epilogue (fused bias/activation/etc.) and store via y_thread[(j, i)].
    unroll! { for i in 0..8 { unroll! { for j in 0..8 {
        y_thread[(j as u32, i as u32)] = /* acc[i][j] + epilogue */;
    }}}}
}

// Host launch
let gx = (N as u32) / BN;
let gy = (M as u32) / BM;
let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
// The nvvm_launch_bound annotation above must match this 16x16x1 block shape.
// Reinterpret f32 device views as Float4 views with checked length/alignment.
let d_x4 = d_x.try_cast_slice::<Float4>().expect("x must be Float4-aligned");
let d_w4 = d_w.try_cast_slice::<Float4>().expect("w must be Float4-aligned");
gemm_kernel::launch(cfg, ctx, md, &d_x4, &d_w4, &mut d_y, kk).unwrap();
```

**Non-negotiables**:

1. **K-major shared-memory tile layout** — store shared tiles physically as
   `As[k_local, m_local]` and `Bs[k_local, n_local]`, using
   `reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0])`. Each thread's
   `Float4` load still writes four disjoint slots, but the compute loop reads
   CUDA-style `tile[k][row_or_col]`. Float4 global loads plus a 16x16 block are
   not sufficient if shared memory stays in the old row-major layout.
2. **`unroll!` the kk / ii / jj loops** so the 64 accumulators + 8-wide
   `a_reg` / `b_reg` stay in registers. Without `unroll!` they spill to
   local memory and you lose the whole point of register tiling.
3. **Compute reads remain scalar `tile_a[idx]` / `tile_b[idx]`** — multiple
   threads reading the same slot (broadcast) cannot be expressed as a disjoint
   partition, so these keep their `setp+selp` bounds check. Use the K-major
   indices `tile_a[kk * BM as usize + row_off + ii]` and
   `tile_b[kk * BN as usize + col_off + jj]`; the 8x8 register tile amortizes
   this cost (64 FMAs per shared load).
4. **Take `x`/`w` as `&[Float4]`, not `&[f32]`** — four consecutive scalar
   `ld.global.f32` per thread per K-tile will **not** be auto-coalesced into
   a `ld.global.v4.f32` by the current SeGuRu codegen (tensor-view indexing
   forces u32→u64 promotion per element, breaking the peephole). Using
   `&[Float4]` emits the 128-bit load directly and drops the tile-load
   `ld.global` count from 8 scalar to 2 vector ops, matching nvcc's PTX shape.
   The host-side reinterpret is zero-cost (no realloc, no H2D copy); the
   device pointer is identical.
5. **Keep the 16x16/8x8 geometry as one contract** — `BDIM_X = 16`,
   `BDIM_Y = 16`, `TM = 8`, and `TN = 8` together produce the 128x128 output
   tile. If any value changes, retune the full tile shape and update the
   launch-bound annotation in the same patch.
6. **Use `#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]` only when it matches
   the real launch** — it is a register/occupancy contract with NVVM/ptxas, not
   a replacement for `gpu_config!(..., 16, 16, 1, ...)`.
7. **Do not reintroduce `.open_tile()` or `.row_mut()` for this recipe** unless
   the core API is separately justified and benchmarked. Direct `chunk_mut`
   indexing expresses the same output ownership without restoring invasive row
   view/codegen helpers.

**When to deviate**:

- **Non-multiples of 128/128/8**: either pad on the host or write a guarded
  tail kernel. The Recipe as stated assumes aligned shapes.
- **Smaller problems (M, N ≤ 256)**: fall back to a 32×32 tile with 4×4
  register tile — 128×128 tiles waste SMs when `gridDim < 2 × num_SMs`.
- **Use cuBLAS**. For pure dense FP32 matmul at meaningful sizes, cuBLAS
  with TF32 acceleration beats any hand-written FP32 kernel by ~3–4× because
  it uses tensor cores. User-written SeGuRu matmul is appropriate only when
  fused with a non-standard epilogue, for small sizes, or when precise FP32
  is required.

For current benchmark snapshots and branch-specific conclusions, use
[`docs/cuda-to-seguru-porting-progress.md`](cuda-to-seguru-porting-progress.md)
and [`docs/kernelbench-c-float4-16x16-design.md`](kernelbench-c-float4-16x16-design.md).

## Convolution Recipe

For 2-D convolution `y[B, Cout, Ho, Wo] = conv2d(x[B, Cin, H, W], W[Cout, Cin, Kh, Kw])`.
First match the raw CUDA kernel's thread geometry. Do not introduce a "safe"
shared-memory patch if it changes the launch shape or leaves many threads idle.

**Validated 3x3 no-padding pattern** (`conv_relu_biasadd`):

- Use `BDIM_X = BDIM_Y = 16`, one output pixel per thread.
- Match raw CUDA's output geometry:
  - 3-D image grid: launch `(Wo.div_ceil(16), Ho.div_ceil(16), B * Cout)`;
    `z = blockIdx.z`, `b = z / Cout`, `co = z % Cout`, and output offset is
    `(z * Ho + ho) * Wo + wo`. Store through
    `reshape_map!([1] | [(gx, Wo), (gy, Ho), gz] => layout: [i0, t0, t1, t2])`
    so generated map code invalidates edge-tile threads; do not add a manual
    unsafe `ScopeUniqueMap` in benchmark/example code.
  - Flattened row grid: launch `(Wo.div_ceil(16), (B * Cout * Ho).div_ceil(16), 1)`;
    `row = blockIdx.y * 16 + threadIdx.y`, `bco = row / Ho`, `ho = row % Ho`,
    `b = bco / Cout`, `co = bco % Cout`, and store with `Map2D::new(Wo)`.
- Guard edge threads with `wo < Wo` plus either `ho < Ho` for the 3-D grid or
  `row < B * Cout * Ho` for the flattened grid.
- In the 3x3 body, row-slice once per input channel:
  `x0 = &x[x_ci..]`, `x1 = &x[x_ci + W..]`, `x2 = &x[x_ci + 2*W..]`, then emit
  the nine FMAs. Avoid recomputing full `((ci * H + ho + kh) * W + wo + kw)`
  addresses in the innermost loop.

**Why not the old 14x14 smem patch?** It was memory-safe, but for
`Ho=Wo=126` it launched a 9x9 block grid instead of raw CUDA's 8x8 grid, so it
ran 26.6% more blocks. Only 196/256 threads produced outputs, and each channel
slab added barriers. Replacing it with the direct 16x16 pattern plus row slices
changed `conv_relu_biasadd` from 1.317x slower than raw CUDA to parity:
`seguru=103.2 ms`, `cuda=102.9 ms`, `seguru_from_cuda=103.5 ms`. Applying the
same geometry rule to `conv_relu_hardswish` changed the SeGuRu-from-CUDA ratio
from 1.312x to 0.986x.

Use shared-memory input tiling only after measuring that it improves the raw
CUDA algorithm you are matching. If it changes occupancy, block count, or active
thread fraction, re-check ptxas/SASS before keeping it.

## Softmax Recipe

Softmax over a row of length D (tensor shape `[B, D]`, softmax over dim=1).
This is the pattern for any row-wise normalizing reduction (softmax,
log_softmax, masked attention softmax, etc.). Empirically softmax was the
weakest point of KernelBench-B (SeGuRu 0.71× PyTorch). The recipe below produces
CUDA-parity kernels.

**Non-negotiables (correctness)**:

1. **Subtract the row max before `exp()`.** Never compute `exp(x)` directly.
   Raw exp overflows for `x > 88.7` (fp32) and saturates to `+inf`. The
   identity `softmax(x) = softmax(x - max(x))` is free and mandatory.
2. **Accumulate `max` and `sum` in fp32** even if input is fp16/bf16.
   Convert on load, convert on store.
3. **Identity for max-reduction is `f32::NEG_INFINITY`**, not `0.0`. Threads
   with `idx >= D` in the tail must contribute `-inf` to the max and `0.0`
   to the sum (already handled if you use the online-softmax loop below).
4. **Masked softmax fully-masked row** (all positions masked → sum=0 after
   exp) produces NaN. Decide explicitly: output zeros, uniform `1/D`, or NaN.
   Document the choice.

**Use the single-kernel online (Milakov-Gimelshein) formulation** — it's
strictly better than the 2-kernel stats-then-apply split. Single-kernel
saves one kernel launch plus the round-trip through `row_max[]` and
`row_sum[]` global buffers. See `examples/kernelbench-b/src/from_cuda/softmax.rs`.

**Decomposition**:

- Row length `D <= 32`: one warp per row. `blockDim = (32, num_rows, 1)`.
  No smem needed; warp reductions handle everything.
- Row length `32 < D <= 1024`: one block per row, `blockDim = 256`
  (or 128 for low-D). Grid = `(B, 1, 1)`. Uses one smem slot per warp
  (`num_warps = blockDim / 32`).
- Row length `D > 1024`: split-row, multi-block reduction. Two-stage
  kernel (partial max+sum per chunk, then combine). Rare in practice —
  attention seqlens ≤ 4096 fit a single block with `blockDim=256` and
  stride loop.

**Canonical SeGuRu shape (one block per row, `BLOCK=256`)**:

```rust
#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[f32], y: &mut [f32], D: u32) {
    let warp = ThreadWarpTile::<32>;
    let block2warp = build_chunk_scope(Block, warp);
    let warp2thread = build_chunk_scope(warp, Thread);

    let tid = thread_id::<DimX>();
    let lane_id = warp.thread_rank();
    let num_warps = warp.meta_group_size(); // BLOCK / 32

    // Reserve 32 slots (max warps for BLOCK=1024). Unused slots padded with identity.
    let mut smem_max = GpuShared::<[f32; 32]>::zero();
    let mut smem_sum = GpuShared::<[f32; 32]>::zero();

    // Golden Rule #6: subslice the row for O(1) row-read bounds checks.
    let row = block_id::<DimX>() as usize;
    let x_row = &x[(row * D as usize)..((row + 1) * D as usize)];

    // --- Pass 1: online max+sum (one read of x_row per thread, strided) ---
    let mut local_max = f32::NEG_INFINITY;  // identity for max
    let mut local_sum = 0.0f32;              // identity for sum
    let mut i = tid;
    while i < D {
        let v = x_row[i as usize];
        let old_max = local_max;
        local_max = local_max.max(v);
        // Rescale running sum when max grows, then add new exp term.
        local_sum *= GPUDeviceFloatIntrinsics::exp(old_max - local_max);
        local_sum += GPUDeviceFloatIntrinsics::exp(v - local_max);
        i += block_dim::<DimX>();
    }

    // --- Block-wide max reduce: warp → smem → all-lanes block_max ---
    let warp_max = warp.redux(ReduxMax, local_max);
    {
        let mut s = smem_max
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 { s[0] = warp_max; }
    }
    sync_threads();
    // All warps load the `num_warps` warp-maxes, pad rest with -inf,
    // run another XOR-butterfly. Every lane ends up with block_max.
    let sv = if lane_id < num_warps { smem_max[lane_id as usize] }
             else { f32::NEG_INFINITY };
    let block_max = warp.redux(ReduxMax, sv);

    // Rescale each thread's partial sum from local_max → block_max.
    local_sum *= GPUDeviceFloatIntrinsics::exp(local_max - block_max);

    // --- Block-wide sum reduce (same shape, identity = 0.0) ---
    let warp_sum = warp.redux(ReduxAdd, local_sum);
    {
        let mut s = smem_sum
            .chunk_to_scope(block2warp, MapContinuousLinear::new(1))
            .chunk_to_scope(warp2thread, MapContinuousLinear::new(1));
        if lane_id == 0 { s[0] = warp_sum; }
    }
    sync_threads();
    let sv = if lane_id < num_warps { smem_sum[lane_id as usize] } else { 0.0f32 };
    let block_sum = warp.redux(ReduxAdd, sv);

    let inv_sum = 1.0f32 / block_sum;

    // --- Pass 2: write normalized output (re-read x_row — cheap, in L1/L2) ---
    let out_map = reshape_map!(
        [D / block_dim::<DimX>()]
            | [block_dim::<DimX>(), grid_dim::<DimX>()]
            => layout: [t0, i0, t1]
    );
    let mut y_chunk = chunk_mut(y, out_map);
    let mut slot = 0u32;
    let mut i = tid;
    while i < D {
        y_chunk[slot] = GPUDeviceFloatIntrinsics::exp(x_row[i as usize] - block_max) * inv_sum;
        i += block_dim::<DimX>();
        slot += 1;
    }
}
```

**Pitfalls to avoid**:

- **Two-kernel split (`stats_kernel` + `apply_kernel`)**: measured ~1.3×
  slower than single-kernel fused on `[128, 4096]`. Launch overhead + two
  extra global buffers kills you. Only split if the row must span blocks.
- **Using `0.0` as max identity**: `max(0, -5) = 0` corrupts the result
  for all-negative rows. Always `f32::NEG_INFINITY`.
- **Computing `1.0 / sum` in the inner write loop**: hoist `inv_sum`
  outside. Scalar division is expensive; multiply is ~4× faster.
- **Reshape-map output for non-power-of-two D**: the `[t0, i0, t1]` layout
  above requires `D % BLOCK == 0`. For arbitrary D, fall back to
  `chunk_mut(y, MapContinuousLinear::new(1))` with explicit index
  `row * D + i`.
- **`log_softmax`**: drop the division, subtract `ln(sum)` instead.
  `y[i] = (x[i] - max) - ln_f32(sum)`. Same single-kernel structure.

**Masked softmax**: apply the mask in pass 1 before updating `local_max`:
```rust
let v = if mask_row[i as usize] != 0.0 {
    x_row[i as usize] - f32::INFINITY  // mask → -inf contributes 0 to exp
} else {
    x_row[i as usize]
};
```
Fully-masked row produces `block_sum == 0`. Guard `inv_sum`:
```rust
let inv_sum = if block_sum > 0.0 { 1.0 / block_sum } else { 0.0 };
```

## Launch Config & Occupancy

SeGuRu kernels inherit CUDA's SIMT execution model; the same occupancy and
block-size rules apply. For kernels that need an explicit CUDA launch-bound
contract, use `#[gpu::attr(nvvm_launch_bound(x, y, z, min_blocks_per_sm))]`
and keep it synchronized with the actual `gpu_config!` block dimensions.

**Defaults that almost always work**:

- **1-D elementwise / reductions**: `blockDim = 256`, grid =
  `ceil_div(N, 256)`. Move to 128 if the kernel spills registers; move
  to 512 only if measured better (rare).
- **2-D tile kernels (GEMM, conv)**: `blockDim = (16, 16) = 256` or
  `(32, 8) = 256`. Innermost (`threadIdx.x`) must map to the stride-1
  memory dimension.
- **One-warp-per-row**: only for rows ≤ 32. Above that, a full block
  (8 warps) gives much better latency hiding.
- **Avoid `blockDim = 1024`**: max 2 resident blocks/SM → scheduler has
  only 2 block-state machines → poor latency hiding on memory-bound
  kernels. Prefer 256 or 512.
- **Always a multiple of 32.** A 48-thread block wastes half of the last
  warp (fp32 throughput drops to 48/64 = 75%) and confuses
  `warp.thread_rank()` arithmetic.

**Check register spills** (SeGuRu emits PTX at
`target/release/deps/gpu/*.ptx` after `cargo build --release`):

```bash
grep -E "\.local|st\.local" target/release/deps/gpu/<crate>.ptx | wc -l
```

Any `.local` stores = register spill. Fix by:
1. Reducing tile size (e.g., 128×128 → 64×64 for GEMM),
2. Reducing per-thread register footprint (fewer accumulators),
3. Splitting the kernel.

Compare with `nvcc -ptx -arch=sm_80 ref.cu` to see baseline register
usage for the same algorithm. Target ≤ 32 registers/thread for 100%
occupancy at `blockDim=256`; ≤ 64 for 50%.

**Tail-block guard** — every elementwise kernel must have:

```rust
let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
if idx < N { ... }   // or use reshape_map! which encodes this
```

Skipping the guard on the tail block silently writes past the end of
`y[]`; SeGuRu's bounds checks catch it but cost a branch per thread.
`reshape_map!` lets the compiler prove the bound away.

**Tail effects (launch overhead)**: when `gridDim < 4 × num_SMs`, the tail
block dominates total time. For small problems (N < 100K elements),
launch overhead (~5–20 µs) may exceed kernel time — consider fusing
neighbouring kernels or batching.

## Shared-Memory Bank Conflicts

SeGuRu's `GpuShared::<[T; N]>` maps to CUDA's `__shared__` with the same
32-bank, 4-byte-wide layout. Bank index for element `smem[i]` with
`sizeof(T) = 4`: `i % 32`.

**When bank conflicts bite**:

- A 32×32 fp32 tile accessed both row-wise (consecutive threads → stride 1
  → bank `i % 32`, all different → 0 conflict) and column-wise (consecutive
  threads → stride 32 → bank `(i*32) % 32 = 0` for all threads → 32-way
  conflict, 32× slowdown for that access).
- GEMM kernels that store a *transposed* tile in smem for the inner loop.

**Fix**: pad the row by 1 element.

```rust
let mut smem_a = GpuShared::<[[f32; TILE_K + 1]; TILE_M]>::zero();
//                                  ^^^^^^^^^ pad
```

Effect: element `smem_a[row][col]` now maps to bank
`(row * (TILE_K + 1) + col) % 32`. When 32 threads access `[0..32][col]`
(column-wise), the rows shift by `(TILE_K+1) % 32 != 0`, breaking the
periodic conflict.

**When you do NOT need padding**:

- **K-major GEMM shared tiles in this recipe** — no padding needed for the
  retained 128x8 layout. The `+1` pad is for transposed column-wise tiles that
  create periodic bank conflicts.
- **fp16 smem** — two fp16 per bank means padding must be 2, not 1.
  Prefer storing `half2` pairs to avoid this.

**Verification**: `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`
should read ≤ 5% of smem transactions. If it doesn't, your pad is
wrong-dimension or wrong-size.

## Warp Divergence

SIMT: all 32 lanes of a warp execute the same instruction per cycle. When
lanes take different branches, the hardware executes each branch serially
with the inactive lanes masked off — cost = sum of both paths.

**Classify every branch in the inner loop**:

| Class | Example | Cost | Action |
|---|---|---|---|
| Warp-uniform | `if blockIdx.x == 0` | 0 | none |
| Geometry | `if lane_id < num_warps` | ≤ 1 cycle (mask-based) | none |
| Data-dependent | `if x[i] > 0 { ... } else { ... }` | sum of both paths | consider restructure |
| Boundary | `if idx < N` (tail guard) | ≤ 1 cycle on non-tail warps | keep; it's correctness |

**Rule of thumb**: don't spend time on divergence in a memory-bound
kernel — memory latency dominates. Softmax/reduction kernels gain < 5%
from divergence optimization. Compute-bound kernels (GEMM, elementwise
transcendentals) can see 20–40%.

**Boundary divergence trick — loop peeling**: split the loop into
"full-warp" iterations (no bounds check) and a tail (with bounds check):

```rust
let n_full = N - (N % 32);
let mut i = tid;
while i < n_full { /* no guard */ ...; i += block_dim::<DimX>(); }
while i < N { /* with guard */ ...; i += block_dim::<DimX>(); }
```

For stride-B loops this is usually not worth it — the tail is one warp.

## Debugging Checklist

When a SeGuRu kernel produces wrong output, work through this before
chasing performance or compiler bugs:

1. **Reproduce on the smallest failing input.** Try sizes `{31, 63, 64,
   65, 127, 128, 129}` — power-of-two hides boundary bugs.
2. **Classify the error pattern.**
   - All elements off by a constant factor → epilogue bug (wrong scale,
     missing epsilon, wrong dtype cast).
   - Last few rows/cols wrong → tail-block bounds guard missing.
   - Random scatter of wrong values → race condition (missing
     `sync_threads()` after smem write).
   - NaN → division by zero (empty reduction, fully-masked softmax) or
     `sqrt`/`log` of negative.
   - Inf → forgot max-subtraction in softmax.
3. **Sentinel-fill the output buffer** before launch — use `f32::NAN`,
   not `0.0`. Kernels that forget to write an element silently return 0
   otherwise, hiding the bug.
4. **Write the index formula on paper** for every global memory access.
   Most indexing bugs are mechanical stride/row/col confusion.
5. **Single-block isolation**: guard `if block_id::<DimX>() != 0 { return; }`
   at the kernel top. Removes all inter-block ordering effects. If output
   is still wrong, it's a single-block issue.
6. **Run under `compute-sanitizer`**:
   ```bash
   compute-sanitizer --tool memcheck target/release/examples/<bin>
   ```
   Catches OOB reads/writes, uninitialized smem reads, misaligned vector
   loads. Mandatory for non-deterministic bugs.
7. **Compare against a minimal-CUDA or CPU reference** element-by-element
   for the smallest failing input. Use `fp64` CPU if numerical-drift is
   suspected.
8. **After fix**: test at sizes `{31, 32, 33, 63, 64, 65, 127, 128, 129,
   255, 256, 257, 1023, 1024, 1025}` before declaring done. Partial-tile
   bugs love powers-of-two adjacency.

Common SeGuRu-specific traps:
- Forgetting `d_out.copy_to_host(&mut h_out)` before reading (silent
  all-zeros output — see host-side gotcha memory).
- Using `blockDim.x` in the Rust host launcher but thread-reading `D` in
  the kernel with a different stride.
- `reshape_map!` silently clipping output to `BLOCK * GRID` elements
  when `D % BLOCK != 0` — use `MapContinuousLinear::new(1)` + explicit
  index for non-divisible D.
