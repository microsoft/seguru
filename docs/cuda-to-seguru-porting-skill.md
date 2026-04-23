# CUDA to SeGuRu Porting Skill

## Overview

This skill documents battle-tested strategies for porting CUDA C++ GPU kernels to SeGuRu (Safe GPU Rust). These patterns were validated by porting all 21 PolybenchGPU benchmarks plus 3 classic kernels, and cross-referenced with the production llm-rs codebase (GPT-2 training/inference in SeGuRu).

## Performance Summary

With idiomatic patterns applied:
- **5/19 benchmarks at or below CUDA parity** (≤1.0×)
- **10/19 within 2× of CUDA**
- **Geometric mean overhead: ~1.6×** across all benchmarks

## Golden Rules

1. **Use `u32` for all GPU-side index variables and size parameters** — GPU has 32-bit ALUs; `usize` (64-bit) doubles arithmetic cost (30-45% overhead)
2. **Use subslice + iteration for row traversals** — `&a[row*n..(row+1)*n]` does ONE bounds check; per-element `a[i*n+k]` checks every access
3. **Use `MapContinuousLinear`** instead of `MapLinear` for 1D `chunk_mut` — uses `u32` index math internally
4. **Kernel function name must differ from crate name** — `histogram::histogram` causes MLIR mangling bug; use `histogram::histogram_kernel`
5. **Tests must use helper functions** — don't put `cuda_ctx` closure directly in `#[test]` fns (causes `{}` in MLIR symbol names)
6. **`chunk_mut` uses LOCAL indices** — write `c[0]`, not `c[global_idx]`
7. **Row reductions need block- or warp-per-row, not 1-thread-per-row** — reusing the elementwise template (`gs=B.div_ceil(bs)`, `if row<B`) for `sum(x,dim=-1)`-style kernels underutilizes the GPU by 10–20×. See "Row-Reduction Strategy".

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

**The #1 mistake LLMs make on reduction kernels** (`sum`, `mean`, `norm`,
`max`, `softmax`, …). The "subslice for row" pattern above is correct for
the *inner loop*, but it does NOT answer the question "how many threads
should cooperate on one row?".

Wrong choice here is a 10–20× performance cliff, not a 5% overhead.

### The pitfall: 1 thread per row

For `y[B] = sum(x[B, D], dim=-1)` with `B=128, D=16384`, an LLM that naively
reuses the elementwise template writes:

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
(measured on `sum_dim` in phase B.10). `B=128` threads can't saturate an
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

Used in `layer_norm.rs` phase B.10. Each block hosts `warps_per_block` warps,
and each warp owns one row. See `examples/kernelbench-b/src/layer_norm.rs`
for a working template using `warp.subgroup_id()` to index within the block.

### Red flags that you chose wrong

- `gs = B.div_ceil(256)` on a reduction where `B < ~10_000` → 1 thread/row, too few threads.
- No shared memory, no `warp.redux`, no tree reduction inside the kernel → you're not reducing cooperatively.
- Only `block_id` is used to identify the output row, no `thread_id`/`lane` cooperation inside → fine only if `D` is tiny.

If you wrote a reduction kernel that compiles and passes correctness but
looks suspiciously similar to your elementwise template, re-read this section.

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

For memory-bandwidth-bound kernels, use `Float4` for coalesced 128-bit loads (from llm-rs):
```rust
use gpu::Float4;

#[gpu::cuda_kernel]
pub fn kernel(out: &mut [Float4], inp: &[Float4], C: u32) {
    let mut out = chunk_mut(out, MapContinuousLinear::new(1));
    let idx = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    if idx < N {
        out[0] = inp[idx as usize].add(wpe[(t * C4 + c4) as usize]);
    }
}
```

Host-side: cast `TensorView<[f32]>` to `TensorView<[Float4]>`:
```rust
let inp = unsafe { &*(inp as *const _ as *const TensorView<'_, [Float4]>) };
```

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
    d_out.copy_to_host(&mut h_out).unwrap();  // ← REQUIRED, see Golden Rule #7
});
```

### Golden Rule #7 (host-side) — readback is NOT automatic

> `new_tensor_view(h_vec.as_mut_slice())` snapshots the host buffer to device
> at construction. After a kernel writes to that `TensorViewMut`, you **must**
> call `d_out.copy_to_host(&mut h_out).unwrap()` before reading or persisting
> `h_out`. Dropping the `TensorViewMut` does **not** trigger a readback.
>
> Symptom if omitted: `h_out` stays all zeros (or its initial value), and
> your downstream verification silently reports max-abs-err equal to the
> magnitude of the expected output. Observed in 2/3 LLM phase-B ports.

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
| Mechanical 1:1 port of memory-bound kernel | **1.3-1.5×** | See LayerNorm case study — port the algorithm, not syntax |

## Case Study: PyTorch LayerNorm (algorithm > idioms)

Porting PyTorch's `vectorized_layer_norm_kernel` revealed a critical nuance about
how SeGuRu reaches CUDA parity. Three SeGuRu variants were written from scratch
against a hand-tuned CUDA reference on M=8192, N=1024:

| Variant | Algorithm | SeGuRu idioms | Time | vs CUDA |
|---|---|---|---|---|
| CUDA reference | Fused Welford + float4 + warp shfl | (hand-written CUDA) | 62.7 µs | 1.00× |
| SeGuRu vectorized | Fused stats + Float4 | `reshape_map`, `ThreadWarpTile::redux`, subslice | **65.9 µs** | **1.05×** |
| SeGuRu naive | 3-pass (mean, var, out), scalar, block-per-row | tree reduction only | 82.3 µs | 1.31× |
| SeGuRu "idiomatic" | 3-pass scalar, warp-per-row | `reshape_map`, `redux`, `ldcs` | 89.7 µs | 1.43× |

### The surprise: idioms ≠ performance

The "idiomatic" variant — using every SeGuRu performance pattern (warp-cooperative
reduction, `ldcs`, `reshape_map` strided output, subslice rows) — was **slower than
the naive 1:1 port**. Both run the same 3-pass algorithm; the idiomatic version
loses because one warp per row gives only 1024 blocks of parallelism vs the naive's
8192 blocks.

### The rule

**Port the algorithm first, then the idioms.** A clean SeGuRu rendering of a bad
algorithm will not reach parity. What unlocks parity is:

1. **Fuse passes** — compute mean and variance in a single traversal (local `s` +
   `sq` accumulators, two `warp.redux` calls). Halves global memory traffic for
   memory-bound kernels.
2. **Vectorize loads** — use `Float4` for contiguous reads/writes. Host side builds
   `Vec<Float4>`; kernel declares `x: &[Float4]`; index as `x[i][k]` for lanes.
3. **Then apply SeGuRu idioms** — `reshape_map!` for per-thread output slots,
   `ThreadWarpTile::redux` for cross-lane reductions, subslice for row access.

Only when the algorithm is right do SeGuRu's safety abstractions become free.

### Implication for automated porting

A mechanical CUDA→SeGuRu translator will land at 1.3–2.0× overhead even with
perfect idiom usage, because the algorithmic opportunities (pass fusion, vector
width selection, Welford online updates) require **semantic** rewrites, not
syntactic translation. Automation should:

- Detect multi-pass statistics (mean+var, max+sum) and fuse them.
- Detect contiguous float loads of stride 4/8 and lift them to `Float4`/`Float8`.
- Detect warp-reducible patterns and emit `ThreadWarpTile::redux`.
- Leave block/grid geometry choices to a tuning pass — warp-per-row vs block-per-row
  is workload-dependent.

### Fused-stats warp kernel template

```rust
#[gpu::cuda_kernel]
pub fn layernorm_vectorized(
    x: &[Float4], gamma: &[Float4], beta: &[Float4], y: &mut [Float4],
) {
    let warp = ThreadWarpTile::<32>;
    let warps_per_block = warp.meta_group_size();
    let row = block_id::<DimX>() * warps_per_block + warp.subgroup_id();
    let lane = warp.thread_rank();

    const N4: u32 = N / 4;
    let x_row = &x[(row * N4) as usize..((row + 1) * N4) as usize];

    // ONE pass: accumulate sum and sumsq together.
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    let mut i = lane;
    while i < N4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 { let vk = v[k]; s += vk; sq += vk * vk; }
        i += warp.size();
    }
    let sum = warp.redux(ReduxAdd, s);
    let sumsq = warp.redux(ReduxAdd, sq);
    let inv_n = 1.0 / (N as f32);
    let mean = sum * inv_n;
    let rstd = (sumsq * inv_n - mean * mean + 1e-5).rsqrt();

    // Strided Float4 output via reshape_map.
    let mut y_chunk = chunk_mut(y, reshape_map!(
        [N4 / 32] | [32, warps_per_block * grid_dim::<DimX>()] => layout: [t0, i0, t1]
    ));
    let mut slot = 0u32;
    let mut i = lane;
    while i < N4 {
        let v = x_row[i as usize];
        let g = gamma[i as usize];
        let b = beta[i as usize];
        let mut out = Float4::new([0.0; 4]);
        for k in 0..4 { out[k] = (v[k] - mean) * rstd * g[k] + b[k]; }
        y_chunk[slot] = out;
        i += warp.size();
        slot += 1;
    }
}
```

Source: `examples/bench-layernorm/` (all three variants + host harness); CUDA
reference at `benchmarks/cuda/layernorm_pytorch.cu`.

## Case Study: KernelBench L1 skill-doc stress test

Five KernelBench Level-1 problems were ported to SeGuRu using **only patterns
documented above**. All five compiled on the first attempt and produced correct
output. Performance vs PyTorch (A100):

| Problem | Shape | SeGuRu | PyTorch | Ratio |
|---|---|---|---|---|
| 19_ReLU | 4096×16384 | 350 µs | 325 µs | **1.08×** |
| 21_Sigmoid | 4096×16384 | 353 µs | 325 µs | **1.09×** |
| 23_Softmax (dim=1) | 4096×4096 | 208 µs | 98 µs | 2.12× |
| 40_LayerNorm (vectorized variant) | 8192×1024 | 66 µs | — | 1.05× vs CUDA ref |
| 1_SquareMatmul | 4096×4096 | 31510 µs | 7308 µs | 4.31× |

### What worked

- **Elementwise kernels reach PyTorch parity** with the documented
  `chunk_mut(MapContinuousLinear::new(1))` + bounds-guarded global-thread-id
  pattern. No shared memory, no vectorization, no idiom tricks — just the
  basic elementwise template.
- **All correctness checks passed** on the first build — sync analysis,
  bounds guards, and chunk-based writes caught nothing the author had missed,
  meaning the documented rules suffice to produce compilable code for these
  patterns.

### Gaps surfaced

**Gap 1 — fused multi-pass reductions** (softmax at 2.12×, layernorm-idiomatic
at 1.43×): the doc now has a "port the algorithm first" note in the LayerNorm
case study, but the general lesson is worth stating once as a Golden Rule:
*any reduction pattern (max+sum, mean+var, logsumexp) should be fused into a
single pass over global memory*. The 3-pass "natural translation" landing
around 2× is predictable.

**Gap 2 — register tiling for GEMM** (matmul at 4.31×). The skill doc's
GEMM-family guidance stops at the 16×16 shared-memory tile pattern. Reaching
competitive GEMM performance requires:
- Larger block tiles (e.g., 128×128) with each thread computing a **register
  tile** (e.g., 8×8 outputs held in registers across the K-loop).
- Double-buffered shared-memory loads (software pipelining).
- For f32, `tf32` via WMMA/tensor cores to approach cuBLAS.

None of this is in the doc. The existing `bench_gemm_tiled` in
`examples/bench/src/main.rs` has a comment: *"Register tiling (each thread
computes NxM outputs) is the next optimization axis"* — that axis is
undocumented.

**Practical rule for now:** For large dense f32 matmul, the doc already points
at cuBLAS via `cublasSgemm_v2`. Use it. Treat user-written SeGuRu matmul as
appropriate only for small sizes or non-standard shapes where cuBLAS doesn't
fit. The 4.3× gap is structural, not a skill-doc failure — but the doc should
say so explicitly instead of implying a 1.7-2.0× tile-only ratio is tight.

### Revised Golden Rule

> **Port the algorithm before the idioms.**
> Fuse multi-pass reductions (max+sum, mean+var, logsumexp) into one pass.
> Choose a vector width (Float4/Float8) before writing the loop.
> Choose a tile size with register tiling *before* writing the GEMM kernel.
> Then apply SeGuRu's chunk_mut/reshape_map/warp-redux patterns.

### Reproducing

- SeGuRu: `cargo run --release -p kernelbench`
- PyTorch: `python3 examples/kernelbench/python/run_torch_baseline.py`

Both crates at `examples/kernelbench/`.

## Case Study: KernelBench L1 (phase B — LLM-driven generation)

Three fresh LLM sub-agents (Claude Sonnet, one-shot, access only to this skill
doc + the phase-A examples) were asked to port:

- **LeakyReLU** — elementwise with scalar param
- **Tanh** — elementwise with intrinsic
- **RMSNorm** — strided reduction over 4D `(B, C, H, W)` tensor along dim=1

Results (after a 1-line host-side fix noted below):

| problem | torch | LLM-generated SeGuRu | speedup |
|---------|-------|----------------------|---------|
| LeakyReLU (4096×393216) | 7.68 ms | 8.35 ms | 0.92× |
| Tanh (4096×393216)      | 7.67 ms | 8.54 ms | 0.90× |
| RMSNorm (112×64×512×512)| 24.2 ms | 16.5 ms | **1.47×** |

All 3 compiled first try. All 3 kernels produced correct output. The LLM
actually *beat* PyTorch on RMSNorm by picking a sensible two-kernel
decomposition (one reduction kernel writing an `inv_rms` auxiliary buffer,
one elementwise apply kernel) — faster than PyTorch's fused-but-temporary path.

### The one observed LLM failure mode: host-side `copy_to_host`

**2 of 3 sub-agents produced a silent correctness bug**: after running the
kernel, they dropped the `TensorViewMut` *without* calling
`d_out.copy_to_host(&mut h_out)`. This yielded all-zeros output without any
warning. The skill doc does show the right pattern in "Host-Side Patterns /
Basic launch" but it's a one-line comment-less call in a code block; LLMs
focused on kernel design missed it.

**Golden Rule #7 (host-side):**

> After a kernel writes to a `TensorViewMut<[T]>` backed by a host vector,
> you **must** call `d_out.copy_to_host(&mut h_out).unwrap()` before reading
> or persisting `h_out`. `new_tensor_view` snapshots the host data to device
> at construction; there is no automatic readback on drop. This is the single
> most frequent mistake observed in LLM-generated ports — both Tanh and
> LeakyReLU agents hit it while writing otherwise-correct kernels.

### Takeaway for phase B

Even with a thorough skill doc, the *host plumbing* is what trips up LLMs,
not the GPU code itself. Kernel authorship is well-covered; the missing
piece is a "host recipe" section with explicit `read → device → launch →
sync → copy_to_host → write` scaffolding that the LLM can copy verbatim.

### Phase B.full results — safety vs. raw CUDA (same LLM)

Same model (Claude Sonnet sub-agent, one-shot) was asked to port the
same three problems to (a) SeGuRu using this skill doc and (b) raw CUDA
with `torch::Tensor` + `PYBIND11_MODULE`. Both arms ran against the
same PyTorch reference on identical input tensors. All six generated
kernels use float4 vectorization + grid-stride + (SeGuRu: `reshape_map!` /
`chunk_map`; CUDA: explicit `float4*` reinterpret).

| Problem   | PyTorch    | SeGuRu         | Raw CUDA       | overhead |
|-----------|-----------:|---------------:|---------------:|---------:|
| leaky_relu| 7.68 ms    | 8.35 ms (0.92×)| 8.05 ms (0.95×)|    −3.7% |
| tanh      | 7.67 ms    | 8.55 ms (0.90×)| 7.97 ms (0.96×)|    −7.3% |
| rms_norm  | 24.25 ms   |16.50 ms (1.47×)|13.27 ms (1.83×)|    −24%  |

Correctness: **3/3 for both arms** (max-abs-err ≤ 8e-6). On memory-bound
elementwise ops the safety layer is invisible; on reductions SeGuRu
currently pays ~20% vs. hand-managed shared memory in raw CUDA, but
*both* arms still beat PyTorch. The conclusion for KernelBench-style
LLM codegen: SeGuRu is a viable safe target whose `fast_N` scores
should track raw CUDA closely on memory-bound L1, with a measurable
but not disqualifying gap on reduction kernels.

### Phase B.10 results — scaled to 10 KernelBench L1 problems

Same setup, expanded to ten problems spanning elementwise / reduction /
softmax / norm categories. Same Claude Sonnet sub-agent ports each problem
to both SeGuRu (with this skill doc) and raw CUDA (with the symmetric
`docs/cuda-raw-kernel-skill.md`). Driver: `examples/kernelbench-b/python/compare2.py`.

| Problem    | PyTorch eager | SeGuRu          | Raw CUDA        |
|------------|--------------:|----------------:|----------------:|
| leaky_relu |       640.9µs |  698.1µs (0.92×)|  668.3µs (0.96×)|
| tanh       |       654.0µs |  707.2µs (0.92×)|  663.7µs (0.99×)|
| relu       |       653.7µs |  696.9µs (0.94×)|  669.1µs (0.98×)|
| sigmoid    |       655.5µs |  703.2µs (0.93×)|  664.5µs (0.99×)|
| gelu       |       663.6µs |  728.9µs (0.91×)|  665.7µs (1.00×)|
| softmax    |       229.5µs |  324.8µs (0.71×)|  244.6µs (0.94×)|
| layer_norm |       261.5µs |  338.8µs (0.77×)|  215.3µs (1.21×)|
| rms_norm   |     24269.7µs |16574.7µs (1.46×)|13268.3µs (1.83×)|
| sum_dim    |       177.5µs |  225.2µs (0.79×)|  152.9µs (1.16×)|
| l2_norm    |       452.9µs |  299.5µs (1.51×)|  208.4µs (2.17×)|

Aggregate (`fast_N` = pct of problems with speedup ≥ N× vs PyTorch eager):

| Arm    | correct | fast_1 | fast_2 | avg speedup |
|--------|--------:|-------:|-------:|------------:|
| SeGuRu |   10/10 |    20% |     0% |       0.99× |
| CUDA   |   10/10 |    40% |    10% |       1.22× |

`sum_dim` and `l2_norm` above reflect the re-port using the
"Row-Reduction Strategy" section of this skill doc (the first-pass LLM
sub-agent produced `sum_dim=2981.8µs` / `l2_norm=1675.3µs` — 13× and
5.6× slower respectively — before the skill doc explicitly called out
the 1-thread-per-row pitfall).

i.e. with the updated skill doc an LLM sub-agent matches PyTorch eager
on average across 10 L1 problems (geomean ≈ 1.0× eager) while staying
10/10 correct. The remaining raw-CUDA edge comes from two reductions
(softmax, layer_norm) where SeGuRu still pays ~20-50% for its safety
layer on cross-warp reduction scaffolding — addressable with further
skill-doc tuning, not a fundamental ceiling.
- **Correctness parity holds at scale.** Both arms 10/10 on a one-shot
  prompt — the skill docs are sufficient context for an LLM to produce
  numerically-correct kernels across all four op categories tested.
- **Memory-bound elementwise (6 problems)**: SeGuRu sits at 0.91–0.94×
  PyTorch, raw CUDA at 0.96–1.00×. The safety layer's ~5% overhead is
  consistent and small enough to not change `fast_N` rankings.
- **Reductions and softmax**: SeGuRu lags more (0.71–0.77× on softmax /
  layer_norm). On `sum_dim` and `l2_norm` the LLM picked a 1-thread/row
  pattern in SeGuRu vs a 1-block/row warp-shuffle pattern in CUDA — that
  is an LLM strategy choice, not a SeGuRu ceiling. With a follow-up port
  using SeGuRu's shared-memory reduce primitives the gap should close.
- **rms_norm**: both arms beat PyTorch significantly (1.46× / 1.83×) —
  PyTorch eager pays a large general-purpose dispatch tax on small norms.
- **`torch.compile` baseline disabled** in this run (TypeError on lambda
  wrap under our torch version). Re-enable in a follow-up by wrapping
  problems in `nn.Module` instead of `lambda`.

### CUDA gotcha learned: `__shfl_down_sync` partial masks deadlock

While porting `l2_norm` the LLM wrote a cross-warp reduce of the form
`if (tid < BLOCK/32) { acc += __shfl_down_sync(0xffffffff, acc, o); }`.
On A100 (sm_80) this hangs the kernel forever (100% GPU util, never
returns). The mask `0xffffffff` declares all 32 lanes participating, but
only `BLOCK/32` lanes entered the branch — undefined behavior.

**Fix pattern**: have all 32 lanes of warp 0 enter the branch, with
inactive lanes contributing 0:

```cpp
if (threadIdx.x < 32) {
    float acc = (threadIdx.x < BLOCK/32) ? warp_sum[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (threadIdx.x == 0) /* write */;
}
```

Now documented in `docs/cuda-raw-kernel-skill.md` so future raw-CUDA
ports avoid this pitfall.

