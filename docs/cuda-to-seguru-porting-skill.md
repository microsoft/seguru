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
    d_out.copy_to_host(&mut h_out).unwrap();
});
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
| GEMM-family with subslice | **1.7-2.0×** | Column-stride `b[]` still per-element |
| Matrix-vector | **1.7-1.8×** | Mix of row (fast) and column (slow) access |
| Column-heavy reductions (corr, covar) | **~2.3×** | Column-stride access; `.ldcs()` helps significantly |
| Iterative with many launches | **1.9×** | Launch overhead (4.7 vs 2.0 µs) accumulates |
