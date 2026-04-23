# CUDA to SeGuRu Porting Skill

## Overview

This skill documents battle-tested strategies for porting CUDA C++ GPU kernels to SeGuRu (Safe GPU Rust). These patterns were validated by porting all 21 PolybenchGPU benchmarks plus 3 classic kernels, achieving near-CUDA performance on an A100 GPU.

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
- `int`/`unsigned int` → `u32` (not `usize`)
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

### When to use subslice
- **Row traversals** (contiguous memory): Always use subslice
- **Column/stride traversals** (non-contiguous): Cannot subslice — keep per-element `while` loop
- **Stencil access** (neighbor elements): Fixed number of accesses — subslice doesn't help

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

## Shared Memory

### Static shared memory
```rust
let mut smem = GpuShared::<[f32; 256]>::zero();
let mut chunk = smem.chunk_mut(MapLinear::new(1));
chunk[0] = value;  // write
sync_threads();
let val = *smem[i];  // read (dereference)
```

### Dynamic shared memory
```rust
#[gpu::cuda_kernel(dynamic_shared)]
pub fn kernel(input: &[f32], output: &mut [f32], n: u32) {
    let smem = smem_alloc.alloc::<f32>(block_dim::<DimX>() as usize);
    let mut chunk = smem.chunk_mut(MapContinuousLinear::new(1));
    chunk[0] = value;
    sync_threads();
    let val = *smem[i as usize];
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

## Host Side Translation

| CUDA | SeGuRu |
|------|--------|
| `cudaMalloc + cudaMemcpy H→D` | `ctx.new_tensor_view(&host_data)` |
| `cudaMemcpy D→H` | `d_data.copy_to_host(&mut host_data)` |
| `cudaFree` | automatic (Drop) |
| `cudaDeviceSynchronize` | `ctx.sync()` |
| `kernel<<<grid,block,smem>>>(args)` | `kernel::launch(gpu_config!(...), ctx, m, args)` |

### Launch configuration
```rust
// gpu_config!(grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes)
let config = gpu_host::gpu_config!(num_blocks, 1, 1, 256, 1, 1, 0);
kernel::launch(config, ctx, m, &d_input, &mut d_output, n as u32).unwrap();
```

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

## Performance Expectations (A100, naive kernels)

| Kernel Category | Expected Ratio vs CUDA | Dominant Factor |
|----------------|----------------------|-----------------|
| Large compute, row access (syrk, syr2k) | **1.00×** | Compute-bound, subslice amortizes checks |
| Stencil (fdtd2d, jacobi2d) | **1.0-1.3×** | Memory-bound, few accesses per thread |
| GEMM-family with subslice | **1.7-2.0×** | Column-stride `b[]` still per-element |
| Matrix-vector | **1.7-1.8×** | Mix of row (fast) and column (slow) access |
| Column-heavy reductions (corr, covar) | **~3×** | All column access, no subslice possible |
| Iterative with many launches | **1.9×** | Launch overhead (4.7 vs 2.0 µs) accumulates |
