# CUDA→SeGuRu Automated Porting: Proof-of-Concept Design

## Problem Statement

SeGuRu provides a systematic 1:1 mapping from CUDA C++ to Rust for GPU kernel programming. We want to validate whether this mapping is regular enough to automate the porting of standalone CUDA kernels to SeGuRu, and what mix of mechanical rule-based transforms vs. LLM-assisted translation is needed.

## Approach

**LLM-First with Validation:** Port 3 classic CUDA kernels to SeGuRu by hand (using the known mapping rules), validate they compile and produce correct results, then assess what percentage of the translation was mechanical vs. required semantic understanding.

## Proof-of-Concept Kernels

### 1. Vector Add

**CUDA patterns exercised:** kernel definition, thread indexing, global memory read/write, grid-stride loop, launch configuration.

**Why:** The simplest possible CUDA kernel. If this doesn't translate cleanly, nothing will. Establishes the baseline for kernel signature mapping, type translation, and host-side boilerplate.

### 2. Parallel Reduction (Sum)

**CUDA patterns exercised:** shared memory (`__shared__`), barrier synchronization (`__syncthreads()`), warp-level shuffle operations, multi-stage tree reduction, conditional thread participation.

**Why:** Tests the most common complex CUDA pattern — cooperative computation within a thread block using shared memory. This is where SeGuRu's `GpuShared`, `sync_threads()`, and `shuffle!` macro get exercised.

### 3. Histogram

**CUDA patterns exercised:** atomic operations (`atomicAdd`), shared memory atomics, boundary checking, two-phase algorithms (local histogram → global merge).

**Why:** Tests atomic operations which have a different API surface in SeGuRu (`Atomic::new(GpuGlobal::new(data))` wrapper pattern). Also tests combining shared memory with atomics.

## Translation Pipeline

For each kernel, translation follows these steps:

### Step 1: Kernel Signature Mapping

```
CUDA:    __global__ void kernel(float *data, int n)
SeGuRu:  #[gpu::cuda_kernel]
         pub fn kernel(data: &mut [f32], n: usize)
```

**Rules:**
- `__global__` → `#[gpu::cuda_kernel]`
- `float*` (read-only) → `&[f32]`
- `float*` (read-write) → `&mut [f32]`
- `int`/`int64_t` → `i32`/`i64` or `usize` depending on usage
- Pointer + size pairs may collapse to a single slice

### Step 2: Thread Intrinsic Mapping

| CUDA C++ | SeGuRu Rust |
|---|---|
| `threadIdx.x/y/z` | `thread_id::<DimX/Y/Z>()` |
| `blockIdx.x/y/z` | `block_id::<DimX/Y/Z>()` |
| `blockDim.x/y/z` | `block_dim::<DimX/Y/Z>()` |
| `gridDim.x/y/z` | `grid_dim::<DimX/Y/Z>()` |

### Step 3: Memory Pattern Mapping

| CUDA C++ | SeGuRu Rust |
|---|---|
| `__shared__ float smem[N]` | `let mut smem = GpuShared::<[f32; N]>::zero()` |
| `extern __shared__ float smem[]` | `let smem = smem_alloc.alloc::<f32>(n)` (requires `#[gpu::cuda_kernel(dynamic_shared)]`) |

### Step 4: Synchronization Mapping

| CUDA C++ | SeGuRu Rust |
|---|---|
| `__syncthreads()` | `sync_threads()` |
| `__shfl_xor_sync(mask, val, offset, width)` | `shuffle!(xor, val, offset, width)` |
| `__shfl_down_sync(mask, val, offset, width)` | `shuffle!(down, val, offset, width)` |

### Step 5: Atomic Mapping

| CUDA C++ | SeGuRu Rust |
|---|---|
| `atomicAdd(&data[i], val)` (int) | `Atomic::new(GpuGlobal::new(data)).atomic_addi(val)` |
| `atomicAdd(&data[i], val)` (float) | `Atomic::new(GpuGlobal::new(data)).atomic_addf(val)` |
| `atomicAdd(&smem[i], val)` | `SharedAtomic::new(&mut smem).atomic_addi(val)` |
| `atomicMax`, `atomicMin` | `.atomic_maxs()/.atomic_mins()` (signed), `.atomic_maxu()/.atomic_minu()` (unsigned) |

### Step 6: Math Function Mapping

CUDA math functions → Rust trait method calls via `GPUDeviceFloatIntrinsics`:
- `sinf(x)` → `x.sin()`
- `__expf(x)` → `x.exp()`
- `fmaf(a, b, c)` → `(a, b).fma(c)`
- etc.

### Step 7: Host Code Generation

```
CUDA:                                    SeGuRu:
cudaMalloc(&d_data, size)        →       let d_data = ctx.new_tensor_view(&h_data)
cudaMemcpy(d, h, size, H2D)     →       (implicit in new_tensor_view)
kernel<<<grid, block>>>(args)    →       kernel::launch(config, ctx, m, args)
cudaMemcpy(h, d, size, D2H)     →       d_data.copy_to_host(&mut h_data)
cudaFree(d_data)                 →       (automatic via Drop)
cudaDeviceSynchronize()          →       ctx.sync()
```

## File Structure

Following the existing workspace pattern (see `reduce/`, `hello/` examples), each ported kernel is a single workspace crate with both GPU kernel and host code:

```
examples/
├── Cargo.toml              # Add new workspace members
├── vector_add/
│   ├── Cargo.toml           # depends on gpu, gpu_host
│   └── src/
│       └── lib.rs           # Kernel + host launch + #[test]
├── histogram/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
cuda-examples/
│   ├── vector_add.cu        # Original CUDA C++ source
│   ├── reduce.cu            # Original CUDA C++ source (if not already present)
│   └── histogram.cu         # Original CUDA C++ source
```

Note: `reduce/` already exists as an example. We'll verify it covers the parallel reduction patterns we need, and add/extend if necessary.

Each SeGuRu example crate contains:
- Kernel function(s) annotated with `#[gpu::cuda_kernel]`
- Host launch code using `gpu_host::cuda_ctx`
- `#[test]` functions that validate correctness

## Automation Assessment Criteria

After completing the 3 ports, we evaluate:

### Mechanical (Rule-Based) Transforms
- Kernel signature (`__global__` → `#[gpu::cuda_kernel]`)
- Type mapping (`float*` → `&[f32]`)
- Thread intrinsics (`threadIdx.x` → `thread_id::<DimX>()`)
- Synchronization (`__syncthreads()` → `sync_threads()`)
- Memory allocation boilerplate
- Launch configuration
- Math function mapping

### Semantic (LLM-Required) Transforms
- Pointer arithmetic → Rust slice indexing (understanding which pointer is read-only vs mutable)
- Loop restructuring for Rust idioms
- Choosing between `MapLinear`, `Map2D`, `chunk_mut` vs raw indexing
- Handling CUDA's implicit thread-safety assumptions
- Bounds checking strategy (Rust panics vs. CUDA silent OOB)

### Gap Assessment
- Features used in CUDA that have no SeGuRu equivalent
- Patterns that require significant restructuring
- Performance implications of the translation

## Success Criteria

1. All 3 kernels compile with SeGuRu's `rustc-gpu` toolchain
2. Numerical output matches expected results (within f32 epsilon for floating-point)
3. Each kernel demonstrates a distinct CUDA pattern category
4. Assessment document answers: "What % is mechanical? What needs LLM? What are the gaps?"

## Out of Scope

- Building transpiler tooling (this is a PoC, not a tool)
- Templates, CUDA libraries (cuBLAS, cuDNN), multi-file projects
- Performance optimization or benchmarking
- Complex CUDA features: texture memory, unified memory, CUDA graphs, multi-GPU
