# CUDA→SeGuRu Porting PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 3 classic CUDA kernels (vector_add, reduce, histogram) to SeGuRu Rust, validate correctness, and assess automation feasibility.

**Architecture:** Each kernel is a workspace crate under `examples/` following the existing `reduce/` pattern — a single `lib.rs` containing the `#[gpu::cuda_kernel]` function and `#[cfg(test)]` module with host launch + validation. Original CUDA sources are stored in `cuda-examples/` for reference.

**Tech Stack:** Rust (nightly via rust-toolchain.toml), SeGuRu `gpu` + `gpu_host` crates, CUDA runtime.

---

### Task 1: Add CUDA C++ Reference Sources

**Files:**
- Create: `cuda-examples/vector_add.cu`
- Create: `cuda-examples/reduce.cu`
- Create: `cuda-examples/histogram.cu`

These are the original CUDA C++ kernels we're porting. They serve as reference and as part of the porting assessment.

- [ ] **Step 1: Create vector_add.cu**

```cuda
// Classic CUDA vector addition kernel
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, 1, 1, blockSize, 1, 1>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            printf("Error at index %d: %f != 3.0\n", i, h_c[i]);
            return 1;
        }
    }
    printf("PASSED\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

- [ ] **Step 2: Create reduce.cu**

```cuda
// Classic CUDA parallel reduction (sum) kernel
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread into shared memory
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    smem[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

int main() {
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(NUM_BLOCKS * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, NUM_BLOCKS * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) total += h_output[i];
    printf("Sum = %f (expected %f)\n", total, (float)N);

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    return 0;
}
```

- [ ] **Step 3: Create histogram.cu**

```cuda
// Classic CUDA histogram kernel with shared memory + atomics
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void histogram(const unsigned int *data, unsigned int *bins, int n) {
    __shared__ unsigned int smem_bins[NUM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory bins to zero
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        smem_bins[i] = 0;
    }
    __syncthreads();

    // Accumulate into shared memory bins
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&smem_bins[data[i]], 1);
    }
    __syncthreads();

    // Merge shared memory bins into global bins
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], smem_bins[i]);
    }
}

int main() {
    const int N = 1 << 20;
    unsigned int *h_data = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int *h_bins = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));

    // Fill with values 0..255
    for (int i = 0; i < N; i++) h_data[i] = i % NUM_BINS;

    unsigned int *d_data, *d_bins;
    cudaMalloc(&d_data, N * sizeof(unsigned int));
    cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int));

    int blockSize = 256;
    int numBlocks = min((N + blockSize - 1) / blockSize, 1024);
    histogram<<<numBlocks, blockSize>>>(d_data, d_bins, N);

    cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Verify: each bin should have N / NUM_BINS counts
    int expected = N / NUM_BINS;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_bins[i] != expected) {
            printf("Error: bin[%d] = %u, expected %d\n", i, h_bins[i], expected);
            return 1;
        }
    }
    printf("PASSED\n");

    cudaFree(d_data); cudaFree(d_bins);
    free(h_data); free(h_bins);
    return 0;
}
```

- [ ] **Step 4: Commit CUDA reference sources**

```bash
cd /home/sanghle/work/seguru
git add cuda-examples/vector_add.cu cuda-examples/reduce.cu cuda-examples/histogram.cu
git commit -m "examples: add CUDA C++ reference sources for porting PoC

Add classic vector_add, reduce (parallel sum), and histogram CUDA
kernels as reference for the CUDA→SeGuRu porting proof-of-concept.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Port Vector Add Kernel to SeGuRu

**Files:**
- Create: `examples/vector_add/Cargo.toml`
- Create: `examples/vector_add/src/lib.rs`
- Modify: `examples/Cargo.toml` (add workspace member)

- [ ] **Step 1: Create the crate Cargo.toml**

Create `examples/vector_add/Cargo.toml`:

```toml
[package]
name = "vector_add"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Add workspace member**

In `examples/Cargo.toml`, add `"vector_add"` to the `members` list:

```toml
[workspace]
members = [
    "hello",
    "llm-rs-gpu",
    "matrix/matmul-gpu",
    "matrix/matmul-host",
    "reduce",
    "syntax/gpu",
    "syntax/host",
    "vector_add",
]
```

- [ ] **Step 3: Write the kernel and test**

Create `examples/vector_add/src/lib.rs`:

```rust
use gpu::prelude::*;

/// Vector addition: c[i] = a[i] + b[i], using a grid-stride loop.
///
/// CUDA equivalent:
///   __global__ void vector_add(const float *a, const float *b, float *c, int n) {
///       int idx = blockIdx.x * blockDim.x + threadIdx.x;
///       int stride = blockDim.x * gridDim.x;
///       for (int i = idx; i < n; i += stride) {
///           c[i] = a[i] + b[i];
///       }
///   }
#[gpu::cuda_kernel]
pub fn vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let idx = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let stride = (block_dim::<DimX>() * grid_dim::<DimX>()) as usize;
    let mut i = idx;
    while i < n {
        c[i] = a[i] + b[i];
        i += stride;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    #[test]
    fn test_vector_add_basic() {
        let n: usize = 1024;
        let h_a: Vec<f32> = vec![1.0; n];
        let h_b: Vec<f32> = vec![2.0; n];
        let mut h_c: Vec<f32> = vec![0.0; n];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a failed");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b failed");
            let mut d_c = ctx
                .new_tensor_view(h_c.as_mut_slice())
                .expect("alloc c failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, 0);
            vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n)
                .expect("vector_add kernel launch failed");

            d_c.copy_to_host(&mut h_c).expect("copy from device failed");
        });

        for (i, val) in h_c.iter().enumerate() {
            assert!(
                (*val - 3.0).abs() < 1e-6,
                "Mismatch at index {}: {} != 3.0",
                i,
                val
            );
        }
    }

    #[test]
    fn test_vector_add_large() {
        let n: usize = 1 << 16; // 65536
        let h_a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let h_b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut h_c: Vec<f32> = vec![0.0; n];

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a failed");
            let d_b = ctx.new_tensor_view(h_b.as_slice()).expect("alloc b failed");
            let mut d_c = ctx
                .new_tensor_view(h_c.as_mut_slice())
                .expect("alloc c failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, 0);
            vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n)
                .expect("vector_add kernel launch failed");

            d_c.copy_to_host(&mut h_c).expect("copy from device failed");
        });

        for (i, val) in h_c.iter().enumerate() {
            let expected = n as f32;
            assert!(
                (*val - expected).abs() < 1e-1,
                "Mismatch at index {}: {} != {}",
                i,
                val,
                expected
            );
        }
    }
}
```

- [ ] **Step 4: Build to verify compilation**

```bash
cd /home/sanghle/work/seguru/examples
cargo build -p vector_add
```

Expected: Build succeeds.

- [ ] **Step 5: Run tests**

```bash
cd /home/sanghle/work/seguru/examples
cargo test -p vector_add
```

Expected: Both `test_vector_add_basic` and `test_vector_add_large` pass.

- [ ] **Step 6: Commit**

```bash
cd /home/sanghle/work/seguru
git add examples/vector_add/ examples/Cargo.toml
git commit -m "examples: add vector_add SeGuRu port from CUDA

Port the classic CUDA vector addition kernel to SeGuRu. Uses
grid-stride loop, global memory read/write, basic thread indexing.
Includes tests with 1K and 64K element arrays.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Port Parallel Reduction Kernel to SeGuRu

The existing `reduce/` example uses SeGuRu-idiomatic chunking (`reshape_map!`, `chunk_mut`). For the porting PoC, we write a **direct CUDA-style translation** that mirrors the CUDA `reduce.cu` code more closely (shared memory indexing, explicit tree reduction), to validate the mechanical translation approach. We'll add this as a separate test in the existing `reduce` crate.

**Files:**
- Modify: `examples/reduce/src/lib.rs` (add `reduce_sum` kernel + tests)

- [ ] **Step 1: Add the CUDA-style reduce_sum kernel**

Append to `examples/reduce/src/lib.rs`, before the `#[cfg(test)]` block:

```rust
/// Direct CUDA-style parallel reduction (sum) — mirrors reduce.cu closely.
///
/// CUDA equivalent:
///   __global__ void reduce_sum(const float *input, float *output, int n) {
///       extern __shared__ float smem[];
///       int tid = threadIdx.x;
///       int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
///       float sum = 0.0f;
///       if (idx < n) sum += input[idx];
///       if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
///       smem[tid] = sum;
///       __syncthreads();
///       for (int s = blockDim.x / 2; s > 0; s >>= 1) {
///           if (tid < s) smem[tid] += smem[tid + s];
///           __syncthreads();
///       }
///       if (tid == 0) output[blockIdx.x] = smem[0];
///   }
#[gpu::cuda_kernel(dynamic_shared)]
pub fn reduce_sum(input: &[f32], output: &mut [f32], n: usize) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = (block_id::<DimX>() * bdim * 2 + tid) as usize;

    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));

    // Load two elements per thread
    let mut sum = 0.0f32;
    if idx < n {
        sum += input[idx];
    }
    if idx + (bdim as usize) < n {
        sum += input[idx + bdim as usize];
    }
    smem_chunk[0] = sum;
    sync_threads();

    // Tree reduction in shared memory
    for order in (0..16).rev() {
        let stride = 1u32 << order;
        if stride >= bdim {
            continue;
        }
        let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
        if tid < stride {
            let right = smem_chunk[1];
            let left = smem_chunk[0];
            smem_chunk[0] = left + right;
        }
        sync_threads();
    }

    // Thread 0 writes block result
    if tid == 0 {
        output[block_id::<DimX>() as usize] = *smem[0];
    }
}
```

- [ ] **Step 2: Add tests for reduce_sum**

Append inside the existing `#[cfg(test)] mod tests` block in `examples/reduce/src/lib.rs`:

```rust
    fn test_reduce_sum_impl(n: usize, block_size: u32) {
        let num_blocks = ((n as u32) + block_size * 2 - 1) / (block_size * 2);
        let h_input: Vec<f32> = vec![1.0; n];
        let mut h_output: Vec<f32> = vec![0.0; num_blocks as usize];

        cuda_ctx(0, |ctx, m| {
            let d_input = ctx
                .new_tensor_view(h_input.as_slice())
                .expect("alloc failed");
            let mut d_output = ctx
                .new_tensor_view(h_output.as_mut_slice())
                .expect("alloc failed");

            let smem_bytes = block_size * core::mem::size_of::<f32>() as u32;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, smem_bytes);
            reduce_sum::launch(config, ctx, m, &d_input, &mut d_output, n)
                .expect("reduce_sum kernel launch failed");

            d_output
                .copy_to_host(&mut h_output)
                .expect("copy from device failed");
        });

        let total: f32 = h_output.iter().sum();
        assert!(
            (total - n as f32).abs() < 1e-2,
            "reduce_sum: got {}, expected {}",
            total,
            n as f32
        );
    }

    #[test]
    fn test_reduce_sum_small() {
        test_reduce_sum_impl(32, 16);
    }

    #[test]
    fn test_reduce_sum_medium() {
        test_reduce_sum_impl(1024, 256);
    }

    #[test]
    fn test_reduce_sum_large() {
        test_reduce_sum_impl(4096, 256);
    }

    #[test]
    fn test_reduce_sum_remainder() {
        // Non-multiple-of-block-size to catch OOB bugs
        test_reduce_sum_impl(100, 16);
    }
```

- [ ] **Step 3: Build to verify compilation**

```bash
cd /home/sanghle/work/seguru/examples
cargo build -p reduce
```

Expected: Build succeeds.

- [ ] **Step 4: Run tests**

```bash
cd /home/sanghle/work/seguru/examples
cargo test -p reduce
```

Expected: All tests pass (both existing `reduce_per_grid` tests and new `reduce_sum` tests).

- [ ] **Step 5: Commit**

```bash
cd /home/sanghle/work/seguru
git add examples/reduce/src/lib.rs
git commit -m "examples: add CUDA-style reduce_sum kernel for porting PoC

Add a direct translation of the classic CUDA parallel reduction kernel
to SeGuRu, mirroring the CUDA C++ structure closely. Uses dynamic
shared memory, sync_threads(), and explicit tree reduction.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Port Histogram Kernel to SeGuRu

**Files:**
- Create: `examples/histogram/Cargo.toml`
- Create: `examples/histogram/src/lib.rs`
- Modify: `examples/Cargo.toml` (add workspace member)

- [ ] **Step 1: Create the crate Cargo.toml**

Create `examples/histogram/Cargo.toml`:

```toml
[package]
name = "histogram"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Add workspace member**

In `examples/Cargo.toml`, add `"histogram"` to the `members` list:

```toml
[workspace]
members = [
    "hello",
    "histogram",
    "llm-rs-gpu",
    "matrix/matmul-gpu",
    "matrix/matmul-host",
    "reduce",
    "syntax/gpu",
    "syntax/host",
    "vector_add",
]
```

- [ ] **Step 3: Write the kernel and test**

Create `examples/histogram/src/lib.rs`:

```rust
use gpu::prelude::*;
use gpu::sync::{Atomic, SharedAtomic};

const NUM_BINS: usize = 256;

/// Histogram with shared memory + atomics.
///
/// CUDA equivalent:
///   __global__ void histogram(const unsigned int *data, unsigned int *bins, int n) {
///       __shared__ unsigned int smem_bins[256];
///       int tid = threadIdx.x;
///       int idx = blockIdx.x * blockDim.x + threadIdx.x;
///       int stride = blockDim.x * gridDim.x;
///       for (int i = tid; i < NUM_BINS; i += blockDim.x) smem_bins[i] = 0;
///       __syncthreads();
///       for (int i = idx; i < n; i += stride) atomicAdd(&smem_bins[data[i]], 1);
///       __syncthreads();
///       for (int i = tid; i < NUM_BINS; i += blockDim.x) atomicAdd(&bins[i], smem_bins[i]);
///   }
#[gpu::cuda_kernel(dynamic_shared)]
pub fn histogram(data: &[u32], bins: &mut [u32], n: usize) {
    let smem_bins = smem_alloc.alloc::<u32>(NUM_BINS);
    let mut smem_chunk = smem_bins.chunk_mut(MapLinear::new(1));

    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = (block_id::<DimX>() * bdim + tid) as usize;
    let stride = (bdim * grid_dim::<DimX>()) as usize;

    // Initialize shared memory bins to zero
    let mut i = tid as usize;
    while i < NUM_BINS {
        smem_chunk[0] = 0;
        i += bdim as usize;
    }
    sync_threads();

    // Accumulate into shared memory bins using atomics
    let smem_atomic = SharedAtomic::new(smem_bins);
    let mut i = idx;
    while i < n {
        let bin = data[i] as usize;
        smem_atomic.index(bin).atomic_addi(1u32);
        i += stride;
    }
    sync_threads();

    // Merge shared memory bins into global bins using atomics
    let bins_atomic = Atomic::new(bins);
    let mut i = tid as usize;
    while i < NUM_BINS {
        let local_count = *smem_bins[i];
        if local_count > 0 {
            bins_atomic.index(i).atomic_addi(local_count);
        }
        i += bdim as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    #[test]
    fn test_histogram_uniform() {
        let n: usize = 1 << 16; // 65536
        // Fill with values 0..255 repeating
        let h_data: Vec<u32> = (0..n).map(|i| (i % NUM_BINS) as u32).collect();
        let mut h_bins: Vec<u32> = vec![0u32; NUM_BINS];

        cuda_ctx(0, |ctx, m| {
            let d_data = ctx
                .new_tensor_view(h_data.as_slice())
                .expect("alloc data failed");
            let mut d_bins = ctx
                .new_tensor_view(h_bins.as_mut_slice())
                .expect("alloc bins failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let num_blocks = num_blocks.min(1024);
            let smem_bytes = (NUM_BINS as u32) * core::mem::size_of::<u32>() as u32;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, smem_bytes);
            histogram::launch(config, ctx, m, &d_data, &mut d_bins, n)
                .expect("histogram kernel launch failed");

            d_bins
                .copy_to_host(&mut h_bins)
                .expect("copy from device failed");
        });

        let expected = (n / NUM_BINS) as u32;
        for (i, count) in h_bins.iter().enumerate() {
            assert_eq!(
                *count, expected,
                "Bin {} has count {}, expected {}",
                i, count, expected
            );
        }
    }

    #[test]
    fn test_histogram_single_bin() {
        let n: usize = 1024;
        // All values map to bin 42
        let h_data: Vec<u32> = vec![42u32; n];
        let mut h_bins: Vec<u32> = vec![0u32; NUM_BINS];

        cuda_ctx(0, |ctx, m| {
            let d_data = ctx
                .new_tensor_view(h_data.as_slice())
                .expect("alloc data failed");
            let mut d_bins = ctx
                .new_tensor_view(h_bins.as_mut_slice())
                .expect("alloc bins failed");

            let block_size: u32 = 256;
            let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;
            let smem_bytes = (NUM_BINS as u32) * core::mem::size_of::<u32>() as u32;
            let config = gpu_host::gpu_config!(num_blocks, 1, 1, block_size, 1, 1, smem_bytes);
            histogram::launch(config, ctx, m, &d_data, &mut d_bins, n)
                .expect("histogram kernel launch failed");

            d_bins
                .copy_to_host(&mut h_bins)
                .expect("copy from device failed");
        });

        assert_eq!(h_bins[42], n as u32, "Bin 42 should have all {} counts", n);
        let total: u32 = h_bins.iter().sum();
        assert_eq!(total, n as u32, "Total count should be {}", n);
    }
}
```

- [ ] **Step 4: Build to verify compilation**

```bash
cd /home/sanghle/work/seguru/examples
cargo build -p histogram
```

Expected: Build succeeds.

- [ ] **Step 5: Run tests**

```bash
cd /home/sanghle/work/seguru/examples
cargo test -p histogram
```

Expected: Both `test_histogram_uniform` and `test_histogram_single_bin` pass.

- [ ] **Step 6: Commit**

```bash
cd /home/sanghle/work/seguru
git add examples/histogram/ examples/Cargo.toml
git commit -m "examples: add histogram SeGuRu port from CUDA

Port the classic CUDA histogram kernel to SeGuRu. Uses shared memory
(GpuShared), shared memory atomics (SharedAtomic), and global atomics
(Atomic) for the two-phase local-accumulate + global-merge pattern.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Write Porting Assessment Document

**Files:**
- Create: `docs/cuda-to-seguru-porting-assessment.md`

- [ ] **Step 1: Write the assessment based on porting experience**

Create `docs/cuda-to-seguru-porting-assessment.md` with the following template. Fill in actual findings after completing Tasks 1-4:

```markdown
# CUDA→SeGuRu Porting Assessment

## Summary

This document assesses the feasibility of automating CUDA C++ → SeGuRu Rust
kernel translation, based on porting 3 classic CUDA kernels.

## Kernels Ported

| Kernel | CUDA Patterns Used | SeGuRu Equivalents | Issues Encountered |
|--------|-------------------|--------------------|--------------------|
| vector_add | Thread indexing, grid-stride loop, global memory | thread_id/block_id, while loop, slice indexing | (fill after porting) |
| reduce_sum | Shared memory, __syncthreads, tree reduction | GpuShared/smem_alloc, sync_threads, while loop | (fill after porting) |
| histogram | Shared memory, atomicAdd (shared + global) | GpuShared, SharedAtomic, Atomic | (fill after porting) |

## Mechanical Transforms (Automatable)

These transforms are purely syntactic and can be handled by string/AST replacement:

1. **Kernel attribute**: `__global__` → `#[gpu::cuda_kernel]` (100% mechanical)
2. **Thread intrinsics**: `threadIdx.x` → `thread_id::<DimX>()` (100% mechanical)
3. **Block intrinsics**: `blockIdx.x` → `block_id::<DimX>()` (100% mechanical)
4. **Dimension intrinsics**: `blockDim.x` → `block_dim::<DimX>()` (100% mechanical)
5. **Sync barriers**: `__syncthreads()` → `sync_threads()` (100% mechanical)
6. **Math functions**: `sinf(x)` → `x.sin()` (100% mechanical)
7. **Host boilerplate**: `cudaMalloc/cudaMemcpy/cudaFree` → `new_tensor_view/copy_to_host/Drop` (100% mechanical)
8. **Launch config**: `kernel<<<grid, block, smem>>>` → `gpu_config!() + ::launch()` (100% mechanical)

## Semantic Transforms (Require LLM)

These transforms require understanding intent, not just syntax:

1. **Type mapping**: `float*` → `&[f32]` vs `&mut [f32]` (need to analyze read/write usage)
2. **Pointer arithmetic** → slice indexing (need to understand access patterns)
3. **Shared memory declaration**: static `__shared__` vs `extern __shared__` → `GpuShared` vs `smem_alloc` (need to determine if size is const)
4. **Atomic wrappers**: `atomicAdd(&x, v)` → `Atomic::new(&mut x).atomic_addi(v)` (need to handle different wrappers for shared vs global memory)
5. **Loop translation**: `for(;;)` → `while` or `for..in` (need to choose idiomatic Rust)
6. **Bounds handling**: CUDA silently ignores OOB; Rust panics (need to add guards or restructure)

## Estimated Automation Breakdown

| Category | % of Translation Effort | Automatable? |
|----------|------------------------|-------------|
| Kernel signature + attribute | 5% | Yes (rule-based) |
| Thread/block intrinsics | 10% | Yes (rule-based) |
| Sync/barrier | 2% | Yes (rule-based) |
| Host boilerplate | 20% | Yes (rule-based) |
| Type mapping | 10% | Partially (heuristics + LLM) |
| Kernel body logic | 40% | LLM required |
| Shared memory patterns | 8% | LLM required |
| Atomic patterns | 5% | LLM required |

**Overall: ~37% purely mechanical, ~63% requires semantic understanding (LLM).**

## Gaps / Missing Features

(Fill after porting — document any CUDA features that have no SeGuRu equivalent)

## Recommendations

(Fill after porting — recommend approach for automation tool)
```

- [ ] **Step 2: Update with actual findings from porting**

After Tasks 2-4 are complete, update all "(fill after porting)" placeholders with actual observations from the porting process.

- [ ] **Step 3: Commit**

```bash
cd /home/sanghle/work/seguru
git add docs/cuda-to-seguru-porting-assessment.md
git commit -m "docs: add CUDA→SeGuRu porting feasibility assessment

Document findings from porting 3 CUDA kernels to SeGuRu, including
mechanical vs semantic transform breakdown and automation recommendations.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
