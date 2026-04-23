# KernelBench Performance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark suite that compares SeGuRu kernel performance against CUDA C and PyTorch equivalents for all 43 KernelBench Level 1 kernels.

**Architecture:** A Rust binary (`bench`) in the kernelbench crate that times both SeGuRu kernels (via existing library) and CUDA C kernels (via FFI to a static library compiled from `cuda/kernels.cu`). A separate Python script benchmarks PyTorch. A comparison script merges all results into CSV + Markdown + terminal output.

**Tech Stack:** Rust (std::time, gpu_host), CUDA C (nvcc, CUDA events), Python (PyTorch, torch.cuda.Event), build.rs for nvcc compilation.

---

### Task 1: Build Infrastructure — build.rs, Cargo.toml, CUDA Header

**Files:**
- Create: `examples/kernelbench/build.rs`
- Create: `examples/kernelbench/cuda/kernels.h`
- Modify: `examples/kernelbench/Cargo.toml`

This task sets up the build pipeline: `build.rs` compiles `cuda/*.cu` via nvcc into a static library, and the Cargo.toml gains a `bench` feature + binary target.

- [ ] **Step 1: Update Cargo.toml**

Add `bench` feature, `cc` build dependency, `bindgen` build dependency, and bench binary:

```toml
[package]
name = "kernelbench"
version = "0.1.0"
edition = "2024"

[features]
bench = []

[dependencies]
gpu = { workspace = true }
gpu_host = { workspace = true }

[dev-dependencies]
rand = "0.9"

[build-dependencies]
cc = "1"

[[bin]]
name = "bench"
path = "src/bin/bench.rs"
required-features = ["bench"]
```

- [ ] **Step 2: Create the CUDA header file**

Create `examples/kernelbench/cuda/kernels.h` with C-linkage declarations for all 43 benchmark functions. Each function takes host pointers, dimensions, warmup count, iteration count, and returns median kernel time in microseconds.

```c
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Elementwise (11)
float bench_relu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_leaky_relu_forward(const float* input, float* output, int n, float alpha, int grid, int block, int warmup, int iters);
float bench_sigmoid_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_tanh_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_swish_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_selu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_hard_sigmoid_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_softplus_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_softsign_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_elu_forward(const float* input, float* output, int n, float alpha, int grid, int block, int warmup, int iters);
float bench_hard_tanh_forward(const float* input, float* output, int n, float min_val, float max_val, int grid, int block, int warmup, int iters);

// GELU (2)
float bench_gelu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);
float bench_mingpt_new_gelu_forward(const float* input, float* output, int n, int grid, int block, int warmup, int iters);

// Matmul (6)
float bench_matmul_forward(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_a(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_b(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_transposed_both(const float* a, const float* b, float* c, int m, int n, int k, int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters);
float bench_matmul_batched(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);
float bench_tensor3d_matmul(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);

// Matvec (3)
float bench_matvec_forward(const float* a, const float* x, float* y, int m, int n, int grid, int block, int warmup, int iters);
float bench_scalar_multiply(const float* input, float* output, float s, int n, int grid, int block, int warmup, int iters);
float bench_tensor3d_matvec(const float* a, const float* b, float* c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters);

// Reduction (4)
float bench_sum_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_mean_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_max_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_min_reduce(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);

// Argreduce (2)
float bench_argmax_reduce(const float* input, unsigned int* output, int batch, int dim, int block, int warmup, int iters);
float bench_argmin_reduce(const float* input, unsigned int* output, int batch, int dim, int block, int warmup, int iters);

// Softmax (2)
float bench_softmax_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_log_softmax_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);

// Norm (5)
float bench_rms_norm_forward(const float* input, float* output, int batch, int dim, float eps, int block, int warmup, int iters);
float bench_frobenius_norm_forward(const float* input, float* output, int n, int block, int warmup, int iters);
float bench_l1_norm_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_l2_norm_forward(const float* input, float* output, int batch, int dim, int block, int warmup, int iters);
float bench_layer_norm_forward(const float* input, float* output, int batch, int dim, float eps, int block, int warmup, int iters);

// Loss (4)
float bench_mse_loss_forward(const float* predictions, const float* targets, float* output, int n, int block, int warmup, int iters);
float bench_huber_loss_forward(const float* predictions, const float* targets, float* output, int n, float delta, int block, int warmup, int iters);
float bench_kl_div_loss_forward(const float* log_predictions, const float* targets, float* output, int n, int block, int warmup, int iters);
float bench_hinge_loss_forward(const float* predictions, const float* targets, float* output, int n, int block, int warmup, int iters);

// Cumulative (4)
float bench_cumsum_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumprod_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumsum_reverse_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);
float bench_cumsum_exclusive_forward(const float* input, float* output, int batch, int dim, int warmup, int iters);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 3: Create build.rs**

```rust
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA when bench feature is enabled
    if env::var("CARGO_FEATURE_BENCH").is_err() {
        return;
    }

    let cuda_dir = "/usr/local/cuda";
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_src = format!("{}/cuda/kernels.cu", manifest_dir);
    let obj_path = format!("{}/kernels.o", out_dir);
    let lib_path = format!("{}/libkernelbench_cuda.a", out_dir);

    // Compile with nvcc
    let status = Command::new(format!("{}/bin/nvcc", cuda_dir))
        .args(&[
            "-c", &cuda_src,
            "-o", &obj_path,
            "-O2",
            "--compiler-options", "-fPIC",
            "-I", &format!("{}/cuda", manifest_dir),
        ])
        .status()
        .expect("Failed to run nvcc — is CUDA installed?");
    assert!(status.success(), "nvcc compilation failed");

    // Create static library
    let status = Command::new("ar")
        .args(&["rcs", &lib_path, &obj_path])
        .status()
        .expect("Failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=kernelbench_cuda");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_dir);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/kernels.h");
    println!("cargo:rerun-if-changed=build.rs");
}
```

- [ ] **Step 4: Verify build.rs compiles (without CUDA file yet)**

Create a placeholder `cuda/kernels.cu`:
```c
#include "kernels.h"
// Placeholder — kernels will be added in Task 2
```

Run: `cd examples && cargo build -p kernelbench` (without bench feature — should succeed since build.rs returns early)

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add examples/kernelbench/Cargo.toml examples/kernelbench/build.rs examples/kernelbench/cuda/
git commit -m "feat(kernelbench): add bench build infrastructure — build.rs, CUDA header, feature gate"
```

---

### Task 2: CUDA Reference Kernels — Elementwise, GELU, Matmul

**Files:**
- Modify: `examples/kernelbench/cuda/kernels.cu`

Implement the first 19 CUDA kernels (elementwise + gelu + matmul) plus their benchmark wrapper functions. Each wrapper: allocates device memory, copies input, runs warmup, times with CUDA events, returns median µs.

**IMPORTANT**: The benchmark harness helper is defined at the top of `kernels.cu` and reused by ALL benchmark functions. It handles: cudaMalloc, cudaMemcpy H2D, warmup loop, CUDA event timing loop, median calculation, cudaFree.

- [ ] **Step 1: Write the benchmark harness helper and elementwise kernels**

Replace the placeholder `cuda/kernels.cu` with:

```cuda
#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

// ---------- Helper: sort for median ----------
static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float median_of(float* arr, int n) {
    qsort(arr, n, sizeof(float), cmp_float);
    return arr[n / 2];
}

// ========== ELEMENTWISE KERNELS ==========

__global__ void k_relu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : 0.f; }
}

__global__ void k_leaky_relu(const float* in, float* out, int n, float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : alpha * x; }
}

__global__ void k_sigmoid(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = 1.f / (1.f + expf(-in[tid]));
}

__global__ void k_tanh_fwd(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = tanhf(in[tid]);
}

__global__ void k_swish(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x / (1.f + expf(-x)); }
}

__global__ void k_selu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        const float alpha = 1.6732632423543772f;
        const float scale = 1.0507009873554805f;
        out[tid] = scale * (x > 0.f ? x : alpha * (expf(x) - 1.f));
    }
}

__global__ void k_hard_sigmoid(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        float v = x / 6.f + 0.5f;
        out[tid] = fminf(fmaxf(v, 0.f), 1.f);
    }
}

__global__ void k_softplus(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = logf(1.f + expf(in[tid]));
}

__global__ void k_softsign(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x / (1.f + fabsf(x)); }
}

__global__ void k_elu(const float* in, float* out, int n, float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : alpha * (expf(x) - 1.f); }
}

__global__ void k_hard_tanh(const float* in, float* out, int n, float min_val, float max_val) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = fminf(fmaxf(in[tid], min_val), max_val);
}

// GELU variants
__global__ void k_gelu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
    }
}

__global__ void k_mingpt_new_gelu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

// --- Elementwise bench wrappers ---
// Pattern: alloc device mem, copy in, warmup, time iters, return median µs

static float bench_elementwise_1in_1out(
    void (*kernel)(const float*, float*, int),
    const float* h_in, float* h_out, int n, int grid, int block, int warmup, int iters
) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
    }

    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_in, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
        times[i] *= 1000.f; // ms → µs
    }
    float result = median_of(times, iters);

    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(times);
    cudaFree(d_in);
    cudaFree(d_out);
    return result;
}

// Wrapper for kernels with function pointer — unfortunately CUDA doesn't allow
// passing __global__ functions as regular function pointers easily. 
// We use a macro to generate each bench function.

#define BENCH_ELEM_1IN(name, kernel_call) \
extern "C" float bench_##name(const float* h_in, float* h_out, int n, int grid, int block, int warmup, int iters) { \
    float *d_in, *d_out; \
    cudaMalloc(&d_in, n * sizeof(float)); \
    cudaMalloc(&d_out, n * sizeof(float)); \
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice); \
    for (int i = 0; i < warmup; i++) { kernel_call; cudaDeviceSynchronize(); } \
    float* times = (float*)malloc(iters * sizeof(float)); \
    cudaEvent_t ev_start, ev_stop; \
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop); \
    for (int i = 0; i < iters; i++) { \
        cudaEventRecord(ev_start); kernel_call; cudaEventRecord(ev_stop); \
        cudaEventSynchronize(ev_stop); cudaEventElapsedTime(&times[i], ev_start, ev_stop); \
        times[i] *= 1000.f; \
    } \
    float result = median_of(times, iters); \
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost); \
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop); free(times); \
    cudaFree(d_in); cudaFree(d_out); return result; \
}

BENCH_ELEM_1IN(relu_forward, k_relu<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(sigmoid_forward, k_sigmoid<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(tanh_forward, k_tanh_fwd<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(swish_forward, k_swish<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(selu_forward, k_selu<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(hard_sigmoid_forward, k_hard_sigmoid<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(softplus_forward, k_softplus<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(softsign_forward, k_softsign<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(gelu_forward, k_gelu<<<grid, block>>>(d_in, d_out, n))
BENCH_ELEM_1IN(mingpt_new_gelu_forward, k_mingpt_new_gelu<<<grid, block>>>(d_in, d_out, n))

// Leaky ReLU, ELU — extra float param
extern "C" float bench_leaky_relu_forward(const float* h_in, float* h_out, int n, float alpha, int grid, int block, int warmup, int iters) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_leaky_relu<<<grid, block>>>(d_in, d_out, n, alpha); cudaDeviceSynchronize(); }
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_leaky_relu<<<grid, block>>>(d_in, d_out, n, alpha); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f;
    }
    float result = median_of(times, iters);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); cudaFree(d_in); cudaFree(d_out);
    return result;
}

extern "C" float bench_elu_forward(const float* h_in, float* h_out, int n, float alpha, int grid, int block, int warmup, int iters) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_elu<<<grid, block>>>(d_in, d_out, n, alpha); cudaDeviceSynchronize(); }
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_elu<<<grid, block>>>(d_in, d_out, n, alpha); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f;
    }
    float result = median_of(times, iters);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); cudaFree(d_in); cudaFree(d_out);
    return result;
}

extern "C" float bench_hard_tanh_forward(const float* h_in, float* h_out, int n, float min_val, float max_val, int grid, int block, int warmup, int iters) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_hard_tanh<<<grid, block>>>(d_in, d_out, n, min_val, max_val); cudaDeviceSynchronize(); }
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_hard_tanh<<<grid, block>>>(d_in, d_out, n, min_val, max_val); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f;
    }
    float result = median_of(times, iters);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); cudaFree(d_in); cudaFree(d_out);
    return result;
}
```

- [ ] **Step 2: Add matmul CUDA kernels + bench wrappers**

Append to `kernels.cu`:

```cuda
// ========== MATMUL KERNELS ==========

__global__ void k_matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[row * k + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

__global__ void k_matmul_ta(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[i * m + row] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

__global__ void k_matmul_tb(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[row * k + i] * b[col * k + i];
        c[row * n + col] = sum;
    }
}

__global__ void k_matmul_tab(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[i * m + row] * b[col * k + i];
        c[row * n + col] = sum;
    }
}

__global__ void k_matmul_batched(const float* a, const float* b, float* c, int batch, int m, int n, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch * m * n;
    if (tid < total) {
        int bi = tid / (m * n);
        int rem = tid % (m * n);
        int row = rem / n;
        int col = rem % n;
        float sum = 0.f;
        for (int i = 0; i < k; i++)
            sum += a[bi * m * k + row * k + i] * b[bi * k * n + i * n + col];
        c[bi * m * n + row * n + col] = sum;
    }
}

// tensor3d_matmul is identical to batched matmul in implementation
__global__ void k_tensor3d_matmul(const float* a, const float* b, float* c, int batch, int m, int n, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch * m * n;
    if (tid < total) {
        int bi = tid / (m * n);
        int rem = tid % (m * n);
        int row = rem / n;
        int col = rem % n;
        float sum = 0.f;
        for (int i = 0; i < k; i++)
            sum += a[bi * m * k + row * k + i] * b[bi * k * n + i * n + col];
        c[bi * m * n + row * n + col] = sum;
    }
}

// Matmul bench wrappers (2D grid)
#define BENCH_MATMUL_2D(name, kernel_call) \
extern "C" float bench_##name(const float* h_a, const float* h_b, float* h_c, int m, int n, int k, \
    int grid_x, int grid_y, int block_x, int block_y, int warmup, int iters) { \
    float *d_a, *d_b, *d_c; \
    cudaMalloc(&d_a, m * k * sizeof(float)); \
    cudaMalloc(&d_b, k * n * sizeof(float)); \
    cudaMalloc(&d_c, m * n * sizeof(float)); \
    cudaMemcpy(d_a, h_a, m * k * sizeof(float), cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, h_b, k * n * sizeof(float), cudaMemcpyHostToDevice); \
    dim3 grid(grid_x, grid_y); dim3 blk(block_x, block_y); \
    for (int i = 0; i < warmup; i++) { kernel_call; cudaDeviceSynchronize(); } \
    float* times = (float*)malloc(iters * sizeof(float)); \
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e); \
    for (int i = 0; i < iters; i++) { \
        cudaEventRecord(ev_s); kernel_call; cudaEventRecord(ev_e); \
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f; \
    } \
    float result = median_of(times, iters); \
    cudaMemcpy(h_c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost); \
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); \
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return result; \
}

BENCH_MATMUL_2D(matmul_forward, k_matmul<<<grid, blk>>>(d_a, d_b, d_c, m, n, k))
BENCH_MATMUL_2D(matmul_transposed_a, k_matmul_ta<<<grid, blk>>>(d_a, d_b, d_c, m, n, k))
BENCH_MATMUL_2D(matmul_transposed_b, k_matmul_tb<<<grid, blk>>>(d_a, d_b, d_c, m, n, k))
BENCH_MATMUL_2D(matmul_transposed_both, k_matmul_tab<<<grid, blk>>>(d_a, d_b, d_c, m, n, k))

// Batched matmul bench wrappers (1D grid)
#define BENCH_MATMUL_BATCHED(name, kernel_call) \
extern "C" float bench_##name(const float* h_a, const float* h_b, float* h_c, int batch, int m, int n, int k, \
    int grid, int block, int warmup, int iters) { \
    int total = batch * m * n; \
    float *d_a, *d_b, *d_c; \
    cudaMalloc(&d_a, batch * m * k * sizeof(float)); \
    cudaMalloc(&d_b, batch * k * n * sizeof(float)); \
    cudaMalloc(&d_c, total * sizeof(float)); \
    cudaMemcpy(d_a, h_a, batch * m * k * sizeof(float), cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, h_b, batch * k * n * sizeof(float), cudaMemcpyHostToDevice); \
    for (int i = 0; i < warmup; i++) { kernel_call; cudaDeviceSynchronize(); } \
    float* times = (float*)malloc(iters * sizeof(float)); \
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e); \
    for (int i = 0; i < iters; i++) { \
        cudaEventRecord(ev_s); kernel_call; cudaEventRecord(ev_e); \
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f; \
    } \
    float result = median_of(times, iters); \
    cudaMemcpy(h_c, d_c, total * sizeof(float), cudaMemcpyDeviceToHost); \
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); \
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return result; \
}

BENCH_MATMUL_BATCHED(matmul_batched, k_matmul_batched<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k))
BENCH_MATMUL_BATCHED(tensor3d_matmul, k_tensor3d_matmul<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k))
```

- [ ] **Step 3: Verify CUDA compilation**

Run: `cd examples/kernelbench && /usr/local/cuda/bin/nvcc -c cuda/kernels.cu -o /tmp/test_kernels.o -O2 -I cuda/`

Expected: Compiles without errors. Remove `/tmp/test_kernels.o` after.

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/cuda/
git commit -m "feat(kernelbench): CUDA reference kernels — elementwise, GELU, matmul (19 kernels)"
```

---

### Task 3: CUDA Reference Kernels — Matvec, Reduction, Argreduce, Softmax, Norm, Loss, Cumulative

**Files:**
- Modify: `examples/kernelbench/cuda/kernels.cu`

Implement the remaining 24 CUDA kernels with bench wrappers.

- [ ] **Step 1: Add matvec + scalar kernels**

Append to `kernels.cu`:

```cuda
// ========== MATVEC KERNELS ==========

__global__ void k_matvec(const float* a, const float* x, float* y, int m, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        float sum = 0.f;
        for (int i = 0; i < n; i++) sum += a[tid * n + i] * x[i];
        y[tid] = sum;
    }
}

__global__ void k_scalar_multiply(const float* in, float* out, float s, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = in[tid] * s;
}

extern "C" float bench_matvec_forward(const float* h_a, const float* h_x, float* h_y, int m, int n, int grid, int block, int warmup, int iters) {
    float *d_a, *d_x, *d_y;
    cudaMalloc(&d_a, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_matvec<<<grid, block>>>(d_a, d_x, d_y, m, n); cudaDeviceSynchronize(); }
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_matvec<<<grid, block>>>(d_a, d_x, d_y, m, n); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f;
    }
    float result = median_of(times, iters);
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times);
    cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);
    return result;
}

extern "C" float bench_scalar_multiply(const float* h_in, float* h_out, float s, int n, int grid, int block, int warmup, int iters) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_scalar_multiply<<<grid, block>>>(d_in, d_out, s, n); cudaDeviceSynchronize(); }
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_scalar_multiply<<<grid, block>>>(d_in, d_out, s, n); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f;
    }
    float result = median_of(times, iters);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times);
    cudaFree(d_in); cudaFree(d_out);
    return result;
}

// tensor3d_matvec reuses bench_tensor3d_matmul from matmul section (same algo)
extern "C" float bench_tensor3d_matvec(const float* h_a, const float* h_b, float* h_c, int batch, int m, int n, int k, int grid, int block, int warmup, int iters) {
    int total_out = batch * m * n;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, batch * m * k * sizeof(float));
    cudaMalloc(&d_b, batch * k * n * sizeof(float));
    cudaMalloc(&d_c, total_out * sizeof(float));
    cudaMemcpy(d_a, h_a, batch * m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, batch * k * n * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) { k_tensor3d_matmul<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k); cudaDeviceSynchronize(); }
    float* times_arr = (float*)malloc(iters * sizeof(float));
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(ev_s); k_tensor3d_matmul<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k); cudaEventRecord(ev_e);
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times_arr[i], ev_s, ev_e); times_arr[i] *= 1000.f;
    }
    float result = median_of(times_arr, iters);
    cudaMemcpy(h_c, d_c, total_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times_arr);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return result;
}
```

- [ ] **Step 2: Add reduction kernels (sum/mean/max/min) + bench wrappers**

Append to `kernels.cu`. Use shared memory tree reduction matching SeGuRu pattern: grid=(batch,1,1), block=(block_size,1,1), shared=block*4 bytes.

```cuda
// ========== REDUCTION KERNELS ==========

__global__ void k_sum_reduce(const float* input, float* output, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    float local_sum = 0.f;
    for (int i = tid; i < dim; i += bdim)
        local_sum += input[bid * dim + i];
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[bid] = smem[0];
}

__global__ void k_mean_reduce(const float* input, float* output, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    float local_sum = 0.f;
    for (int i = tid; i < dim; i += bdim)
        local_sum += input[bid * dim + i];
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[bid] = smem[0] / (float)dim;
}

__global__ void k_max_reduce(const float* input, float* output, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    float local_max = -3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) {
        float v = input[bid * dim + i];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[bid] = smem[0];
}

__global__ void k_min_reduce(const float* input, float* output, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    float local_min = 3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) {
        float v = input[bid * dim + i];
        if (v < local_min) local_min = v;
    }
    smem[tid] = local_min;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] < smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[bid] = smem[0];
}

#define BENCH_REDUCE(name, kernel_call) \
extern "C" float bench_##name(const float* h_in, float* h_out, int batch, int dim, int block, int warmup, int iters) { \
    float *d_in, *d_out; \
    cudaMalloc(&d_in, batch * dim * sizeof(float)); \
    cudaMalloc(&d_out, batch * sizeof(float)); \
    cudaMemcpy(d_in, h_in, batch * dim * sizeof(float), cudaMemcpyHostToDevice); \
    int smem = block * sizeof(float); \
    for (int i = 0; i < warmup; i++) { kernel_call; cudaDeviceSynchronize(); } \
    float* times = (float*)malloc(iters * sizeof(float)); \
    cudaEvent_t ev_s, ev_e; cudaEventCreate(&ev_s); cudaEventCreate(&ev_e); \
    for (int i = 0; i < iters; i++) { \
        cudaEventRecord(ev_s); kernel_call; cudaEventRecord(ev_e); \
        cudaEventSynchronize(ev_e); cudaEventElapsedTime(&times[i], ev_s, ev_e); times[i] *= 1000.f; \
    } \
    float result = median_of(times, iters); \
    cudaMemcpy(h_out, d_out, batch * sizeof(float), cudaMemcpyDeviceToHost); \
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_e); free(times); \
    cudaFree(d_in); cudaFree(d_out); return result; \
}

BENCH_REDUCE(sum_reduce, k_sum_reduce<<<batch, block, smem>>>(d_in, d_out, dim))
BENCH_REDUCE(mean_reduce, k_mean_reduce<<<batch, block, smem>>>(d_in, d_out, dim))
BENCH_REDUCE(max_reduce, k_max_reduce<<<batch, block, smem>>>(d_in, d_out, dim))
BENCH_REDUCE(min_reduce, k_min_reduce<<<batch, block, smem>>>(d_in, d_out, dim))
```

- [ ] **Step 3: Add argreduce, softmax, norm, loss, cumulative kernels + bench wrappers**

Append to `kernels.cu`. Each follows the same pattern as reduction — shared memory tree reduction for row-wise ops, sequential scan for cumulative.

The implementer should write these following the exact same patterns as Steps 1-2:
- **Argreduce** (2): dual shared memory (float vals + uint indices), grid=batch, block=block_size, smem=block*8
- **Softmax** (2): 3-pass (max, sum_exp, normalize), grid=batch, block=block_size, smem=block*4
- **Norm** (5): rms/layer/l1/l2 = row-wise reduction + normalize; frobenius = global reduction to scalar
- **Loss** (4): global reduction (grid=1), smem=block*4. MSE/Huber/KLDiv/Hinge
- **Cumulative** (4): 1 thread per row (grid=batch, block=1), sequential scan, no shared memory

Each kernel's `__global__` function should match the algorithmic approach of its SeGuRu counterpart exactly (read the SeGuRu source files listed in the spec).

- [ ] **Step 4: Verify full CUDA compilation**

Run: `cd examples/kernelbench && /usr/local/cuda/bin/nvcc -c cuda/kernels.cu -o /tmp/test_kernels.o -O2 -I cuda/`

Expected: Compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add examples/kernelbench/cuda/
git commit -m "feat(kernelbench): CUDA reference kernels — matvec, reduction, argreduce, softmax, norm, loss, cumulative (24 kernels)"
```

---

### Task 4: Rust FFI Bindings Module

**Files:**
- Create: `examples/kernelbench/src/cuda_ffi.rs`
- Modify: `examples/kernelbench/src/lib.rs`

Create the FFI declarations that match `cuda/kernels.h`.

- [ ] **Step 1: Create `src/cuda_ffi.rs`**

This module declares all `extern "C"` functions from the CUDA library. It is only compiled when the `bench` feature is active. Since the kernelbench lib.rs is `#![no_std]`, we need to put the FFI module in a separate file that is only used by the bench binary (which is std). So instead, we put the FFI declarations directly in the bench binary or in a cfg-gated module.

Actually, the bench binary is a standard Rust binary (not no_std), so it can have its own FFI module. Create `src/cuda_ffi.rs`:

```rust
//! FFI bindings to CUDA reference kernels (compiled via build.rs)

#[allow(dead_code)]
extern "C" {
    // Elementwise
    pub fn bench_relu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_leaky_relu_forward(input: *const f32, output: *mut f32, n: i32, alpha: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_sigmoid_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tanh_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_swish_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_selu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hard_sigmoid_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_softplus_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_softsign_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_elu_forward(input: *const f32, output: *mut f32, n: i32, alpha: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hard_tanh_forward(input: *const f32, output: *mut f32, n: i32, min_val: f32, max_val: f32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // GELU
    pub fn bench_gelu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_mingpt_new_gelu_forward(input: *const f32, output: *mut f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Matmul
    pub fn bench_matmul_forward(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_a(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_b(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_transposed_both(a: *const f32, b: *const f32, c: *mut f32, m: i32, n: i32, k: i32, grid_x: i32, grid_y: i32, block_x: i32, block_y: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_matmul_batched(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tensor3d_matmul(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Matvec
    pub fn bench_matvec_forward(a: *const f32, x: *const f32, y: *mut f32, m: i32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_scalar_multiply(input: *const f32, output: *mut f32, s: f32, n: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_tensor3d_matvec(a: *const f32, b: *const f32, c: *mut f32, batch: i32, m: i32, n: i32, k: i32, grid: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Reduction
    pub fn bench_sum_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_mean_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_max_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_min_reduce(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Argreduce
    pub fn bench_argmax_reduce(input: *const f32, output: *mut u32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_argmin_reduce(input: *const f32, output: *mut u32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Softmax
    pub fn bench_softmax_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_log_softmax_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Norm
    pub fn bench_rms_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, eps: f32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_frobenius_norm_forward(input: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_l1_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_l2_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_layer_norm_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, eps: f32, block: i32, warmup: i32, iters: i32) -> f32;

    // Loss
    pub fn bench_mse_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_huber_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, delta: f32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_kl_div_loss_forward(log_predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_hinge_loss_forward(predictions: *const f32, targets: *const f32, output: *mut f32, n: i32, block: i32, warmup: i32, iters: i32) -> f32;

    // Cumulative
    pub fn bench_cumsum_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumprod_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumsum_reverse_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
    pub fn bench_cumsum_exclusive_forward(input: *const f32, output: *mut f32, batch: i32, dim: i32, warmup: i32, iters: i32) -> f32;
}
```

Note: This file is NOT added to `lib.rs` (which is `#![no_std]`). It will be included by the bench binary via `#[path]` or `mod` in the binary's own module tree.

- [ ] **Step 2: Commit**

```bash
git add examples/kernelbench/src/cuda_ffi.rs
git commit -m "feat(kernelbench): add CUDA FFI bindings for bench binary"
```

---

### Task 5: Rust Benchmark Binary

**Files:**
- Create: `examples/kernelbench/src/bin/bench.rs`

This is the main benchmark runner. It iterates over all 43 kernels at two sizes (small, large), runs SeGuRu (via gpu_host + `std::time::Instant`) and CUDA (via FFI), and outputs CSV.

- [ ] **Step 1: Create the benchmark binary**

Create `examples/kernelbench/src/bin/bench.rs`. The binary:
1. Defines a `BenchResult` struct: `{ kernel: String, category: String, size_label: String, n_elements: usize, seguru_us: f64, cuda_us: f64, ratio: f64 }`
2. Defines size configs (small + large) for each category
3. For each kernel, calls a `bench_*` function that:
   - Creates input data (deterministic, `(i % 7) as f32 * 0.1`)
   - Runs SeGuRu kernel inside `gpu_host::cuda_ctx` with warmup + timing
   - Calls the corresponding CUDA FFI function
   - Returns `(seguru_us, cuda_us)`
4. Outputs CSV header + rows to stdout
5. Prints a summary table to stderr

The binary should use `#[path = "../cuda_ffi.rs"] mod cuda_ffi;` to include FFI bindings.

**SeGuRu timing pattern** (used for every kernel):
```rust
const WARMUP: u32 = 3;
const ITERS: u32 = 10;

fn time_seguru_kernel(ctx: &impl gpu_host::GpuCtxSpace, m: &gpu_host::GpuModule, f: impl Fn()) -> f64 {
    for _ in 0..WARMUP {
        f();
        ctx.sync().unwrap();
    }
    let mut times = Vec::with_capacity(ITERS as usize);
    for _ in 0..ITERS {
        let start = std::time::Instant::now();
        f();
        ctx.sync().unwrap();
        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERS as usize / 2]
}
```

**Key implementation details:**
- The SeGuRu kernel launches happen inside `cuda_ctx(0, |ctx, m| { ... })` — all benchmarks for all kernels run within one `cuda_ctx` call
- Grid/block configs match the existing test patterns exactly
- Input sizes:
  - Elementwise/GELU/Scalar: small=4096, large=1_048_576
  - Matmul (2D): small=M=N=K=64, large=M=N=K=1024
  - Batched matmul: small=B=4,M=N=K=32, large=B=16,M=N=K=256
  - Matvec: small=M=64,N=64, large=M=4096,N=4096
  - Reduction/Argreduce/Softmax/Norm(row): small=batch=64,dim=256, large=batch=1024,dim=4096
  - Norm(global)/Loss: small=4096, large=1_048_576
  - Cumulative: small=batch=64,dim=256, large=batch=1024,dim=4096

The implementer should write all 43 kernel benchmarks, each following the same pattern. This is a large file but highly repetitive. Group by category and use helper functions to reduce code duplication.

- [ ] **Step 2: Verify build with bench feature**

Run: `cd examples && cargo build -p kernelbench --features bench --bin bench`

Expected: Build succeeds. (If compilation fails due to CUDA or linking, debug and fix.)

- [ ] **Step 3: Run the benchmark**

Run: `cd examples && cargo run -p kernelbench --features bench --bin bench 2>/dev/null`

Expected: CSV output with 86 rows (43 kernels × 2 sizes). All `seguru_us` and `cuda_us` values should be > 0.

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/src/bin/ examples/kernelbench/src/cuda_ffi.rs
git commit -m "feat(kernelbench): benchmark binary — times 43 SeGuRu + CUDA kernels, outputs CSV"
```

---

### Task 6: PyTorch Benchmark Script

**Files:**
- Create: `examples/kernelbench/bench_pytorch.py`

- [ ] **Step 1: Create PyTorch benchmark script**

The script benchmarks the same 43 operations using PyTorch GPU ops. For each:
1. Create input tensors on GPU (same deterministic data)
2. Warmup 3 runs + `torch.cuda.synchronize()`
3. Time 10 iterations with `torch.cuda.Event(enable_timing=True)`
4. Output CSV row: `kernel,category,size_label,n_elements,pytorch_us`

**PyTorch operation mapping:**
```python
ops = {
    "relu_forward": lambda x: torch.relu(x),
    "leaky_relu_forward": lambda x: F.leaky_relu(x, 0.01),
    "sigmoid_forward": lambda x: torch.sigmoid(x),
    "tanh_forward": lambda x: torch.tanh(x),
    "swish_forward": lambda x: x * torch.sigmoid(x),  # SiLU
    "selu_forward": lambda x: F.selu(x),
    "hard_sigmoid_forward": lambda x: F.hardsigmoid(x),
    "softplus_forward": lambda x: F.softplus(x),
    "softsign_forward": lambda x: F.softsign(x),
    "elu_forward": lambda x: F.elu(x, alpha=1.0),
    "hard_tanh_forward": lambda x: F.hardtanh(x, -1.0, 1.0),
    "gelu_forward": lambda x: F.gelu(x),
    "mingpt_new_gelu_forward": lambda x: F.gelu(x, approximate="tanh"),
    "matmul_forward": lambda a, b: torch.matmul(a, b),
    # ... transposed variants, batched, etc.
    "sum_reduce_forward": lambda x: torch.sum(x, dim=-1),
    "mean_reduce_forward": lambda x: torch.mean(x, dim=-1),
    "max_reduce_forward": lambda x: torch.max(x, dim=-1).values,
    "min_reduce_forward": lambda x: torch.min(x, dim=-1).values,
    "argmax_reduce_forward": lambda x: torch.argmax(x, dim=-1),
    "argmin_reduce_forward": lambda x: torch.argmin(x, dim=-1),
    "softmax_forward": lambda x: F.softmax(x, dim=-1),
    "log_softmax_forward": lambda x: F.log_softmax(x, dim=-1),
    "rms_norm_forward": ... # manual: x / sqrt(mean(x²) + eps)
    "layer_norm_forward": lambda x: F.layer_norm(x, [x.shape[-1]]),
    # ... norms, losses, cumulative ops
    "mse_loss_forward": lambda p, t: F.mse_loss(p, t),
    "huber_loss_forward": lambda p, t: F.huber_loss(p, t, delta=1.0),
    "cumsum_forward": lambda x: torch.cumsum(x, dim=-1),
    "cumprod_forward": lambda x: torch.cumprod(x, dim=-1),
}
```

The script should:
- Use same input sizes as the Rust binary
- Handle missing PyTorch functions gracefully (skip + note)
- Output CSV to stdout, status messages to stderr

- [ ] **Step 2: Verify the script runs**

Run: `cd examples/kernelbench && python3 bench_pytorch.py > results/pytorch.csv`

Expected: CSV with 86 rows.

- [ ] **Step 3: Commit**

```bash
git add examples/kernelbench/bench_pytorch.py
git commit -m "feat(kernelbench): PyTorch benchmark script for 43 kernel equivalents"
```

---

### Task 7: Comparison Script + Report Generation

**Files:**
- Create: `examples/kernelbench/compare.py`

- [ ] **Step 1: Create comparison script**

`compare.py` reads `results/seguru_cuda.csv` and `results/pytorch.csv`, merges them, and outputs:
1. `results/comparison.csv` — merged data
2. `results/BENCHMARK_REPORT.md` — formatted markdown report
3. Terminal table (printed to stdout)

The script should:
- Join on `(kernel, size_label)` key
- Compute ratios: `seguru/cuda`, `seguru/pytorch`, `cuda/pytorch`
- Group results by category
- Compute per-category and overall averages
- Handle missing data (if a kernel is missing from one CSV, show "N/A")
- Markdown report includes: summary table, per-category tables, analysis

- [ ] **Step 2: Create a runner script**

Create `examples/kernelbench/run_bench.sh`:
```bash
#!/bin/bash
set -e
mkdir -p results

echo "=== Building benchmark binary ==="
cd "$(dirname "$0")/../.." && cargo build -p kernelbench --features bench --bin bench --release
cd "$(dirname "$0")"

echo "=== Running SeGuRu + CUDA benchmark ==="
../../target/release/bench > results/seguru_cuda.csv

echo "=== Running PyTorch benchmark ==="
python3 bench_pytorch.py > results/pytorch.csv

echo "=== Generating comparison report ==="
python3 compare.py
```

- [ ] **Step 3: Commit**

```bash
git add examples/kernelbench/compare.py examples/kernelbench/run_bench.sh
git commit -m "feat(kernelbench): comparison script + runner for benchmark suite"
```

---

### Task 8: Integration — Run Full Benchmark Suite

**Files:**
- Create: `examples/kernelbench/.gitignore` (ignore `results/`)
- Modify: `examples/kernelbench/REPORT.md` (append benchmark results)

- [ ] **Step 1: Add .gitignore for results**

```
results/
```

- [ ] **Step 2: Build and run the full benchmark**

Run: `cd examples/kernelbench && bash run_bench.sh`

Expected: All three phases complete, `results/` contains CSV files and BENCHMARK_REPORT.md.

- [ ] **Step 3: Review results**

Check the output for:
- All 43 kernels have data in both CSVs
- Ratios are reasonable (SeGuRu/CUDA between 0.5x and 5x for most kernels)
- No "FAIL" entries

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/.gitignore examples/kernelbench/run_bench.sh
git commit -m "feat(kernelbench): complete benchmark suite — CUDA + PyTorch comparison"
```
