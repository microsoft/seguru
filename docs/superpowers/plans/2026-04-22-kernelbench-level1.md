# KernelBench Level 1 — SeGuRu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert ~52 feasible KernelBench Level 1 CUDA kernels from PyTorch definitions to SeGuRu Rust GPU kernels with correctness tests.

**Architecture:** A single `kernelbench` crate under `examples/` with modules per kernel category. Each kernel is a `#[gpu::cuda_kernel]` function with a corresponding `#[test]` that validates correctness against a CPU reference implementation. A final report generator summarizes results.

**Tech Stack:** SeGuRu (`gpu`, `gpu_host` crates), Rust nightly, CUDA 12.8+

---

## File Structure

```
examples/kernelbench/
  Cargo.toml
  src/
    lib.rs              # Re-exports all modules
    elementwise.rs      # 13 simple activation kernels (ReLU through HardTanh)
    gelu_variants.rs    # GELU, MinGPTNewGelu (need math intrinsics)
    softmax.rs          # Softmax, LogSoftmax (shared memory reduction per row)
    matmul.rs           # 10 matmul variants (square, rect, transposed, batched)
    matvec.rs           # Mat-vec and scalar multiply (3 kernels)
    reduction.rs        # Sum, Mean, Max, Min reductions (4 kernels)
    argreduce.rs        # Argmax, Argmin (2 kernels)
    norm.rs             # RMSNorm, L1Norm, L2Norm, FrobeniusNorm, LayerNorm (5 kernels)
    loss.rs             # MSE, Huber, Hinge, KLDiv (4 kernels)
    cumulative.rs       # Cumsum, Cumprod, reverse/exclusive cumsum (4 kernels)
    report.rs           # Report generation helpers
```

## Kernel-to-File Mapping

| KernelBench # | Name | Module | Kernel fn |
|---|---|---|---|
| 19 | ReLU | elementwise | `relu_forward` |
| 20 | LeakyReLU | elementwise | `leaky_relu_forward` |
| 21 | Sigmoid | elementwise | `sigmoid_forward` |
| 22 | Tanh | elementwise | `tanh_forward` |
| 25 | Swish | elementwise | `swish_forward` |
| 27 | SELU | elementwise | `selu_forward` |
| 28 | HardSigmoid | elementwise | `hard_sigmoid_forward` |
| 29 | Softplus | elementwise | `softplus_forward` |
| 30 | Softsign | elementwise | `softsign_forward` |
| 31 | ELU | elementwise | `elu_forward` |
| 32 | HardTanh | elementwise | `hardtanh_forward` |
| 26 | GELU | gelu_variants | `gelu_forward` |
| 88 | MinGPTNewGelu | gelu_variants | `mingpt_gelu_forward` |
| 23 | Softmax | softmax | `softmax_forward` |
| 24 | LogSoftmax | softmax | `log_softmax_forward` |
| 1 | Square matmul | matmul | `matmul_square` |
| 2 | Standard matmul | matmul | `matmul_standard` |
| 3 | Batched matmul | matmul | `matmul_batched` |
| 6 | Large K matmul | matmul | `matmul_large_k` |
| 7 | Small K matmul | matmul | `matmul_small_k` |
| 8 | Irregular matmul | matmul | `matmul_irregular` |
| 9 | Tall-skinny matmul | matmul | `matmul_tall_skinny` |
| 16 | Transposed A matmul | matmul | `matmul_transposed_a` |
| 17 | Transposed B matmul | matmul | `matmul_transposed_b` |
| 18 | Transposed both matmul | matmul | `matmul_transposed_both` |
| 4 | Mat-vec multiply | matvec | `matvec_forward` |
| 5 | Scalar multiply | matvec | `scalar_multiply` |
| 10 | 3D tensor matmul | matvec | `tensor3d_matmul` |
| 47 | Sum reduction | reduction | `sum_reduce` |
| 48 | Mean reduction | reduction | `mean_reduce` |
| 49 | Max reduction | reduction | `max_reduce` |
| 53 | Min reduction | reduction | `min_reduce` |
| 51 | Argmax | argreduce | `argmax_reduce` |
| 52 | Argmin | argreduce | `argmin_reduce` |
| 36 | RMSNorm | norm | `rms_norm_forward` |
| 37 | FrobeniusNorm | norm | `frobenius_norm_forward` |
| 38 | L1Norm | norm | `l1_norm_forward` |
| 39 | L2Norm | norm | `l2_norm_forward` |
| 40 | LayerNorm | norm | `layer_norm_forward` |
| 94 | MSELoss | loss | `mse_loss_forward` |
| 96 | HuberLoss | loss | `huber_loss_forward` |
| 100 | HingeLoss | loss | `hinge_loss_forward` |
| 98 | KLDivLoss | loss | `kl_div_loss_forward` |
| 89 | Cumsum | cumulative | `cumsum_forward` |
| 90 | Cumprod | cumulative | `cumprod_forward` |
| 91 | Cumsum reverse | cumulative | `cumsum_reverse_forward` |
| 92 | Cumsum exclusive | cumulative | `cumsum_exclusive_forward` |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `examples/kernelbench/Cargo.toml`
- Create: `examples/kernelbench/src/lib.rs`
- Modify: `examples/Cargo.toml` — add `kernelbench` to workspace members

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "kernelbench"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = { workspace = true }
gpu_host = { workspace = true }

[dev-dependencies]
rand = "0.9"
```

- [ ] **Step 2: Create src/lib.rs**

```rust
#![no_std]
#![allow(clippy::too_many_arguments)]
#![deny(clippy::cast_possible_truncation)]

pub mod elementwise;
pub mod gelu_variants;
// Additional modules will be added as implemented
```

- [ ] **Step 3: Add to workspace**

In `examples/Cargo.toml`, add `"kernelbench"` to the `members` array.

- [ ] **Step 4: Create stub module files**

Create empty files for each module: `elementwise.rs`, `gelu_variants.rs`, `softmax.rs`, `matmul.rs`, `matvec.rs`, `reduction.rs`, `argreduce.rs`, `norm.rs`, `loss.rs`, `cumulative.rs`, `report.rs`.

- [ ] **Step 5: Verify it compiles**

Run: `cd examples && cargo build -p kernelbench`
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add examples/kernelbench examples/Cargo.toml
git commit -m "feat(kernelbench): scaffold crate for KernelBench Level 1 kernels"
```

---

## Task 2: Element-wise Activation Kernels (11 kernels)

**Files:**
- Create: `examples/kernelbench/src/elementwise.rs`

All element-wise kernels share the same pattern: 1 thread per element, read input, apply activation, write output via `chunk_mut`. Test sizes use small dimensions (e.g., 1024 elements) for correctness.

- [ ] **Step 1: Implement all element-wise kernels**

Write `elementwise.rs` with these kernels:

```rust
use gpu::prelude::*;

// KB#19: ReLU — max(0, x)
#[gpu::cuda_kernel]
pub fn relu_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = if x > 0.0 { x } else { 0.0 };
    }
}

// KB#20: LeakyReLU — x if x > 0, else alpha * x
#[gpu::cuda_kernel]
pub fn leaky_relu_forward(input: &[f32], output: &mut [f32], alpha: f32) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = if x > 0.0 { x } else { alpha * x };
    }
}

// KB#21: Sigmoid — 1 / (1 + exp(-x))
#[gpu::cuda_kernel]
pub fn sigmoid_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = 1.0 / (1.0 + (-x).exp());
    }
}

// KB#22: Tanh
#[gpu::cuda_kernel]
pub fn tanh_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        out[0] = input[tid].tanh();
    }
}

// KB#25: Swish — x * sigmoid(x)
#[gpu::cuda_kernel]
pub fn swish_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = x / (1.0 + (-x).exp());
    }
}

// KB#27: SELU — scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
#[gpu::cuda_kernel]
pub fn selu_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        let alpha: f32 = 1.6732632;
        let scale: f32 = 1.0507010;
        out[0] = if x > 0.0 {
            scale * x
        } else {
            scale * alpha * (x.exp() - 1.0)
        };
    }
}

// KB#28: HardSigmoid — clamp((x + 3) / 6, 0, 1)
#[gpu::cuda_kernel]
pub fn hard_sigmoid_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        let v = (x + 3.0) / 6.0;
        out[0] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
    }
}

// KB#29: Softplus — log(1 + exp(x))
#[gpu::cuda_kernel]
pub fn softplus_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = (1.0 + x.exp()).log();
    }
}

// KB#30: Softsign — x / (1 + |x|)
#[gpu::cuda_kernel]
pub fn softsign_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        let abs_x = if x < 0.0 { -x } else { x };
        out[0] = x / (1.0 + abs_x);
    }
}

// KB#31: ELU — x if x > 0, else alpha * (exp(x) - 1)
#[gpu::cuda_kernel]
pub fn elu_forward(input: &[f32], output: &mut [f32], alpha: f32) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
    }
}

// KB#32: HardTanh — clamp(x, min_val, max_val)
#[gpu::cuda_kernel]
pub fn hardtanh_forward(input: &[f32], output: &mut [f32], min_val: f32, max_val: f32) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        out[0] = if x < min_val {
            min_val
        } else if x > max_val {
            max_val
        } else {
            x
        };
    }
}
```

- [ ] **Step 2: Write tests for all element-wise kernels**

Add `#[cfg(test)]` module at the bottom of `elementwise.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn run_elementwise_test(
        input: &[f32],
        expected: &[f32],
        launch_fn: impl FnOnce(
            &gpu_host::GpuCtxGuard,
            &gpu_host::GpuModule,
            &gpu_host::TensorView<[f32]>,
            &mut gpu_host::TensorViewMut<[f32]>,
        ),
    ) {
        let n = input.len();
        let mut output = vec![0.0f32; n];
        gpu_host::cuda_ctx(0, |ctx, m| {
            let d_input = ctx.new_tensor_view(input).unwrap();
            let mut d_output = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
            launch_fn(ctx, m, &d_input, &mut d_output);
            d_output.copy_to_host(&mut output).unwrap();
        });
        for i in 0..n {
            assert!(
                (output[i] - expected[i]).abs() < 1e-4,
                "mismatch at [{}]: got {} expected {}",
                i,
                output[i],
                expected[i]
            );
        }
    }

    fn grid_block_for(n: usize) -> (u32, u32) {
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        (grid, block)
    }

    #[test]
    fn test_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 3.0];
        let expected: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let (grid, block) = grid_block_for(input.len());
        run_elementwise_test(&input, &expected, |ctx, m, d_in, d_out| {
            let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
            relu_forward::launch(config, ctx, m, d_in, d_out).unwrap();
        });
    }

    #[test]
    fn test_sigmoid() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -5.0, 5.0, 0.5];
        let expected: Vec<f32> = input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let (grid, block) = grid_block_for(input.len());
        run_elementwise_test(&input, &expected, |ctx, m, d_in, d_out| {
            let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
            sigmoid_forward::launch(config, ctx, m, d_in, d_out).unwrap();
        });
    }

    #[test]
    fn test_tanh() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        let (grid, block) = grid_block_for(input.len());
        run_elementwise_test(&input, &expected, |ctx, m, d_in, d_out| {
            let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
            tanh_forward::launch(config, ctx, m, d_in, d_out).unwrap();
        });
    }

    // ... similar test functions for each kernel
    // Tests for leaky_relu, swish, selu, hard_sigmoid, softplus, softsign, elu, hardtanh
}
```

- [ ] **Step 3: Build and run tests**

Run: `cd examples && cargo test -p kernelbench -- elementwise`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/src/elementwise.rs
git commit -m "feat(kernelbench): element-wise activation kernels (ReLU, Sigmoid, Tanh, etc.)"
```

---

## Task 3: GELU Variants (2 kernels)

**Files:**
- Create: `examples/kernelbench/src/gelu_variants.rs`
- Modify: `examples/kernelbench/src/lib.rs` — add `pub mod gelu_variants;`

- [ ] **Step 1: Implement GELU and MinGPTNewGelu**

```rust
use gpu::prelude::*;

// KB#26: GELU — 0.5 * x * (1 + erf(x / sqrt(2)))
// Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[gpu::cuda_kernel]
pub fn gelu_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        let k: f32 = 0.7978845; // sqrt(2/pi)
        out[0] = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
    }
}

// KB#88: MinGPT NewGELU (same formula, separate kernel for benchmark identity)
#[gpu::cuda_kernel]
pub fn mingpt_gelu_forward(input: &[f32], output: &mut [f32]) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        let x = input[tid];
        let k: f32 = 0.7978845;
        out[0] = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
    }
}
```

- [ ] **Step 2: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gelu() {
        // CPU reference
        let input = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| {
                let k: f32 = 0.7978845;
                0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        let mut output = vec![0.0f32; input.len()];
        let n = input.len() as u32;
        gpu_host::cuda_ctx(0, |ctx, m| {
            let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
            let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
            let config = gpu_host::gpu_config!(1, 1, 1, n, 1, 1, 0);
            gelu_forward::launch(config, ctx, m, &d_in, &mut d_out).unwrap();
            d_out.copy_to_host(&mut output).unwrap();
        });
        for i in 0..input.len() {
            assert!((output[i] - expected[i]).abs() < 1e-4);
        }
    }
}
```

- [ ] **Step 3: Build and test**

Run: `cd examples && cargo test -p kernelbench -- gelu`
Expected: Pass.

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/src/gelu_variants.rs examples/kernelbench/src/lib.rs
git commit -m "feat(kernelbench): GELU and MinGPT NewGELU kernels"
```

---

## Task 4: Matrix Multiplications (10 kernels)

**Files:**
- Create: `examples/kernelbench/src/matmul.rs`
- Modify: `examples/kernelbench/src/lib.rs` — add `pub mod matmul;`

All matmul kernels use a tiled approach with 2D thread blocks. Each variant adjusts indexing for the specific operation (transposed, batched, etc.).

- [ ] **Step 1: Implement core matmul and variants**

Base matmul pattern (KB#1, KB#2, KB#6, KB#7, KB#8, KB#9 are same kernel with different test sizes):

```rust
use gpu::prelude::*;

// Core C = A(M×K) * B(K×N) → C(M×N)
// Used for KB#1 (square), KB#2 (standard), KB#6 (large K), KB#7 (small K), KB#8 (irregular), KB#9 (tall-skinny)
#[gpu::cuda_kernel]
pub fn matmul_forward(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if row < m && col < n {
        let mut sum = 0.0f32;
        let a_row = &a[row * k..row * k + k];
        let mut b_idx = col;
        for a_val in a_row {
            sum += a_val * b[b_idx];
            b_idx += n;
        }
        c[(0, 0)] = sum;
    }
}

// KB#16: C = A^T(K×M)^T * B(K×N) → C(M×N), A stored as K×M
#[gpu::cuda_kernel]
pub fn matmul_transposed_a(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if row < m && col < n {
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        while idx < k {
            sum += a[idx * m + row] * b[idx * n + col];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#17: C = A(M×K) * B^T(N×K)^T → C(M×N), B stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if row < m && col < n {
        let mut sum = 0.0f32;
        let a_row = &a[row * k..row * k + k];
        let b_row = &b[col * k..col * k + k];
        let mut idx = 0usize;
        while idx < k {
            sum += a_row[idx] * b_row[idx];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#18: C = A^T * B^T, A stored as K×M, B stored as N×K
#[gpu::cuda_kernel]
pub fn matmul_transposed_both(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if row < m && col < n {
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        while idx < k {
            sum += a[idx * m + row] * b[col * k + idx];
            idx += 1;
        }
        c[(0, 0)] = sum;
    }
}

// KB#3: Batched matmul — batch of independent matmuls
// A(batch×M×K) * B(batch×K×N) → C(batch×M×N)
#[gpu::cuda_kernel]
pub fn matmul_batched(
    a: &[f32], b: &[f32], c: &mut [f32],
    batch: usize, m: usize, n: usize, k: usize,
) {
    let mut c = chunk_mut(c, MapLinear::new(1));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let bid_z = block_id::<DimZ>() as usize;
    if bid_z < batch && row < m && col < n {
        let a_offset = bid_z * m * k;
        let b_offset = bid_z * k * n;
        let c_offset = bid_z * m * n;
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        while idx < k {
            sum += a[a_offset + row * k + idx] * b[b_offset + idx * n + col];
            idx += 1;
        }
        // Write to global offset directly
        let global_idx = c_offset + row * n + col;
        if global_idx < c.len() {
            c[0] = sum;
        }
    }
}
```

Note: `matmul_batched` uses DimZ for batch index. The chunk_mut approach may need adjustment — the exact indexing depends on how SeGuRu maps the 3D grid to linear output. If `Map2D` doesn't support batch indexing, fall back to writing via a single-element chunk with computed global offset. This may need iteration during implementation.

- [ ] **Step 2: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            assert!(
                (actual[i] - expected[i]).abs() < tol,
                "mismatch at [{}]: got {} expected {}",
                i, actual[i], expected[i]
            );
        }
    }

    #[test]
    fn test_matmul_square() {
        let n = 64usize;
        let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
        let expected = cpu_matmul(&a, &b, n, n, n);
        let mut output = vec![0.0f32; n * n];
        gpu_host::cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
            let block = 16u32;
            let grid = ((n as u32) + block - 1) / block;
            let config = gpu_host::gpu_config!(grid, grid, 1, block, block, 1, 0);
            matmul_forward::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n, n, n).unwrap();
            d_c.copy_to_host(&mut output).unwrap();
        });
        assert_close(&output, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_transposed_a() {
        let m = 32usize;
        let n = 48usize;
        let k = 16usize;
        // A stored as K×M (transposed)
        let a_t: Vec<f32> = (0..k * m).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();
        // CPU reference: transpose A_T to get A(M×K), then multiply
        let mut a = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                a[j * k + i] = a_t[i * m + j];
            }
        }
        let expected = cpu_matmul(&a, &b, m, n, k);
        let mut output = vec![0.0f32; m * n];
        gpu_host::cuda_ctx(0, |ctx, module| {
            let d_a = ctx.new_tensor_view(a_t.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
            let block = 16u32;
            let grid_x = ((n as u32) + block - 1) / block;
            let grid_y = ((m as u32) + block - 1) / block;
            let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block, block, 1, 0);
            matmul_transposed_a::launch(config, ctx, module, &d_a, &d_b, &mut d_c, m, n, k).unwrap();
            d_c.copy_to_host(&mut output).unwrap();
        });
        assert_close(&output, &expected, 1e-2);
    }

    // Similar tests for transposed_b, transposed_both, batched
}
```

- [ ] **Step 3: Build and test**

Run: `cd examples && cargo test -p kernelbench -- matmul`

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/src/matmul.rs examples/kernelbench/src/lib.rs
git commit -m "feat(kernelbench): matrix multiplication kernels (square, transposed, batched)"
```

---

## Task 5: Mat-Vec, Scalar Multiply, 3D Tensor Matmul (3 kernels)

**Files:**
- Create: `examples/kernelbench/src/matvec.rs`

- [ ] **Step 1: Implement matvec, scalar multiply, 3D tensor matmul**

```rust
use gpu::prelude::*;

// KB#4: y = A(M×N) * x(N) → y(M)
#[gpu::cuda_kernel]
pub fn matvec_forward(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(y, MapLinear::new(1));
    if tid < m {
        let mut sum = 0.0f32;
        let row = &a[tid * n..tid * n + n];
        let mut idx = 0usize;
        while idx < n {
            sum += row[idx] * x[idx];
            idx += 1;
        }
        out[0] = sum;
    }
}

// KB#5: C = A * s (element-wise scalar multiply)
#[gpu::cuda_kernel]
pub fn scalar_multiply(input: &[f32], output: &mut [f32], s: f32) {
    let tid = global_id::<DimX>() as usize;
    let mut out = chunk_mut(output, MapLinear::new(1));
    if tid < input.len() {
        out[0] = input[tid] * s;
    }
}

// KB#10: 3D tensor matmul — A(B×M×K) * B_tensor(B×K×N) → C(B×M×N)
// Same as batched matmul, reuse pattern
#[gpu::cuda_kernel]
pub fn tensor3d_matmul(
    a: &[f32], b: &[f32], c: &mut [f32],
    batch: usize, m: usize, n: usize, k: usize,
) {
    let mut c_chunk = chunk_mut(c, MapLinear::new(1));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let bid_z = block_id::<DimZ>() as usize;
    if bid_z < batch && row < m && col < n {
        let a_off = bid_z * m * k;
        let b_off = bid_z * k * n;
        let mut sum = 0.0f32;
        let mut idx = 0usize;
        while idx < k {
            sum += a[a_off + row * k + idx] * b[b_off + idx * n + col];
            idx += 1;
        }
        c_chunk[0] = sum;
    }
}
```

- [ ] **Step 2: Write tests and verify**
- [ ] **Step 3: Commit**

---

## Task 6: Reduction Kernels (4 kernels: sum, mean, max, min)

**Files:**
- Create: `examples/kernelbench/src/reduction.rs`

These operate on 2D input (batch_size × dim), reducing along dim 1 to produce (batch_size × 1). Each row is handled by one block using shared memory reduction.

- [ ] **Step 1: Implement reduction kernels**

```rust
use gpu::prelude::*;

// KB#47: Sum reduction over dim=1
// Input: (batch, dim), Output: (batch,)
// One block per row, threads cooperatively reduce within block using shared memory
#[gpu::cuda_kernel(dynamic_shared)]
pub fn sum_reduce(input: &[f32], output: &mut [f32], dim: usize) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    // Each thread accumulates partial sum across the row
    let row_start = bid as usize * dim;
    let mut local_sum = 0.0f32;
    let mut idx = tid as usize;
    while idx < dim {
        local_sum += input[row_start + idx];
        idx += bdim as usize;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_sum;
    sync_threads();

    // Tree reduction in shared memory
    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(output, MapLinear::new(1));
        out[0] = *smem[0];
    }
}

// KB#49: Max reduction over dim=1
#[gpu::cuda_kernel(dynamic_shared)]
pub fn max_reduce(input: &[f32], output: &mut [f32], dim: usize) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid as usize * dim;
    let mut local_max = f32::NEG_INFINITY;
    let mut idx = tid as usize;
    while idx < dim {
        let v = input[row_start + idx];
        if v > local_max { local_max = v; }
        idx += bdim as usize;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_max;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            if right > left { sc[0] = right; }
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(output, MapLinear::new(1));
        out[0] = *smem[0];
    }
}

// KB#53: Min reduction — same as max but with min comparison
#[gpu::cuda_kernel(dynamic_shared)]
pub fn min_reduce(input: &[f32], output: &mut [f32], dim: usize) {
    // Same structure as max_reduce but with min comparisons
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid as usize * dim;
    let mut local_min = f32::INFINITY;
    let mut idx = tid as usize;
    while idx < dim {
        let v = input[row_start + idx];
        if v < local_min { local_min = v; }
        idx += bdim as usize;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_min;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            if right < left { sc[0] = right; }
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(output, MapLinear::new(1));
        out[0] = *smem[0];
    }
}

// KB#48: Mean reduction = sum / dim
#[gpu::cuda_kernel(dynamic_shared)]
pub fn mean_reduce(input: &[f32], output: &mut [f32], dim: usize) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);

    let row_start = bid as usize * dim;
    let mut local_sum = 0.0f32;
    let mut idx = tid as usize;
    while idx < dim {
        local_sum += input[row_start + idx];
        idx += bdim as usize;
    }

    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_sum;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let mut out = chunk_mut(output, MapLinear::new(1));
        out[0] = *smem[0] / dim as f32;
    }
}
```

- [ ] **Step 2: Write tests for each reduction**

Test pattern: create a small 2D array (e.g., 4 rows × 128 cols), compute CPU reference, compare.

- [ ] **Step 3: Build and test**

Run: `cd examples && cargo test -p kernelbench -- reduction`

- [ ] **Step 4: Commit**

```bash
git add examples/kernelbench/src/reduction.rs examples/kernelbench/src/lib.rs
git commit -m "feat(kernelbench): reduction kernels (sum, mean, max, min)"
```

---

## Task 7: Argmax/Argmin (2 kernels)

**Files:**
- Create: `examples/kernelbench/src/argreduce.rs`

Returns index of max/min element per row. Needs shared memory for both value and index.

- [ ] **Step 1: Implement argmax and argmin**

Uses two shared memory arrays (values + indices), tree reduction comparing values and keeping indices.

- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

---

## Task 8: Softmax and LogSoftmax (2 kernels)

**Files:**
- Create: `examples/kernelbench/src/softmax.rs`

Softmax requires: (1) find max per row, (2) compute exp(x - max) per element, (3) sum those, (4) divide. This needs 3 passes over shared memory.

- [ ] **Step 1: Implement softmax_forward and log_softmax_forward**

```rust
use gpu::prelude::*;

// KB#23: Softmax over dim=1
// Input: (batch, dim), Output: (batch, dim)
// One block per row
#[gpu::cuda_kernel(dynamic_shared)]
pub fn softmax_forward(input: &[f32], output: &mut [f32], dim: usize) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let row_start = bid as usize * dim;

    // Step 1: Find max
    let mut local_max = f32::NEG_INFINITY;
    let mut idx = tid as usize;
    while idx < dim {
        let v = input[row_start + idx];
        if v > local_max { local_max = v; }
        idx += bdim as usize;
    }
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_max;
    sync_threads();

    let mut stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            if right > left { sc[0] = right; }
        }
        sync_threads();
        stride /= 2;
    }
    let row_max = *smem[0];
    sync_threads();

    // Step 2: Compute exp(x - max) and sum
    let mut local_sum = 0.0f32;
    idx = tid as usize;
    while idx < dim {
        local_sum += (input[row_start + idx] - row_max).exp();
        idx += bdim as usize;
    }
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    smem_chunk[0] = local_sum;
    sync_threads();

    stride = bdim / 2;
    while stride > 0 {
        if tid < stride {
            let mut sc = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
            let left = sc[0];
            let right = sc[1];
            sc[0] = left + right;
        }
        sync_threads();
        stride /= 2;
    }
    let row_sum = *smem[0];
    sync_threads();

    // Step 3: Write output
    idx = tid as usize;
    while idx < dim {
        let mut out = chunk_mut(output, MapLinear::new(1));
        // Need to position output write at correct global index
        // This may need adjustment based on SeGuRu's chunk mapping for strided writes
        out[0] = (input[row_start + idx] - row_max).exp() / row_sum;
        idx += bdim as usize;
    }
}
```

Note: The output write in step 3 writes one element per thread per iteration. The `chunk_mut` with `MapLinear::new(1)` gives each thread one element. For strided writes where each thread writes multiple elements, iteration with repositioned chunks may be needed. This will need testing and may require using `reshape_map!` or a different approach.

- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

---

## Task 9: Normalization Kernels (5 kernels)

**Files:**
- Create: `examples/kernelbench/src/norm.rs`

Norms follow a reduction-then-normalize pattern:
1. Compute a statistic per row (L2 norm, mean, variance, etc.)
2. Normalize each element by the statistic

- [ ] **Step 1: Implement RMSNorm, L1Norm, L2Norm, FrobeniusNorm, LayerNorm**

Each uses shared memory reduction for the statistic, then a second pass to normalize.

- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

---

## Task 10: Loss Function Kernels (4 kernels)

**Files:**
- Create: `examples/kernelbench/src/loss.rs`

Losses compute element-wise differences, then reduce to a scalar.

- [ ] **Step 1: Implement MSE, Huber, Hinge, KLDiv loss**

Pattern: element-wise operation + sum reduction + divide by N.

- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

---

## Task 11: Cumulative Operations (4 kernels)

**Files:**
- Create: `examples/kernelbench/src/cumulative.rs`

Prefix scan (cumsum, cumprod) per row. Use a simple sequential scan within each block (one block per row, single thread per row for small dims, or Blelloch scan for larger).

- [ ] **Step 1: Implement cumsum, cumprod, cumsum_reverse, cumsum_exclusive**
- [ ] **Step 2: Write tests**
- [ ] **Step 3: Commit**

---

## Task 12: Report Generation & Final Verification

**Files:**
- Create: `examples/kernelbench/src/report.rs`

- [ ] **Step 1: Create report module**

The report will be generated by running all tests with timing. Add a test that collects timing info:

```rust
// In report.rs — a test that summarizes all kernel results
#[cfg(test)]
mod tests {
    #[test]
    fn generate_report() {
        // This test prints a summary table
        println!("\n=== KernelBench Level 1 — SeGuRu Report ===");
        println!("{:<5} {:<30} {:<10} {:<15}", "KB#", "Kernel", "Status", "Notes");
        println!("{}", "-".repeat(60));
        // Each entry is manually listed with pass/fail status
        // The actual pass/fail is determined by whether the individual tests pass
    }
}
```

- [ ] **Step 2: Run full test suite**

Run: `cd examples && cargo test -p kernelbench -- --test-threads=1 2>&1 | tee kernelbench_results.txt`

- [ ] **Step 3: Commit everything**

```bash
git add examples/kernelbench/
git commit -m "feat(kernelbench): complete KernelBench Level 1 implementation with tests and report"
```

---

## Implementation Notes

**SeGuRu-specific constraints to watch for:**
1. `chunk_mut` uses LOCAL indices — `c[0]` is the thread's element, not global index 0
2. No closures inside `#[gpu::cuda_kernel]` functions
3. Block dim product ≤ 1024
4. `gpu_config!` is non-Copy — recreate before each launch in loops
5. Kernel function names must not match crate name (`kernelbench`)
6. Use `f32::NEG_INFINITY` / `f32::INFINITY` for reduction init values — verify these are available in no_std context (may need `core::f32::NEG_INFINITY`)

**Test sizes:** Use small dimensions (64–1024 elements) for correctness verification, not the large KernelBench sizes which are for benchmarking.

**Tolerance:** Use 1e-4 for element-wise ops, 1e-2 for matmul (accumulated floating point error), 1e-3 for reductions/norms.
