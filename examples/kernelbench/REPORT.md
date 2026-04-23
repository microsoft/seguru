# KernelBench Level 1 → SeGuRu Conversion Report

## Summary

Converted **43 CUDA kernels** from [KernelBench Level 1](https://github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench) into safe Rust GPU kernels compilable with SeGuRu. All kernels pass correctness tests.

| Metric | Value |
|---|---|
| Total kernels implemented | 43 |
| Tests passing | 43/43 (100%) |
| Test execution time | ~13.4s (sequential, single GPU) |
| Categories covered | 10 |
| Kernels excluded | ~57 (convolutions, pooling, transposed conv — not yet supported) |

## Kernel Inventory

### 1. Element-wise Activations (11 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 19 | `relu_forward` | `elementwise.rs` | ✅ Pass |
| 20 | `leaky_relu_forward` | `elementwise.rs` | ✅ Pass |
| 21 | `sigmoid_forward` | `elementwise.rs` | ✅ Pass |
| 22 | `tanh_forward` | `elementwise.rs` | ✅ Pass |
| 26 | `selu_forward` | `elementwise.rs` | ✅ Pass |
| 27 | `hard_sigmoid_forward` | `elementwise.rs` | ✅ Pass |
| 28 | `softplus_forward` | `elementwise.rs` | ✅ Pass |
| 29 | `softsign_forward` | `elementwise.rs` | ✅ Pass |
| 30 | `elu_forward` | `elementwise.rs` | ✅ Pass |
| 31 | `hard_tanh_forward` | `elementwise.rs` | ✅ Pass |
| 25 | `swish_forward` | `elementwise.rs` | ✅ Pass |

### 2. GELU Variants (2 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 23 | `gelu_forward` | `gelu_variants.rs` | ✅ Pass |
| 88 | `mingpt_new_gelu_forward` | `gelu_variants.rs` | ✅ Pass |

### 3. Matrix Multiplication (6 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 1 | `matmul_forward` | `matmul.rs` | ✅ Pass |
| 3 | `matmul_transposed_a_forward` | `matmul.rs` | ✅ Pass |
| 4 | `matmul_transposed_b_forward` | `matmul.rs` | ✅ Pass |
| 5 | `matmul_transposed_both_forward` | `matmul.rs` | ✅ Pass |
| 8 | `matmul_batched_forward` | `matmul.rs` | ✅ Pass |
| 10 | `tensor3d_matmul_forward` | `matmul.rs` | ✅ Pass |

### 4. Matrix-Vector & Scalar (3 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 2 | `matvec_forward` | `matvec.rs` | ✅ Pass |
| 5 | `scalar_multiply_forward` | `matvec.rs` | ✅ Pass |
| 9 | `tensor3d_matmul_forward` | `matvec.rs` | ✅ Pass |

### 5. Reductions (4 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 39 | `sum_reduce_forward` | `reduction.rs` | ✅ Pass |
| 40 | `mean_reduce_forward` | `reduction.rs` | ✅ Pass |
| 41 | `max_reduce_forward` | `reduction.rs` | ✅ Pass |
| 42 | `min_reduce_forward` | `reduction.rs` | ✅ Pass |

### 6. Arg-Reductions (2 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 51 | `argmax_reduce_forward` | `argreduce.rs` | ✅ Pass |
| 52 | `argmin_reduce_forward` | `argreduce.rs` | ✅ Pass |

### 7. Softmax (2 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 24 | `softmax_forward` | `softmax.rs` | ✅ Pass |
| 81 | `log_softmax_forward` | `softmax.rs` | ✅ Pass |

### 8. Normalization (5 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 36 | `rms_norm_forward` | `norm.rs` | ✅ Pass |
| 37 | `frobenius_norm_forward` | `norm.rs` | ✅ Pass |
| 38 | `l1_norm_forward` | `norm.rs` | ✅ Pass |
| 96 | `l2_norm_forward` | `norm.rs` | ✅ Pass |
| 40 | `layer_norm_forward` | `norm.rs` | ✅ Pass |

### 9. Loss Functions (4 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 93 | `mse_loss_forward` | `loss.rs` | ✅ Pass |
| 95 | `huber_loss_forward` | `loss.rs` | ✅ Pass |
| 98 | `kl_div_loss_forward` | `loss.rs` | ✅ Pass |
| 100 | `hinge_loss_forward` | `loss.rs` | ✅ Pass |

### 10. Cumulative Operations (4 kernels)

| KB# | Kernel | Source | Status |
|-----|--------|--------|--------|
| 89 | `cumsum_forward` | `cumulative.rs` | ✅ Pass |
| 90 | `cumprod_forward` | `cumulative.rs` | ✅ Pass |
| 91 | `cumsum_reverse_forward` | `cumulative.rs` | ✅ Pass |
| 92 | `cumsum_exclusive_forward` | `cumulative.rs` | ✅ Pass |

## Test Execution Breakdown

| Test Suite | Tests | Time |
|---|---|---|
| argreduce | 2 | 0.71s |
| cumulative | 4 | 1.25s |
| elementwise | 11 | 3.35s |
| gelu_variants | 2 | 0.67s |
| loss | 4 | 1.26s |
| matmul | 6 | 1.85s |
| matvec | 3 | 0.98s |
| norm | 5 | 1.57s |
| reduction | 4 | 1.26s |
| softmax | 2 | 0.66s |
| **Total** | **43** | **~13.4s** |

## Excluded Kernels (Not Yet Feasible)

The following KernelBench Level 1 categories were excluded because SeGuRu does not yet have built-in support for their underlying operations:

- **Convolutions** (1D, 2D, 3D, depthwise, grouped, dilated, transposed) — ~30 kernels
- **Pooling** (max, avg, adaptive) — ~10 kernels
- **Batch/Instance/Group Normalization** with learned parameters — ~5 kernels
- **Other** (scatter, gather, complex indexing patterns) — ~12 kernels

## Key SeGuRu Patterns Used

| Pattern | Used In |
|---|---|
| `chunk_mut` + `MapContinuousLinear::new(1)` | All element-wise, GELU, matvec, scalar |
| `chunk_mut` + `Map2D::new(n)` | matmul (non-batched) |
| `reshape_map!` for multi-element output | softmax, norm, cumulative |
| Shared memory tree reduction | reduction, argreduce, norm, loss |
| Dynamic shared memory (`smem_alloc`) | reduction, argreduce, softmax, norm, loss |
| Atomics (`Atomic::new`) | — (not needed for Level 1) |
| `u32` indices throughout | All kernels |

## Notes

- All GPU thread indices use `u32` as requested (cast to `usize` only for slice indexing)
- Tolerances: 1e-4 (elementwise), 1e-2 (matmul), 1e-3 (reductions/norms/loss)
- Tests use deterministic data patterns for reproducibility
- Kernel function names avoid matching the crate name (MLIR mangling bug)
- `gpu_config!` is recreated before each launch call (non-Copy type)
