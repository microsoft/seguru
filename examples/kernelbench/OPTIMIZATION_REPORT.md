# KernelBench Level 1 — SeGuRu Optimization Report

## Performance Progression Summary

| Configuration | Avg vs CUDA (all) | Avg vs CUDA (large) | Kernels ≤1.1× | Faster than CUDA |
|---|---|---|---|---|
| Baseline (original) | 2.02× | 2.02× | 2 | 1 |
| + reshape_map refactor | 2.03× | 2.02× | 2 | 1 |
| + Subslice row traversal | **2.00×** | **2.01×** | 2 | 1 |
| + DISABLE_GPU_BOUND_CHECK | **1.83×** | **1.72×** | 6 | 1 |
| + 4× Loop unrolling | **1.82×** | **1.73×** | 5 | **2** |

## Optimization Techniques Applied

### 1. Subslice Row Traversal (committed)
**Impact: -0.02× avg, -37.5% on matmul_forward**

Replace per-element `a[row * k + i]` with `&a[row*k..(row+1)*k]` subslicing. The subslice does ONE bounds check at creation; subsequent accesses check `i < len` instead of computing row offsets repeatedly.

Key wins:
- `matmul_forward` (large): 3.56× → 2.22× (-37.5%)
- `matmul_batched` (large): 3.77× → 2.74× (-27.4%)

### 2. DISABLE_GPU_BOUND_CHECK=true (build-time flag)
**Impact: -0.17× avg, massive on compute-bound kernels**

Disables all array bounds checks at the codegen level. Every GPU array access normally generates a bounds check; disabling this is the single largest optimization.

Key wins:
- `matmul_transposed_a` (large): 3.88× → 1.41× (-63.6%)
- `matmul_forward` (large): 2.22× → 1.06× (-52.5%)
- `cumsum_forward` (large): 2.18× → 1.07× (-51.7%)
- `cumprod_forward` (large): 2.01× → 1.13× (-43.8%)
- `matvec_forward` (large): 1.28× → 1.03× (-19.4%)

### 3. 4× Loop Unrolling (committed)
**Impact: matmul_transposed_a beats CUDA at 0.94×**

Manually unroll inner K-dimension loop by factor of 4 in all matmul and matvec kernels. Helps the GPU pipeline multiply-add instructions.

Key wins (combined with NoBC):
- `matmul_transposed_a` (large): 1.41× → **0.94×** (FASTER than CUDA!)
- `matmul_forward` (large): 1.06× → 1.03× (near parity)
- `matmul_batched` (large): 2.06× → 1.90× (-7.8%)

### 4. Fast-math LLVM flags (build-time flag)
**Impact: marginal (~0.01×)**

Tested: `--fp-contract=fast --nvptx-prec-divf32=0 --nvptx-prec-sqrtf32=0`

Minimal impact because most kernels don't have divides/sqrts in hot loops. Elementwise kernels with transcendentals (sigmoid, tanh) showed <1% improvement since the transcendental cost dominates.

## Remaining Performance Gap Analysis

### Why SeGuRu is still ~1.8× vs CUDA on average:

1. **Kernel launch overhead** (~5µs SeGuRu vs ~5µs CUDA for small): Similar launch cost, but SeGuRu's overhead is slightly higher due to MLIR→PTX compilation path.

2. **Small-kernel overhead** (all small sizes ~2.0×): Launch overhead dominates. The actual compute is negligible, so overhead is 100% of execution.

3. **Loss/Frobenius kernels** (3.0-3.6×): These use global atomic reduction patterns where each thread does a global atomic add. The SeGuRu atomic wrapper adds overhead vs raw CUDA atomics.

4. **Elementwise kernels** (1.4-1.8× large): Memory-bandwidth bound. SeGuRu generates slightly less optimal load/store patterns. Could benefit from Float4 vectorized loads (4 elements per transaction).

## Build Configurations

```bash
# Default (safety + reasonable perf)
cd examples && cargo build --release --features bench -p kernelbench

# Max performance (no bounds checks)
cd examples && DISABLE_GPU_BOUND_CHECK=true cargo build --release --features bench -p kernelbench

# Max performance + fast math
cd examples && DISABLE_GPU_BOUND_CHECK=true RUSTFLAGS='--cfg seguru -C llvm-args=--fp-contract=fast -C llvm-args=--nvptx-prec-divf32=0' cargo build --release --features bench -p kernelbench
```

## Future Optimization Opportunities

1. **Float4 vectorized loads** for elementwise kernels (est. 20-30% improvement)
2. **Tiled shared-memory matmul** for larger matrices (est. 2-5× improvement for large N)
3. **Warp shuffle reductions** instead of shared memory tree reductions
4. **Register tiling** for matmul (multiple output elements per thread)
