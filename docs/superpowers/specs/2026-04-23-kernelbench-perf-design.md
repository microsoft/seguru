# KernelBench Performance Benchmark Design

## Problem

We have 43 SeGuRu GPU kernels (KernelBench Level 1) with correctness tests but no performance data. We need to compare their execution speed against equivalent CUDA C kernels and PyTorch operations.

## Approach

All-in-one Rust binary with CUDA FFI for kernel-level comparison, plus a Python script for PyTorch comparison. Results merged into CSV + Markdown report + terminal table.

## Architecture

```
examples/kernelbench/
├── cuda/
│   ├── kernels.cu          # 43 CUDA C reference kernel implementations
│   └── bench_runner.cu     # C API: each kernel wrapped with CUDA event timing
├── src/
│   ├── lib.rs              # Existing 43 SeGuRu kernels (unchanged)
│   └── bin/
│       └── bench.rs        # Rust binary: runs SeGuRu + CUDA kernels, outputs CSV
├── build.rs                # Compiles cuda/*.cu via nvcc → libkernelbench_cuda.a
├── bench_pytorch.py        # PyTorch comparison → pytorch_results.csv
├── compare.py              # Merge CSVs → final report
└── results/                # Output directory (gitignored)
    ├── seguru_cuda.csv
    ├── pytorch.csv
    ├── comparison.csv
    └── BENCHMARK_REPORT.md
```

## Components

### 1. CUDA Reference Kernels (`cuda/kernels.cu`)

All 43 kernels implemented as standard CUDA `__global__` functions. Naming convention: `cuda_{kernel_name}` (e.g., `cuda_relu_forward`).

**Kernel list by category:**

| Category | Kernels | CUDA Pattern |
|---|---|---|
| Elementwise (11) | relu, leaky_relu, sigmoid, tanh, swish, selu, hard_sigmoid, softplus, softsign, elu, hard_tanh | 1 thread per element, trivial |
| GELU (2) | gelu, mingpt_new_gelu | 1 thread per element with math |
| Matmul (6) | matmul, transposed_a/b/both, batched, tensor3d | Naive nested loop (no tiling) |
| Matvec (3) | matvec, scalar_multiply, tensor3d_matmul | Row per thread / element per thread |
| Reduction (4) | sum, mean, max, min | Shared memory tree reduction |
| Argreduce (2) | argmax, argmin | Dual shared memory (val + idx) |
| Softmax (2) | softmax, log_softmax | 3-pass: max → sum_exp → normalize |
| Norm (5) | rms, frobenius, l1, l2, layer | Shared memory reduction |
| Loss (4) | mse, huber, kl_div, hinge | Reduction to scalar |
| Cumulative (4) | cumsum, cumprod, cumsum_reverse, cumsum_exclusive | 1 thread per row sequential scan |

**Important**: CUDA kernels should match the same algorithmic approach as the SeGuRu versions (naive, no advanced tiling/warp-level optimizations) for fair comparison. The goal is measuring SeGuRu compiler overhead, not algorithm differences.

### 2. CUDA Bench Runner (`cuda/bench_runner.cu`)

Exposes a C API for each kernel:

```c
// For each kernel, a function that:
// 1. Allocates device memory
// 2. Copies input to device
// 3. Warmup runs (3)
// 4. Times N iterations with CUDA events
// 5. Returns median time in microseconds
// 6. Frees device memory

extern "C" float bench_cuda_relu_forward(
    const float* h_input, float* h_output,
    int n, int grid, int block, int warmup, int iters
);
```

Each bench function returns the median kernel execution time in microseconds (excludes memory transfer, includes only kernel dispatch + execution).

### 3. Build Script (`build.rs`)

```rust
// Compile cuda/*.cu with nvcc
// Link as static library: libkernelbench_cuda.a
// Pass CUDA include/lib paths
// Feature-gate: only build when "bench" feature is enabled
```

Uses `cc` crate or direct `nvcc` invocation. Detects CUDA path from `CUDA_HOME` or `/usr/local/cuda`.

### 4. Rust Benchmark Binary (`src/bin/bench.rs`)

Single binary that:
1. Initializes CUDA context via `gpu_host::cuda_ctx`
2. For each kernel × each size (small, large):
   a. Prepares input data (deterministic, same as CUDA side)
   b. Runs SeGuRu kernel: warmup 3 times, measure 10 iterations, record median
   c. Runs CUDA kernel (via FFI): same warmup/iterations, get median from C side
3. Outputs CSV to stdout (can redirect to file)

**Timing for SeGuRu kernels:**
```rust
// Warmup
for _ in 0..warmup {
    kernel::launch(config, ctx, m, &input, &mut output).unwrap();
    ctx.sync().unwrap();
}

// Measure
let mut times = Vec::new();
for _ in 0..iters {
    let start = std::time::Instant::now();
    kernel::launch(config, ctx, m, &input, &mut output).unwrap();
    ctx.sync().unwrap();
    times.push(start.elapsed().as_micros() as f64);
}
times.sort_by(|a, b| a.partial_cmp(b).unwrap());
let median = times[iters / 2];
```

**CSV output format:**
```csv
kernel,category,size_label,n_elements,grid,block,seguru_us,cuda_us,ratio
relu_forward,elementwise,small,4096,16,256,12.3,11.8,1.04
relu_forward,elementwise,large,1048576,4096,256,45.2,43.1,1.05
...
```

`ratio` = seguru_us / cuda_us (>1 means SeGuRu is slower, <1 means faster).

### 5. PyTorch Benchmark (`bench_pytorch.py`)

```python
import torch
import torch.nn.functional as F
import csv, time

# For each kernel equivalent:
# 1. Create input tensors on GPU
# 2. Warmup (3 runs + torch.cuda.synchronize)
# 3. Measure N iterations with torch.cuda.Event
# 4. Record median time

# Output: pytorch_results.csv
# kernel,size_label,n_elements,pytorch_us
```

PyTorch operations mapped to SeGuRu kernels:
- `relu_forward` → `F.relu(x)`
- `matmul_forward` → `torch.matmul(a, b)`
- `softmax_forward` → `F.softmax(x, dim=-1)`
- `sum_reduce_forward` → `torch.sum(x, dim=-1)`
- etc.

### 6. Comparison Script (`compare.py`)

Merges `seguru_cuda.csv` + `pytorch.csv` into:
1. `comparison.csv` — full merged data
2. `BENCHMARK_REPORT.md` — formatted markdown with tables
3. Terminal output — colored table summary

**Markdown report structure:**
```markdown
# KernelBench Level 1 Performance Report

## Summary
| Metric | Value |
| Total kernels | 43 |
| SeGuRu avg overhead vs CUDA | X.XX× |
| SeGuRu avg overhead vs PyTorch | X.XX× |

## Results by Category
### Elementwise (11 kernels)
| Kernel | Size | SeGuRu (μs) | CUDA (μs) | PyTorch (μs) | SeGuRu/CUDA | SeGuRu/PyTorch |
...

## Analysis
- Categories where SeGuRu is competitive
- Categories with highest overhead
- Impact of input size on overhead ratio
```

## Input Sizes

| Category | Small | Large |
|---|---|---|
| Elementwise, GELU | n=4096 | n=1,048,576 |
| Matmul | M=N=K=64 | M=N=K=1024 |
| Batched matmul | B=4, M=N=K=32 | B=16, M=N=K=256 |
| Tensor3D matmul | B=4, M=N=K=32 | B=16, M=N=K=256 |
| Matvec | M=64, N=64 | M=4096, N=4096 |
| Scalar multiply | n=4096 | n=1,048,576 |
| Reduction | rows=64, cols=256 | rows=1024, cols=4096 |
| Argreduce | rows=64, cols=256 | rows=1024, cols=4096 |
| Softmax | rows=64, cols=256 | rows=1024, cols=4096 |
| Norm (RMS, Layer) | rows=64, cols=256 | rows=1024, cols=4096 |
| Norm (Frobenius, L1, L2) | n=4096 | n=1,048,576 |
| Loss | n=4096 | n=1,048,576 |
| Cumulative | rows=64, cols=256 | rows=1024, cols=4096 |

## Grid/Block Configuration

Match SeGuRu's existing launch configurations:
- Elementwise: grid=ceil(n/256), block=256
- Matmul: grid=(ceil(N/16), ceil(M/16)), block=(16,16)
- Reduction: grid=rows, block=min(next_pow2(cols/2), 256)
- Softmax: grid=rows, block=min(cols, 256)
- Loss: grid=1, block=min(n, 256)
- Cumulative: grid=rows, block=1

## Feature Gating

Add `bench` feature to `Cargo.toml`:
```toml
[features]
bench = []

[[bin]]
name = "bench"
path = "src/bin/bench.rs"
required-features = ["bench"]
```

The `build.rs` only compiles CUDA when `bench` feature is active, so normal `cargo test` is unaffected.

## Error Handling

- If CUDA compilation fails → skip CUDA columns, output SeGuRu-only CSV
- If PyTorch not available → skip PyTorch comparison
- If a kernel fails → record "FAIL" in CSV, continue to next kernel

## Dependencies

- `nvcc` (CUDA toolkit) — for compiling reference kernels
- `cc` crate — for build.rs CUDA compilation
- Python 3 + PyTorch — for PyTorch comparison (optional)
- No new Rust runtime dependencies (std::time for timing)
