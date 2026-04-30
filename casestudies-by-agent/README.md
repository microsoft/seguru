# SeGuRu Case Studies (AI-Generated)

This workspace contains 5 GPU case studies ported to SeGuRu by AI agents, demonstrating safe Rust GPU programming across domains including cryptography, sorting, homomorphic encryption, neural network operations, and linear algebra.

## Prerequisites

- SeGuRu compiler toolchain (`rustc-gpu`) built from `crates/`
- CUDA toolkit (for benchmark features with CUDA reference comparisons)
- LLVM 20 on PATH
- NVIDIA GPU with compatible driver

The workspace `.cargo/config.toml` configures the build automatically:
```toml
[build]
rustc = "rustc-gpu"
rustflags = ["--emit=mir", "-Zmir-opt-level=3"]

[env]
USE_FAST = "true"
USE_FTZ = "true"
NVPTX_FEATURES = "+ptx87"
```

## Quick Start

```bash
cd casestudies-by-agent

# Run all tests (requires GPU)
cargo test --release

# Run tests for a specific case study
cargo test -p aes-gpu --release
cargo test -p gpusorting-by-agent --release
cargo test -p heongpu-gpu --release
cargo test -p polybench-atax --release
```

## Case Studies

### AES — AES-128 ECB Encryption

T-table optimized AES-128 encryption/decryption on GPU. Each thread processes one 16-byte block using shared-memory lookup tables.

| | |
|---|---|
| **Kernels** | 2 (encrypt + decrypt) |
| **Tests** | 5 (NIST vectors, roundtrip) |
| **Benchmark** | SeGuRu vs CUDA vs CPU |
| **Peak throughput** | 138 GB/s (487× over CPU) |

```bash
# Tests
cargo test -p aes-gpu --release

# Benchmark (compiles CUDA reference)
cargo run -p aes-gpu --release --features bench --bin aes-bench
```

---

### GPUSorting — Radix Sort

256-radix sort (4 passes for 32-bit keys) with upsweep histogram, parallel scan, and downsweep scatter phases.

| | |
|---|---|
| **Kernels** | 3 (upsweep, scan, downsweep) |
| **Tests** | 8 (correctness across sizes) |
| **Benchmark** | N/A (correctness-focused) |

```bash
cargo test -p gpusorting-by-agent --release
```

---

### HEonGPU — Homomorphic Encryption

Complete port of 149 CUDA kernels from [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) covering BFV/CKKS homomorphic encryption operations with Barrett-reduction modular arithmetic.

| | |
|---|---|
| **Kernels** | 42 GPU + 107 CPU reference |
| **Tests** | 41 |
| **Benchmark** | SeGuRu vs CUDA vs CPU |
| **Speedup** | 6.5–25.1× over CPU |

```bash
# Tests
cargo test -p heongpu-gpu --release

# Benchmark (compiles CUDA reference)
cargo run -p heongpu-gpu --release --features bench --bin heongpu-bench
```

---

### KernelBench — Neural Network Operations

24 LLM-generated kernels covering activations (ReLU, GELU, Swish), normalizations (LayerNorm, RMSNorm), reductions (softmax, argmax), and pooling. Includes both PyTorch-style and raw-CUDA-style variants.

| | |
|---|---|
| **Kernels** | 48 (24 PyTorch + 24 from-CUDA) |
| **Tests** | 0 (benchmark-only) |
| **Benchmark** | CLI runner with JSON output |

```bash
# Run a specific kernel benchmark
cargo run -p kernelbench --release -- \
  --problem relu \
  --in-dir /path/to/inputs \
  --out-dir /path/to/outputs \
  --iters 100 \
  --shape 1024,1024
```

**Supported problems:** `relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `softplus`, `leaky_relu`, `softmax`, `log_softmax`, `layer_norm`, `rms_norm`, `l1_norm`, `l2_norm`, `sum_dim`, `mean_dim`, `max_dim`, `argmax_dim`, `cumsum`, `mse_loss`, `max_pool1d`, `empty`

Append `_fc` for the from-CUDA variant (e.g., `relu_fc`).

Input format: raw little-endian f32 binary files.

---

### PolyBench — Linear Algebra Suite

19 PolyBench kernels covering BLAS operations, stencil codes, and matrix factorizations.

| | |
|---|---|
| **Kernels** | ~38 (2 per sub-crate) |
| **Tests** | 19 (1 per sub-crate) |
| **Benchmark** | N/A |

**Sub-crates:** atax, bicg, conv2d, conv3d, corr, covar, doitgen, fdtd2d, gemm, gesummv, gramschm, jacobi1d, jacobi2d, lu, mvt, syr2k, syrk, threemm, twomm

```bash
# Test all PolyBench kernels
cargo test --release -p polybench-atax -p polybench-bicg -p polybench-conv2d \
  -p polybench-conv3d -p polybench-corr -p polybench-covar -p polybench-doitgen \
  -p polybench-fdtd2d -p polybench-gemm -p polybench-gesummv -p polybench-gramschm \
  -p polybench-jacobi1d -p polybench-jacobi2d -p polybench-lu -p polybench-mvt \
  -p polybench-syr2k -p polybench-syrk -p polybench-threemm -p polybench-twomm
```

## Benchmarking

Case studies with benchmarks (AES, HEonGPU) use the `bench` Cargo feature to gate CUDA reference compilation:

```bash
# Build and run with CUDA comparison
cargo run -p <crate> --release --features bench --bin <bench-binary>
```

Benchmark methodology:
- **Warmup:** 3–10 iterations (discarded)
- **Timed:** 100 iterations (median reported)
- **Metrics:** Wall-clock time (µs), throughput (GB/s), speedup ratios

For KernelBench, the CLI produces JSON output suitable for automated comparison.

## Environment Variables

| Variable | Purpose |
|---|---|
| `DISABLE_GPU_BOUND_CHECK=true` | Disable bounds checking (faster, requires clean rebuild) |
| `RUST_MIN_STACK=67108864` | Increase stack for large kernel compilation |
| `CUDA_PATH` | Custom CUDA installation path (default: `/usr/local/cuda`) |

When toggling `DISABLE_GPU_BOUND_CHECK`, clean before rebuilding:
```bash
cargo clean
DISABLE_GPU_BOUND_CHECK=true cargo test --release
```

## Summary

| Case Study | Domain | GPU Kernels | Tests | CUDA Ref | Benchmark |
|---|---|---|---|---|---|
| AES | Cryptography | 2 | 5 | ✅ | ✅ |
| GPUSorting | Sorting | 3 | 8 | — | — |
| HEonGPU | Homomorphic Encryption | 42 | 41 | ✅ | ✅ |
| KernelBench | Neural Network Ops | 48 | — | — | ✅ |
| PolyBench | Linear Algebra | ~38 | 19 | — | — |
