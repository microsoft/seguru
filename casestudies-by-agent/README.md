# Case Studies by Agent

This directory contains GPU benchmark case studies ported to Rust with SeGuRu.

The original benchmark sources are:

- [polybenchGpu](https://github.com/sgrauerg/polybenchGpu)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)

## Contents

- `polybench/`: PolyBench/GPU kernels ported directly to Rust with SeGuRu. These benchmarks only include the Rust/SeGuRu versions.
- `kernelbench/`: KernelBench operators ported through multiple stages:
  - `cuda/`: CUDA kernels generated from the original PyTorch operators.
  - `src/from_cuda/`: Rust/SeGuRu ports derived from the CUDA kernels.
  - `src/`: Rust/SeGuRu implementations written directly against SeGuRu.

The top-level `Cargo.toml` defines a Rust workspace containing the KernelBench crate and each PolyBench benchmark crate.
