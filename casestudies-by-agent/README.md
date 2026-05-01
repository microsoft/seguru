# Case Studies by Agent

This directory contains GPU benchmark case studies ported to Rust with SeGuRu.

The original benchmark sources are:

- [polybenchGpu](https://github.com/sgrauerg/polybenchGpu)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- [AES](https://github.com/cihangirtezcan/CUDA_AES)
- [gpusorting] (https://github.com/b0nes164/GPUSorting)
- [heongpu](https://github.com/Alisah-Ozcan/HEonGPU)


## Contents

- `aes/`: AES-128 encryption ported from CUDA_AES. Includes CUDA reference for comparison benchmarks.
- `gpusorting/`: Radix sort ported from GPUSorting (OneSweep algorithm). Includes CUDA reference for comparison benchmarks.
- `heongpu/`: Homomorphic encryption NTT/INTT and Barrett multiplication ported from HEonGPU. Includes CUDA reference for comparison benchmarks.
- `kernelbench/`: KernelBench operators ported through multiple stages:
  - `cuda/`: CUDA kernels generated from the original PyTorch operators.
  - `src/from_cuda/`: Rust/SeGuRu ports derived from the CUDA kernels.
  - `src/`: Rust/SeGuRu implementations written directly against SeGuRu.
- `polybench/`: PolyBench/GPU kernels ported directly to Rust with SeGuRu. These benchmarks only include the Rust/SeGuRu versions.

The top-level `Cargo.toml` defines a Rust workspace containing all case study crates.
