# HEonGPU BFV Port — Performance Report

## Summary

The SeGuRu GPU port of HEonGPU's BFV kernels achieves **up to 25× speedup** over
a single-threaded CPU baseline on element-wise modular arithmetic operations. GPU
kernel time remains nearly constant across ring sizes, while CPU time scales
linearly with element count.

## Methodology

- **Iterations:** 100 timed iterations per operation, preceded by 5 warmup
  iterations (excluded from measurements).
- **Timing:** Wall-clock microsecond precision (`std::time::Instant`).
- **GPU:** Kernel launch + execution time, including device synchronisation.
- **CPU:** Single-threaded loop over the same element count.
- **RNS levels:** 2 (total elements = N × 2).
- **Modulus:** 64-bit prime, Barrett reduction for multiplication.

## Benchmark Results

### Ring size N = 4 096 (8 192 elements)

| Operation       | GPU (µs) | CPU (µs) | Speedup |
|-----------------|----------|----------|---------|
| Addition        | 4.3      | 10.3     | 2.4×    |
| Barrett Mul     | 5.5      | 26.6     | 4.9×    |
| SK Multiply     | 5.2      | 29.5     | 5.7×    |
| Cipher × Plain  | 4.5      | 29.0     | 6.5×    |

### Ring size N = 8 192 (16 384 elements)

| Operation       | GPU (µs) | CPU (µs) | Speedup |
|-----------------|----------|----------|---------|
| Addition        | 4.5      | 16.3     | 3.6×    |
| Barrett Mul     | 5.3      | 49.7     | 9.3×    |
| SK Multiply     | 4.6      | 50.6     | 10.9×   |
| Cipher × Plain  | 5.0      | 57.0     | 11.5×   |

### Ring size N = 16 384 (32 768 elements)

| Operation       | GPU (µs) | CPU (µs) | Speedup |
|-----------------|----------|----------|---------|
| Addition        | 4.4      | 32.6     | 7.4×    |
| Barrett Mul     | 5.4      | 108.8    | 20.1×   |
| SK Multiply     | 4.8      | 98.8     | 20.5×   |
| Cipher × Plain  | 4.7      | 116.6    | 25.1×   |

## Analysis

### GPU time is nearly constant

All GPU operations complete in **4–5.5 µs** regardless of whether the ring size
is 4 096 or 16 384. At small sizes the kernel launch overhead dominates; as ring
size grows, the GPU fills more threads but the overall latency barely changes.

### CPU time scales linearly

CPU time roughly doubles each time the element count doubles, which is expected
for O(n) element-wise loops.

### Compute-intensive operations benefit more

Addition (a single modular add) shows the smallest GPU advantage because the
per-element work is minimal — the kernel launch cost is a larger fraction.
Barrett multiplication and the HE-specific kernels (SK multiply, cipher × plain)
perform more ALU work per element and therefore achieve higher speedups.

### Speedup increases with ring size

| Ring size | Addition | Barrett Mul | SK Multiply | Cipher × Plain |
|-----------|----------|-------------|-------------|----------------|
| 4 096     | 2.4×     | 4.9×        | 5.7×        | 6.5×           |
| 8 192     | 3.6×     | 9.3×        | 10.9×       | 11.5×          |
| 16 384    | 7.4×     | 20.1×       | 20.5×       | 25.1×          |

Larger ring sizes amortise the fixed kernel-launch cost over more elements,
improving GPU utilisation and widening the gap with the CPU.

## Comparison with Original HEonGPU

Direct performance comparison with the original CUDA HEonGPU library is **future
work**. It requires building HEonGPU's CUDA kernels and wrapping them via FFI so
both implementations can be measured on identical hardware with the same inputs.
The goal is to quantify the overhead (if any) of SeGuRu's Rust-to-PTX
compilation path relative to hand-written CUDA.

## Porting Effort

| Metric | Value |
|--------|-------|
| Rust source lines (excl. bench) | ~1 650 |
| Benchmark harness | ~240 lines |
| GPU kernels | 7 |
| CPU reference implementations | 6 |
| Tests | 29 |
| Total Rust LOC | ~1 900 |

The port covers the core element-wise BFV operations. NTT (Number Theoretic
Transform) and key-switching kernels are not yet ported and represent the next
phase of work.
