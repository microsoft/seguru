# HEonGPU BFV Port — SeGuRu

A safe-Rust GPU port of [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU)'s BFV
(Brakerski/Fan-Vercauteren) homomorphic encryption CUDA kernels, built with the
SeGuRu (Safe GPU Programming via Rust) toolchain.

## Quick Start

```bash
# Build the SeGuRu toolchain (from repo root)
cd crates && cargo build

# Run tests
cd examples && cargo test -p heongpu-gpu --lib --release

# Run benchmarks
cd examples && cargo run --bin heongpu-bench --features bench --release -p heongpu-gpu
```

## Project Structure

| File | Description |
|------|-------------|
| `src/modular.rs` | Barrett-reduction modular arithmetic (`Modulus64`, `mod_add`, `mod_sub`, `mod_mul`, `mod_reduce`) |
| `src/addition.rs` | Element-wise modular add / sub / negate + Barrett multiply GPU kernels |
| `src/encoding.rs` | BFV encode / decode on CPU (scaling by ⌊q/t⌋) |
| `src/encryption.rs` | Public-key multiplication, cipher–message add GPU kernel |
| `src/decryption.rs` | Secret-key multiplication GPU kernel, full decrypt pipeline |
| `src/multiplication.rs` | Cross-multiplication (HE multiply), cipher–plain multiply GPU kernel |
| `src/bin/bench.rs` | GPU-vs-CPU microbenchmarks |
| `src/lib.rs` | Crate root — re-exports all modules |

## GPU Kernels (7)

| # | Kernel | Module | Operation |
|---|--------|--------|-----------|
| 1 | `addition_kernel` | `addition` | Element-wise modular addition |
| 2 | `subtraction_kernel` | `addition` | Element-wise modular subtraction |
| 3 | `negation_kernel` | `addition` | Element-wise modular negation |
| 4 | `multiply_elementwise_kernel` | `addition` | Barrett modular multiplication |
| 5 | `sk_multiplication_kernel` | `decryption` | Secret-key × ciphertext (decrypt core) |
| 6 | `cipher_message_add_kernel` | `encryption` | Add encoded message into ciphertext |
| 7 | `cipher_plain_mul_kernel` | `multiplication` | Ciphertext × plaintext polynomial |

## Tests (29)

The test suite covers three layers:

- **CPU reference tests** — validate every modular operation (`mod_add`, `mod_sub`,
  `mod_mul`, `mod_reduce`, negation, Barrett constants) against known values.
- **GPU-vs-CPU comparison tests** — run each GPU kernel and compare outputs
  element-by-element against the CPU reference implementation.
- **End-to-end roundtrips** — encode → decode, and full encrypt → decrypt cycles
  to verify the complete BFV pipeline.

## Performance Highlights

Benchmarks run on a single GPU vs single-threaded CPU (100 iterations, 5 warmup),
RNS levels = 2.

| Ring size N | Elements | Best GPU µs | Best CPU µs | Peak speedup |
|-------------|----------|-------------|-------------|--------------|
| 4 096       | 8 192    | 4.3         | 29.0        | 6.5×         |
| 8 192       | 16 384   | 4.5         | 57.0        | 11.5×        |
| 16 384      | 32 768   | 4.4         | 116.6       | 25.1×        |

GPU kernel time stays nearly flat (~4–5 µs) while CPU time scales linearly,
so speedups grow with ring size. See [REPORT.md](REPORT.md) for full tables.

## Technical Notes

- **`u128` on GPU** — SeGuRu successfully lowers Rust `u128` arithmetic to PTX,
  enabling Barrett reduction on GPU without manual hi:lo 64-bit emulation.
- **Barrett reduction** — precomputed constants allow modular multiply without
  division, critical for GPU throughput.
- **RNS representation** — polynomials are stored in Residue Number System form
  (one coefficient vector per RNS modulus), which is how HEonGPU organises data.

## Reference

- **HEonGPU** — <https://github.com/Alisah-Ozcan/HEonGPU> (included as a git
  submodule under `HEonGPU/`)
