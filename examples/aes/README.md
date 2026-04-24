# AES-128 ECB on GPU — SeGuRu Example

AES-128 ECB encryption and decryption on NVIDIA GPUs using the [T-table optimization](https://en.wikipedia.org/wiki/AES_key_schedule#T-tables). Each GPU thread processes one 16-byte AES block independently, making ECB mode embarrassingly parallel.

This crate includes both a **SeGuRu (Rust)** implementation and a **CUDA C++** reference, with benchmarks comparing the two.

## Quick Start

```bash
# Build the compiler toolchain first
cd crates && cargo build

# Run tests (5 tests: key expansion, NIST vector, CPU roundtrip, GPU encrypt, GPU roundtrip)
cd examples && cargo test -p aes-gpu --lib --release

# Run benchmarks (SeGuRu vs CUDA C++)
cd examples && cargo run --bin bench --features bench --release -p aes-gpu
```

## What's Implemented

| Variant | Encrypt | Decrypt | Shared Memory |
|---------|---------|---------|---------------|
| T-table (SeGuRu) | ✅ | ✅ | 4 KB (TE0–TE3 / TD0–TD3) |
| T-table (CUDA C++) | ✅ | ✅ | `__constant__` memory |
| Textbook S-box (CUDA C++) | ✅ | ✅ | S-box in shared memory |

The SeGuRu kernels use the T-table variant (4 precomputed 256-entry lookup tables) for all 10 AES rounds, with S-box values extracted from TE0 for the final encrypt round and inv_sbox read from global memory for the final decrypt round.

## Performance

SeGuRu vs CUDA C++ (T-table, median of 100 iterations):

| Data Size | Encrypt Ratio | Decrypt Ratio |
|-----------|:------------:|:------------:|
| 16 KB     | 1.64×        | 1.81×        |
| 1 MB      | 1.19×        | 1.27×        |
| 4 MB      | 1.00×        | 1.17×        |
| 16 MB     | **0.90×** ✨  | 1.05×        |

> **0.90×** = SeGuRu is 10% *faster* than hand-written CUDA at 16 MB.
> Overhead at small sizes is fixed launch latency (~6 µs), not kernel compute.

See [REPORT.md](REPORT.md) for full results and analysis.

## Project Structure

```
aes/
├── src/
│   ├── lib.rs           # GPU kernels + tests
│   ├── aes_common.rs    # AES-128 constants, key expansion, CPU reference
│   ├── cuda_ffi.rs      # FFI bindings to CUDA bench functions
│   └── bin/
│       └── bench.rs     # Benchmark binary (SeGuRu vs CUDA)
├── cuda/
│   ├── aes_kernels.cu   # CUDA C++ reference (textbook + T-table)
│   └── aes_kernels.h    # C header for FFI
├── build.rs             # Feature-gated nvcc compilation
├── Cargo.toml
└── REPORT.md            # Detailed performance report
```

## Kernel Design

### Data layout

- **Input/output**: `&[u32]` — 4 big-endian u32 per 16-byte AES block
- **T-tables**: 1024 u32 in shared memory: `[T0(256) | T1(256) | T2(256) | T3(256)]`
- **Round keys**: 44 u32 (AES-128 = 11 rounds × 4 words)

### Shared memory pattern

SeGuRu requires `smem_alloc` and `chunk_mut` to be called **unconditionally** by all threads. The computation is then guarded with an `if`:

```rust
#[gpu::cuda_kernel(dynamic_shared)]
pub fn aes128_encrypt_ttable_kernel(input: &[u32], output: &mut [u32], ...) {
    let smem = smem_alloc.alloc::<u32>(1024);          // all threads
    let mut sc = smem.chunk_mut(MapLinear::new(4));     // all threads write
    sc[0] = te_tables[(ltid * 4) as usize];
    // ...
    sync_threads();

    if tid < num_blocks {                               // only valid threads compute
        let val = *smem[((s0 >> 24) & 0xff) as usize]; // read via *smem[i]
        // ... 10 rounds of T-table lookups ...
    }
}
```

### Thread mapping

- **1 thread = 1 AES block** (16 bytes)
- Block size fixed at 256 (matches T-table cooperative load: 256 threads × 4 entries = 1024)
- Grid size = `ceil(num_blocks / 256)`

## Tests

| Test | What it checks |
|------|---------------|
| `test_key_expansion` | AES-128 key schedule against FIPS 197 |
| `test_nist_encrypt` | CPU encrypt against NIST test vector |
| `test_roundtrip` | CPU encrypt → decrypt roundtrip |
| `test_aes_encrypt_nist_ttable` | GPU encrypt against NIST test vector |
| `test_aes_roundtrip_ttable` | GPU encrypt → decrypt roundtrip (64 blocks) |
