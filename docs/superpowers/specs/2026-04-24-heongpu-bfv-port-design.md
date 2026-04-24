# HEonGPU BFV Pipeline — SeGuRu Port Design

## Problem

Port the core BFV (Brakerski-Fan-Vercauteren) homomorphic encryption GPU kernels from [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) to SeGuRu Rust. This demonstrates SeGuRu's capability for cryptographic workloads requiring 64-bit modular arithmetic, NTT transforms, and RNS (Residue Number System) representations.

## Scope

**In scope**: 6 kernel files from `src/lib/kernel/` (~2.4K LOC, 58 CUDA kernels):
- `addition.cu` — 12 kernels (element-wise modular ops)
- `small_ntt.cu` — 6 device functions (NTT forward/inverse)
- `encoding.cu` — 10 kernels (plaintext encoding/decoding)
- `encryption.cu` — 7 kernels (BFV/CKKS encrypt)
- `decryption.cu` — 15 kernels (BFV/CKKS decrypt)
- `multiplication.cu` — 14 kernels (homomorphic multiply)

**Out of scope** (future work):
- `switchkey.cu` (37 kernels, key switching/relinearization)
- `keygeneration.cu` (31 kernels, key generation with RNG)
- `bootstrapping.cu` (23 kernels, TFHE bootstrapping)
- `contextpool.cpp` (host-side context management)

## Approach

### 1. HEonGPU as Git Submodule

Add HEonGPU as a git submodule rather than copying CUDA source:
```
examples/heongpu-by-agent/
├── HEonGPU/              # git submodule → github.com/Alisah-Ozcan/HEonGPU
├── src/
│   ├── lib.rs            # GPU kernels + tests
│   ├── modular.rs        # Modular arithmetic primitives (host + device)
│   ├── addition.rs       # Addition/subtraction kernels
│   ├── ntt.rs            # Small NTT kernels
│   ├── encoding.rs       # Encode/decode kernels
│   ├── encryption.rs     # BFV encrypt kernels
│   ├── decryption.rs     # BFV decrypt kernels
│   ├── multiplication.rs # Homomorphic multiply kernels
│   ├── cuda_ffi.rs       # FFI to HEonGPU CUDA (bench feature)
│   └── bin/
│       └── bench.rs      # Benchmark binary
├── build.rs              # Feature-gated nvcc compilation
├── Cargo.toml
├── README.md
└── REPORT.md
```

### 2. Core Data Types

HEonGPU operates on polynomials in RNS representation. Key types to implement:

```rust
/// Barrett reduction parameters for a 64-bit modulus
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Modulus64 {
    pub value: u64,    // prime modulus q
    pub bit_: u64,     // bit length of q
    pub sbit: u64,     // shift for Barrett
    pub ratio0: u64,   // Barrett ratio low
    pub ratio1: u64,   // Barrett ratio high
}

type Data64 = u64;
```

### 3. Modular Arithmetic Primitives

The foundation layer — all kernels depend on these:

```rust
#[gpu::device]
fn mod_add(a: u64, b: u64, m: &Modulus64) -> u64 {
    let sum = a + b;
    if sum >= m.value { sum - m.value } else { sum }
}

#[gpu::device]
fn mod_sub(a: u64, b: u64, m: &Modulus64) -> u64 {
    if a >= b { a - b } else { a + m.value - b }
}

#[gpu::device]
fn mod_mul(a: u64, b: u64, m: &Modulus64) -> u64 {
    // Barrett reduction using u128 intermediate
    // or manual hi:lo 64-bit multiply if u128 unavailable on GPU
    let product = (a as u128) * (b as u128);
    (product % (m.value as u128)) as u64
}
```

**Critical question**: Whether SeGuRu supports `u128` on GPU. If not, implement Barrett reduction using `u64` pairs:
```rust
fn mulhi(a: u64, b: u64) -> u64 { ((a as u128 * b as u128) >> 64) as u64 }
```

### 4. Kernel Porting Strategy

Each HEonGPU kernel follows a consistent pattern:
- 3D grid: `(ring_size/block, rns_count, cipher_count)`
- Thread → polynomial coefficient index
- `blockIdx.y` → RNS modulus index
- `blockIdx.z` → ciphertext component index

SeGuRu translation pattern:
```rust
#[gpu::cuda_kernel]
pub fn addition_kernel(
    in1: &[u64], in2: &[u64], out: &mut [u64],
    modulus: &[Modulus64], n_power: u32,
) {
    let idx = gpu::global_id::<gpu::DimX>() as u32;
    let idy = gpu::block_id::<gpu::DimY>() as u32;
    let idz = gpu::block_id::<gpu::DimZ>() as u32;
    let grid_y = gpu::grid_dim::<gpu::DimY>() as u32;

    let location = (idx + (idy << n_power) + ((grid_y * idz) << n_power)) as usize;

    let mut c = gpu::chunk_mut(out, gpu::MapLinear::new(1));
    // ... compute with mod_add ...
}
```

**SeGuRu-specific adaptations**:
- `chunk_mut()` for all output writes (no direct slice indexing)
- No closures inside kernels — inline all modular ops or use `#[gpu::device]` helpers
- `smem_alloc` + `chunk_mut` + `sync_threads` pattern for NTT shared memory
- 3D grid via `gpu_config!(grid_x, grid_y, grid_z, block_x, 1, 1, shared_bytes)`

### 5. Module-by-Module Details

#### addition.rs (12 kernels)
Element-wise modular operations on RNS polynomials. Simplest kernels — no shared memory, no NTT. Each kernel is ~10 lines of modular arithmetic.

Kernels: `addition`, `substraction`, `negation`, `addition_plain_bfv_poly`, `substraction_plain_bfv_poly`, `addition_plain_ckks_poly`, `substraction_plain_ckks_poly`, `addition_plain_bfv_poly_inplace`, `substraction_plain_bfv_poly_inplace`, `addition_plain_ckks_poly_inplace`, `substraction_plain_ckks_poly_inplace`, `negation_inplace`

#### ntt.rs (6 device functions)
Number Theoretic Transform using Cooley-Tukey (forward) and Gentleman-Sande (inverse) butterfly operations. Uses shared memory for in-block polynomial storage. Template on Data32/Data64.

Functions: `SmallForwardNTT<Data32>`, `SmallForwardNTT<Data64>`, `SmallInverseNTT<Data32>`, `SmallInverseNTT<Data64>` (plus wrapper globals)

#### encoding.rs (10 kernels)
Converts plaintext values to/from polynomial RNS representation. Handles both BFV (integer) and CKKS (floating-point) encodings.

Key kernels: `encode_kernel_bfv`, `decode_kernel_bfv`, `encode_kernel_double_ckks_conversion`, `compose_kernel`

#### encryption.rs (7 kernels)
BFV/CKKS encryption using public key. Requires random number sampling (via pre-generated random arrays, not inline curand). Key operations: public-key multiply, error addition, modulus scaling.

Key kernels: `pk_u_kernel`, `enc_div_lastq_kernel`, `cipher_message_add_kernel`

#### decryption.rs (15 kernels)
Secret-key multiplication and noise removal. Uses shared memory for warp reductions. Multiple optimization paths (batch x3, fusion variants).

Key kernels: `sk_multiplication`, `decryption_kernel`, `compose_kernel`, `find_max_norm_kernel`

#### multiplication.rs (14 kernels)
Homomorphic ciphertext multiplication with RNS base conversion. Most complex arithmetic — cross-multiplication of ciphertext components with modulus switching.

Key kernels: `fast_convertion`, `cross_multiplication`, `fast_floor`, `cipherplain_kernel`

### 6. Testing Strategy

Each module gets correctness tests using known BFV parameters:
- **Parameters**: N=4096 (polynomial degree), t=65537 (plaintext modulus), q = product of 60-bit primes
- **Test pattern**: Generate random polynomials on host → run SeGuRu kernel → compare against CPU reference implementation
- **CPU reference**: Implement basic modular polynomial arithmetic in host-side Rust for verification
- **Roundtrip test**: encode → encrypt → decrypt → decode and verify plaintext recovery

### 7. Benchmarking

Feature-gated CUDA compilation (same pattern as AES):
- Build HEonGPU kernels via nvcc from submodule
- FFI bindings for key operations
- Time SeGuRu vs CUDA for each operation category
- Measure at different polynomial degrees (N=2048, 4096, 8192, 16384)

## Key Risks

1. **u128 on GPU**: SeGuRu may not support `u128` in device code. Mitigation: implement Barrett reduction with `u64` hi/lo pairs.
2. **NTT shared memory**: Requires careful `smem_alloc` + `sync_threads` coordination. Mitigated by AES experience with the unconditional allocation pattern.
3. **3D grid mapping**: HEonGPU uses 3D grids (ring × RNS × cipher). SeGuRu supports this via `gpu_config!` but `chunk_mut` indexing needs adaptation for multi-dimensional output.
4. **curand dependency**: Encryption kernels use curand for random sampling. Mitigation: pre-generate random arrays on host and pass as kernel parameters.

## Implementation Order

1. Scaffold crate + submodule setup
2. Modular arithmetic primitives (`modular.rs`)
3. Addition kernels + tests (validates primitive layer)
4. Small NTT + tests (enables encrypt/decrypt)
5. Encoding + tests (plaintext handling)
6. Encryption + tests
7. Decryption + tests (roundtrip validation)
8. Multiplication + tests (homomorphic ops)
9. Build.rs + CUDA FFI + benchmarks
10. README + REPORT
