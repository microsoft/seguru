# HEonGPU BFV Pipeline Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 58 HEonGPU BFV homomorphic encryption CUDA kernels to SeGuRu Rust, with tests and benchmarks against the original CUDA via submodule.

**Architecture:** Add HEonGPU as a git submodule. Implement Barrett-reduction modular arithmetic as host+GPU functions, then port each kernel file (addition → NTT → encoding → encryption → decryption → multiplication) with host-side CPU reference tests. Feature-gated CUDA FFI benchmarks compare SeGuRu vs original.

**Tech Stack:** Rust nightly (SeGuRu), CUDA 12.8+, gpu/gpu_host crates, u64 modular arithmetic with u128 intermediates for Barrett reduction.

---

### Task 1: Scaffold Crate and Git Submodule

**Files:**
- Create: `examples/heongpu-by-agent/Cargo.toml`
- Create: `examples/heongpu-by-agent/src/lib.rs`
- Create: `examples/heongpu-by-agent/src/modular.rs`
- Modify: `examples/Cargo.toml` (add workspace member)
- Submodule: `examples/heongpu-by-agent/HEonGPU`

- [ ] **Step 1: Add HEonGPU as git submodule**

```bash
cd /home/ziqiaozhou/seguru
git submodule add https://github.com/Alisah-Ozcan/HEonGPU.git examples/heongpu-by-agent/HEonGPU
```

- [ ] **Step 2: Create Cargo.toml**

Create `examples/heongpu-by-agent/Cargo.toml`:

```toml
[package]
name = "heongpu-gpu"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = { workspace = true }
gpu_host = { workspace = true }

[dev-dependencies]
rand = "0.9"

[features]
bench = []

[[bin]]
name = "heongpu-bench"
path = "src/bin/bench.rs"
required-features = ["bench"]
```

- [ ] **Step 3: Add workspace member**

Add `"heongpu-by-agent"` to `examples/Cargo.toml` members list.

- [ ] **Step 4: Create src/lib.rs**

```rust
pub mod modular;
```

- [ ] **Step 5: Create src/modular.rs with Barrett arithmetic + tests**

Host-side Barrett reduction modular arithmetic mirroring HEonGPU's `OPERATOR_GPU_64`. Key types:

```rust
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Modulus64 {
    pub value: u64,  // prime modulus
    pub bit: u64,    // ceil(log2(value))
    pub mu: u64,     // Barrett parameter: floor(2^(2*bit+1) / value)
}
```

Functions: `mod_add`, `mod_sub`, `mod_mul` (Barrett with u128), `mod_reduce`, `mod_reduce_signed`, `centered_reduction`, `mod_reduce_forced`.

Tests: verify add/sub wraparound, multiply with 60-bit primes, centered_reduction sign flip.

- [ ] **Step 6: Build and run tests**

```bash
cd examples && cargo test -p heongpu-gpu --lib -- modular
```

- [ ] **Step 7: Commit**

```bash
git add .gitmodules examples/heongpu-by-agent examples/Cargo.toml
git commit -m "feat(heongpu): scaffold crate with submodule and Barrett modular arithmetic"
```

---

### Task 2: Addition Kernels (3 core + GPU tests)

**Files:**
- Create: `examples/heongpu-by-agent/src/addition.rs`
- Modify: `examples/heongpu-by-agent/src/lib.rs`

Element-wise modular ops on RNS polynomials. 3D grid: `(ring_size/block, rns_count, cipher_count)`.

- [ ] **Step 1: Write CPU reference functions**

`addition_cpu`, `subtraction_cpu`, `negation_cpu` — iterate over the 3D index space: `location = idx + (idy << n_power) + ((grid_y * idz) << n_power)`.

- [ ] **Step 2: Write CPU tests verifying basic modular arithmetic**

Test with 2 RNS moduli (60-bit primes), ring_size=16, verify wraparound and negation identity.

- [ ] **Step 3: Write GPU kernels**

`addition_kernel`, `subtraction_kernel`, `negation_kernel` — each uses `global_id::<DimX>`, `block_id::<DimY/DimZ>`, `chunk_mut` for output. Modular ops inlined (single add/compare for add/sub, no Barrett needed).

- [ ] **Step 4: Write GPU tests comparing against CPU reference**

N=4096, 2 RNS moduli. Launch with `gpu_config!(grid_x, rns, 1, 256, 1, 1, 0)`.

- [ ] **Step 5: Build and run all tests**

```bash
cd examples && cargo test -p heongpu-gpu --lib -- addition --release
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(heongpu): addition/subtraction/negation GPU kernels with tests"
```

---

### Task 3: Barrett Multiplication GPU Kernel

**Files:**
- Modify: `examples/heongpu-by-agent/src/addition.rs` (add multiply_elementwise_kernel)
- Modify: `examples/heongpu-by-agent/src/modular.rs` (add barrett_mul_inline)

Barrett multiply is the performance-critical primitive — validate it works on GPU with u128 intermediates.

- [ ] **Step 1: Add GPU element-wise multiply kernel**

Inline Barrett reduction: `a*b` via u128, shift, multiply by mu, subtract. This validates that SeGuRu supports `u128` arithmetic (`as u128` casts) in device code.

- [ ] **Step 2: Write GPU test comparing against CPU mod_mul**

N=4096, 2 RNS moduli with 60-bit primes. Verify every element matches.

- [ ] **Step 3: Build and test**

```bash
cd examples && cargo test -p heongpu-gpu --lib -- test_gpu_multiply --release
```

If u128 fails on GPU: fall back to manual hi:lo multiply using `mul128` with two u64 multiplies.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(heongpu): Barrett modular multiplication GPU kernel"
```

---

### Task 4: BFV Encoding/Decoding

**Files:**
- Create: `examples/heongpu-by-agent/src/encoding.rs`
- Modify: `examples/heongpu-by-agent/src/lib.rs`

- [ ] **Step 1: Implement CPU encode/decode for BFV**

`encode_bfv_cpu`: scale message by floor(q/t), lift to RNS.
`decode_bfv_cpu`: scale down by t/q, round to recover plaintext.

- [ ] **Step 2: Write roundtrip test (encode → decode = identity)**

Small ring (N=16), t=65537, q=60-bit prime. Verify messages 0..15 survive roundtrip.

- [ ] **Step 3: Build and test**

```bash
cd examples && cargo test -p heongpu-gpu --lib -- encoding
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(heongpu): BFV encoding/decoding with CPU roundtrip test"
```

---

### Task 5: Encryption and Decryption

**Files:**
- Create: `examples/heongpu-by-agent/src/encryption.rs`
- Create: `examples/heongpu-by-agent/src/decryption.rs`
- Modify: `examples/heongpu-by-agent/src/lib.rs`

- [ ] **Step 1: Implement CPU pk_u_mul and cipher_message_add**

Element-wise NTT-domain multiply and add.

- [ ] **Step 2: Implement CPU sk_multiplication (decrypt core)**

`output[i] = c0[i] + c1[i] * sk[i] mod q`

- [ ] **Step 3: Write CPU tests**

Verify trivial case: c1=0, sk=anything → decrypt returns c0.

- [ ] **Step 4: Add GPU kernels for sk_multiplication and cipher_message_add**

3D grid, inline Barrett for multiply, simple add for message addition.

- [ ] **Step 5: Write GPU encrypt-decrypt roundtrip test**

Trivial encryption (c0=message, c1=0) → decrypt with any sk → verify recovery.

- [ ] **Step 6: Build and test**

```bash
cd examples && cargo test -p heongpu-gpu --lib --release
```

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(heongpu): encryption/decryption GPU kernels with roundtrip test"
```

---

### Task 6: Homomorphic Multiplication

**Files:**
- Create: `examples/heongpu-by-agent/src/multiplication.rs`
- Modify: `examples/heongpu-by-agent/src/lib.rs`

- [ ] **Step 1: Implement CPU cross_multiplication**

Given ct1=(a,b), ct2=(c,d): out0=a*c, out1=a*d+b*c, out2=b*d (element-wise, NTT domain).

- [ ] **Step 2: Implement CPU cipher_plain_mul**

Element-wise multiply of ciphertext by plaintext polynomial.

- [ ] **Step 3: Write CPU tests**

Verify cross-multiplication produces correct components.

- [ ] **Step 4: Add GPU kernels**

Cross-multiplication with 3 output arrays, cipher-plain multiply.

- [ ] **Step 5: Write GPU tests**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(heongpu): homomorphic multiplication GPU kernels"
```

---

### Task 7: Benchmark Infrastructure

**Files:**
- Create: `examples/heongpu-by-agent/build.rs`
- Create: `examples/heongpu-by-agent/cuda/bench_wrapper.cu`
- Create: `examples/heongpu-by-agent/src/cuda_ffi.rs`
- Create: `examples/heongpu-by-agent/src/bin/bench.rs`

- [ ] **Step 1: Create build.rs**

Feature-gated nvcc compilation of bench_wrapper.cu with HEonGPU include paths.

- [ ] **Step 2: Create CUDA bench wrapper**

C-linkage functions that call HEonGPU kernels (addition, multiply, encrypt, decrypt) with host-pointer interfaces and timing.

- [ ] **Step 3: Create cuda_ffi.rs**

Extern C bindings for bench wrapper functions.

- [ ] **Step 4: Create bench.rs**

Benchmark SeGuRu vs CUDA vs CPU for each operation at N=4096, 8192, 16384. Include 10-second CPU timeout with extrapolation.

- [ ] **Step 5: Run benchmarks**

```bash
cd examples && cargo run --bin heongpu-bench --features bench --release -p heongpu-gpu
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(heongpu): benchmark infrastructure — SeGuRu vs CUDA vs CPU"
```

---

### Task 8: Documentation

**Files:**
- Create: `examples/heongpu-by-agent/README.md`
- Create: `examples/heongpu-by-agent/REPORT.md`

- [ ] **Step 1: Write README.md**

Quick start, project structure, kernel list, test descriptions, performance highlights.

- [ ] **Step 2: Write REPORT.md**

Full benchmark tables, throughput analysis, SeGuRu vs CUDA ratios, GPU speedup vs CPU.

- [ ] **Step 3: Commit**

```bash
git commit -m "docs(heongpu): README and performance report"
```
