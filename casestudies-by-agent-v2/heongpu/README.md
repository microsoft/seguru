# heongpu-gpu — HEonGPU primitives in safe Rust on SeGuRu

A port of the core GPU kernels of [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU)
— word-size modular arithmetic, the negacyclic number-theoretic transform, and
the element-wise ciphertext operations that sit on top of them — written in
**safe Rust** and compiled to PTX by SeGuRu.

Everything device-side is safe Rust. The only `unsafe` in the crate is in
`src/cuda_ffi.rs`, which declares the `extern "C"` entry points of the CUDA
reference used for benchmarking; the SeGuRu kernels themselves contain none.

`cuda/heongpu_ref.cu` is an **instruction-for-instruction mirror** of the Rust
kernels: same tiling, same radix-8 register-resident butterflies, same global
pass decomposition, same Shoup/Barrett formulations, `const __restrict__`
pointers throughout. It is a deliberately *strong* baseline — the point of this
case study is to find out where SeGuRu loses, not to manufacture a win.

## Parameters

* Modulus: `DEFAULT_Q = 576460752300015617`, a 59-bit NTT-friendly prime with
  `2^17 | q - 1`, so a primitive `2N`-th root of unity exists for every ring
  size up to `N = 65536`.
* Ring sizes: `N ∈ {4096, 8192, 16384, 32768, 65536}`, `Z_q[x]/(x^N + 1)`.

## Operations

### `modular` — arithmetic mod q

* `add_mod` / `sub_mod` / `neg_mod` — branch-free conditional correction.
* `mul_mod` — Barrett reduction with `mu = floor(2^(2*bit+1) / q)`, the
  formulation used by HEonGPU's `OPERATOR_GPU_64`. Used when both operands are
  runtime values.
* `mul_mod_shoup` — Shoup multiplication by a precomputed operand carrying
  `w' = floor(w * 2^64 / q)`. Two multiplies, one multiply-high, one
  conditional subtraction. This is what every NTT butterfly uses.

Both multiply paths need a 64×64→128 product. SeGuRu lowers Rust `u128`
arithmetic to PTX `mul.hi.u64` / `mul.lo.u64`, so the natural
`(a as u128) * (b as u128)` spelling works verbatim in device code.

### `ntt` — negacyclic NTT over `Z_q[x]/(x^N + 1)`

* Forward: Cooley–Tukey decimation-in-time, `psi_rev` twiddles, natural-order
  input, bit-reversed output.
* Inverse: Gentleman–Sande decimation-in-frequency, bit-reversed input,
  natural output, `n^-1` folded into the last stage.

The pair composes to the identity with no explicit bit-reversal permutation
(the standard Longa–Naehrig formulation used by HE libraries).

### `arith` — element-wise ciphertext operations

`poly_add`, `poly_sub`, `poly_neg`, `poly_mul` (Barrett), `poly_mul_scalar`
(Shoup), `cipher_plain_mul`. All are trivially memory-bound streaming kernels.

## NTT design

* `ntt_forward_tile` / `ntt_inverse_tile` hold a `TILE = 4096` coefficient
  chunk (32 KiB) in `GpuShared` and run **12 butterfly stages** with only three
  block-wide barriers. 512 threads own 8 coefficients each, so every group of
  three stages is a **radix-8 sub-transform done entirely in registers**, and
  shared memory is touched once per three stages rather than once per stage.
* The remaining `log2(N) - 12` stages (at most four, for `N ≤ 65536`) have
  butterfly distances larger than a tile and run as separate global passes
  (`ntt_stage_forward` / `ntt_stage_inverse`), four butterflies per thread.
* For `N = 4096` the whole transform is a single kernel launch with one global
  read and one global write.

### Making shared memory work under SeGuRu's rules

SeGuRu only admits a shared-memory **write** through a static `reshape_map!`,
which proves at compile time that the threads' write sets are disjoint. That
rules out the textbook variable-distance butterfly, where the write index
depends on a runtime stage counter.

The way out is the radix-8 formulation above: the per-round gather patterns are
exactly `pos = t_lo + j*LOW + t_hi*8*LOW` with `LOW ∈ {512, 64, 8, 1}`, which
`reshape_map!` expresses natively. Reads are ordinary slice reads. **This is a
real constraint, and it did shape the design** — but it pushed it in the right
direction: 3 barriers instead of 12, and the arithmetic stays in registers.

It is also the reason several of the optimisations below had to be expressed as
`macro_rules!` rather than `#[gpu::device]` functions (see experiment C).

---

## Optimisation log

Starting point: forward NTT at **1.52–1.58×** CUDA, inverse at **1.29–1.35×**,
element-wise ops at 1.01–1.03× (already at the memory roof, deliberately left
untouched).

Method: extract SeGuRu's PTX (`target/release/deps/gpu/heongpu_gpu-*.ptx`) and
the CUDA PTX (`nvcc -arch=sm_80 -O3 -ptx`), split by `.entry`, bucket
instructions by class, and diff. Then SASS via
`ptxas -arch=sm_80 -O3 -v` + `cuobjdump -sass`.

### Baseline PTX diff — `ntt_forward_tile`

| class | SeGuRu | CUDA |
|---|---|---|
| **`ld.global.u32`** | **128** | **0** |
| **`ld.global.u64`** | **0** | **64** |
| `st.global.u32` | 16 | 0 |
| `st.global.u64` | 0 | 8 |
| `setp` | 216 | 144 |
| `selp` | 288 | 144 |
| `cvt` | 58 | 4 |
| `mul.hi` | 48 | 48 |
| `mul.lo` | 112 | 117 |
| `ld.shared.u64` / `st.shared.u64` | 20 / 24 | 24 / 24 |
| **total** | **1540** | **954** |

**The port author's `u64` / `.align 1` diagnosis is CONFIRMED.** 128
`ld.global.u32` vs 64 `ld.global.u64`, and 16 `st.global.u32` vs 8
`st.global.u64` — exactly 2×, with *zero* 64-bit global accesses on the SeGuRu
side. Kernel pointer parameters are emitted as:

```
.param .u64 .ptr .align 1 heongpu_gpu::ntt::ntt_forward_tile_param_0
```

Each `u64` global load expands to a ten-instruction sequence:

```
cvt.u64.u32; setp.gt.u64; mul.wide.u32; selp.b64; selp.b64; add.s64;
ld.global.u32; ld.global.u32; shl.b64; or.b64
```

— the two 32-bit halves plus the merge (`.align 1` prevents a wider access),
wrapped in the bounds check (`setp` + two `selp` clamping the base to null).

The arithmetic itself is not wasteful: 48 `mul.hi.u64` + 112 `mul.lo.u64` for
48 Shoup butterflies per thread is exactly the two multiplies per butterfly
that the algorithm requires, matching CUDA's 48/117.

Total ratio 1540/954 = **1.61×**, against a measured 1.58× — so at baseline the
kernel was essentially instruction-issue bound and the PTX count predicted the
runtime.

---

### Experiment A — elide twiddle bounds checks with a power-of-two mask · **KEPT**

Adapted from `polybench/src/bin/boundscheck.rs`: give LLVM a provable range
fact instead of asking it to prove one.

Every twiddle index this crate generates is `< N ≤ 65536` by construction
(verified algebraically for the round-3 forward and round-0 inverse maxima,
which come out to exactly `N-1`). So:

```rust
pub const WMAX: usize = 65536;
const WMASK: u32 = (WMAX - 1) as u32;

let wp = &wp[..WMAX];          // constant-length subslice: len is a constant
... wp[(idx & WMASK) as usize] // idx & WMASK < WMAX == wp.len(), provably
```

The device twiddle tables are padded to `WMAX` in `DeviceTables::upload`, so the
mask is a **no-op on the data**, not a silent correctness change.

**PTX evidence** (`ntt_forward_tile`): 1540 → **1293** instructions.
`setp` 216 → 160 (−56, exactly the 56 twiddle accesses), `selp` 288 → 176
(−112, two per access). Every twiddle bounds check is gone.

**Measured** (N=4096): forward 1.59× → **1.51×**, inverse 1.30× → **1.13×**.

Kept. This was the single largest win, and the inverse NTT benefited far more
than the forward.

---

### Experiment B — `U32_2` twiddle tables · **REJECTED**

Hypothesis: if `.align 1` is the problem, giving the element type an alignment
should fix it. `U32_2` is `#[repr(C, align(8))]`. Changed the twiddle parameters
to `&[U32_2]` with an `unpack` helper and packed `Vec<U32_2>` on the host.

**PTX evidence: no change whatsoever.** Zero `ld.global.v2.u32`; still 128
`ld.global.u32`; instruction counts byte-identical. An 8-byte-aligned element
type is not enough to make the backend emit a wider access.

Rejected and reverted. Cost: nothing gained, one extra unpack helper.

(Note: `U32_2` is not in the `gpu::*` prelude — it needs
`use gpu::vector::U32_2;`. Only `U32_4` is re-exported.)

---

### Experiment C — `U32_4`-packed twiddle pairs, loaded from macros · **KEPT**

Since 8-byte alignment did nothing, try 16. The two twiddle tables (`w` and
`w_shoup`) were interleaved into a single `[U32_4]`, one element per twiddle,
lanes `(w_lo, w_hi, ws_lo, ws_hi)`. Kernels now take one `wp: &[U32_4]`
parameter instead of two `&[u64]`.

**First attempt failed.** With the load inside a `#[inline(always)]
#[gpu::device] fn twiddle(...)` helper, the PTX still had **zero**
`ld.global.v4.u32` — four scalar 32-bit loads, as before.

**Diagnostic that found the cause:** writing the identical index expression
*textually inside a kernel body* produced **one `ld.global.v4.u32`**. So the
vectorised path is only reached when the `U32_4` index expression is lexically
in the kernel, not when it arrives via an inlined device function. (Taking or
not taking the `&wp[..WMAX]` subslice made no difference — that was ruled out
separately.)

**Fix within safe Rust:** convert `twiddle`, `fwd_radix8` and `inv_radix8` from
`#[gpu::device] fn` to `macro_rules!`, so the loads land textually in the kernel
body. Semantically identical, purely a lowering workaround.

**PTX evidence** (`ntt_forward_tile`): 1293 → **1223** instructions.
Global memory ops become 28 `ld.global.v4.u32` + 28 `ld.global.u64` +
16 `ld.global.u32` + 16 `st.global.u32`.

Two things worth calling out:

* This **proves alignment is the mechanism**. With a 16-byte-aligned element
  type the backend happily emits `ld.global.u64` *and* `ld.global.v4.u32`. With
  `[u64]` or `[U32_2]` it never does. It is an alignment-propagation defect,
  not a missing feature.
* The 28 `ld.global.u64` are **redundant** — the backend re-reads lanes 0–1 of
  the same `U32_4` it just loaded. A CSE failure, worth 28 wasted loads.

**Measured** (N=4096): forward 1.51× → **1.47×**, inverse 1.13× → **1.10×**.

Kept.

---

### Experiment D — permute `U32_4` lanes to defeat the redundant load · **REJECTED**

To stop the backend re-reading lanes 0–1 as a `u64`, the lanes were permuted to
`(w_lo, ws_lo, w_hi, ws_hi)` so that neither 64-bit field is contiguous.

**PTX evidence: it worked, and it made things worse.** The redundant
`ld.global.u64` disappeared (global memory ops 88 → 60), but total instructions
went *up*, 1223 → 1307, because reassembling each `u64` from non-adjacent lanes
costs `cvt` + `shl` + `or`.

**Measured** (N=4096): forward 1.47× → **1.49×**, inverse 1.10× → **1.12×**.
A real regression.

Rejected and reverted. Useful negative result: on this kernel, trading 28
redundant *cached* loads for 84 ALU instructions is a bad trade.

---

### Experiment E — route input reads through `chunk_mut` · **REJECTED**

`GlobalGroupChunk` indexing derives its index from a static map, so its bounds
check folds away where a raw slice index's does not. Declaring the input
parameter `&mut [u64]` allows a read-only chunk to be built over it with a map
mirroring the access pattern, e.g.
`reshape_map!([8] | [512, ngrid] => layout: [t0, i0, t1])` for the forward tile.

**PTX evidence, mixed:**

* Tile kernels: **byte-identical** instruction counts (1223 / 1361). No change
  at all — the tile input reads were already as cheap as the chunk path.
* Stage kernels: `ntt_stage_forward` 301 → **277**, `ntt_stage_inverse`
  359 → **335** (−24 each, from `cvt` 23→15 and `mul.lo` 27→18 — address
  arithmetic, not bounds checks; `setp`/`selp` were unchanged).

**Measured, back-to-back in the same session:**

| | with E | without E (kept state) |
|---|---|---|
| fwd N=4096 | 1.46× | 1.48× |
| fwd N=65536 | 1.43× | 1.43× |
| inv N=65536 | **1.38×** (667.4 µs) | **1.32×** (639.8 µs) |

Forward: no change (the tile PTX is identical, so this is expected). Inverse:
a real **4 % regression** at large `N`, despite 24 *fewer* PTX instructions in
the stage kernels — presumably scheduling or register-pressure fallout from the
chunk's precondition plumbing.

Rejected and reverted. A clean example of a PTX improvement that does not show
up in wall-clock time, which is exactly why "measure" is a separate step from
"count instructions".

---

### Measurement F — how much is CUDA's read-only cache worth? · **measurement only**

SASS analysis of the kept state turned up the most interesting difference:

```
CUDA:    LDG.E.64.CONSTANT   (read-only / non-coherent path, from const __restrict__)
SeGuRu:  LDG.E.64  /  LDG.E  (ordinary path)
```

SeGuRu **never** emits `ld.global.nc` anywhere in the workspace — there is no
`__restrict__` or read-only-cache equivalent in the API. The natural hypothesis
was that this explains most of the residual gap.

To quantify it, `const`/`__restrict__` was temporarily stripped from the four
NTT kernels in `cuda/heongpu_ref.cu`, the benchmark re-run, and the file
restored byte-exactly (verified with `diff`). **The shipped CUDA baseline is the
unmodified, strong version.**

| N | CUDA fwd, with `const __restrict__` | CUDA fwd, without |
|---|---|---|
| 4096 | 148.9 µs | 148.2 µs |
| 16384 | 327.8 µs | 339.6 µs |
| 65536 | 501.9 µs | 527.6 µs |

**The hypothesis is refuted.** The read-only cache is worth 0–5 % to CUDA on
these kernels, not the ~45 % that would be needed to explain the gap. Reported
here because it was our leading theory and it turned out to be wrong.

---

## Results after optimisation

A100 80GB PCIe, CUDA 13.3, Rust nightly-2025-03-28, release build, 5 warm-up +
50 timed iterations, kernel-only timing. GPU outputs are cross-checked for exact
equality against both the CUDA reference and the CPU oracle before any timing is
reported. Each row transforms **4 Mi coefficients** (a batch of `4194304 / N`
polynomials), so rows are directly comparable.

### Negacyclic NTT

| N | batch | SeGuRu fwd (µs) | CUDA fwd (µs) | SG/CUDA | SeGuRu inv (µs) | CUDA inv (µs) | SG/CUDA |
|---|---|---|---|---|---|---|---|
| 4096 | 1024 | 219.7 | 148.9 | **1.48×** | 136.2 | 122.8 | **1.11×** |
| 8192 | 512 | 345.5 | 238.6 | **1.45×** | 262.5 | 216.2 | **1.21×** |
| 16384 | 256 | 470.9 | 327.8 | **1.44×** | 389.2 | 306.5 | **1.27×** |
| 32768 | 128 | 594.9 | 414.1 | **1.44×** | 513.8 | 395.9 | **1.30×** |
| 65536 | 64 | 719.2 | 501.9 | **1.43×** | 639.8 | 484.4 | **1.32×** |

### Before / after

| | before | after |
|---|---|---|
| forward, SG/CUDA | 1.52–1.58× | **1.43–1.48×** |
| inverse, SG/CUDA | 1.29–1.35× | **1.11–1.32×** |

Forward improved ~7 %, inverse ~3–14 % (best at small `N`, where the tile kernel
dominates and the twiddle-bounds-check win is proportionally largest).

The SG/CUDA ratio stays flat across `N`, confirming both sides really do run the
same algorithm and that the growth with `N` is just the extra global pass per
doubling.

### Element-wise ciphertext operations (unchanged, same 4 Mi coefficients)

| N | add SeGuRu (µs) | add CUDA (µs) | mul SeGuRu (µs) | mul CUDA (µs) | c×p SeGuRu (µs) | c×p CUDA (µs) | add GB/s |
|---|---|---|---|---|---|---|---|
| 4096 | 65.2 | 64.3 | 66.9 | 65.0 | 48.7 | 46.8 | 1544 |
| 8192 | 65.3 | 63.8 | 66.7 | 65.3 | 49.0 | 46.9 | 1543 |
| 16384 | 65.0 | 63.8 | 66.6 | 65.0 | 48.7 | 47.0 | 1548 |
| 32768 | 65.3 | 63.8 | 66.6 | 65.2 | 48.9 | 47.0 | 1542 |
| 65536 | 65.0 | 63.8 | 66.6 | 65.4 | 48.7 | 46.9 | 1548 |

Element-wise kernels are memory-bound and land within **1–3 %** of CUDA at
~1.54 TB/s, ~79 % of the A100's peak HBM bandwidth. Deliberately untouched.

---

## Why the remaining gap

The honest answer is that after experiments A and C, **the PTX and SASS
instruction counts no longer explain the measured gap**, and we do not have a
complete account of the residual.

**What we ruled out:**

* **Register pressure / occupancy.** `ptxas -arch=sm_80 -O3 -v`: no spills on
  either side. SeGuRu `ntt_forward_tile` uses 56 registers vs CUDA's 48; both
  use 32768 B shared memory at 512 threads, so both are capped at 2 blocks/SM
  (50 % occupancy) by shared memory, not registers. Not a differentiator.
* **The read-only cache** (measurement F): worth 0–5 %, not 45 %.
* **Algorithm.** The SG/CUDA ratio is flat across `N`; the gap is code
  generation, not algorithm choice.

**SASS instruction mix, `ntt_forward_tile` (SeGuRu / CUDA):**

| | SeGuRu | CUDA |
|---|---|---|
| IMAD | 931 | 916 |
| IADD3 | 365 | 344 |
| ISETP | 320 | 288 |
| SEL | 352 | 288 |
| LOP3 | 44 | 5 |
| LDG | 72 | 64 |
| STG | **16** | **8** |
| LDS / STS | identical | identical |
| total | ~2151 | ~1964 |

That is only **1.10× the SASS instructions for 1.43–1.48× the time.** Neither
side is close to issue-bound at 100 % efficiency (both would finish in ~58 µs;
measured 219 µs and 149 µs), so both are stall-bound and SeGuRu stalls more.

**Best remaining explanation** — the *memory pipeline*, not the instruction
count. The 8 data loads and 8 data stores per thread are still split into 16
32-bit `LDG`/`STG`, so SeGuRu issues 88 global memory instructions where CUDA
issues 72 (1.22×), and each generates twice the L1 wavefronts for the same
bytes. This is the one structural difference that survives every experiment
above, and it is the part we could not fix from Rust: see issue 5 below.

This is a hypothesis, not a measurement. `ncu` cannot read performance counters
on this machine (`ERR_NVGPUCTRPERM`), so the L1 wavefront counts that would
settle it are not obtainable here. Stated as the best available explanation, not
as a confirmed result.

---

## Compiler-side issues found (reported, not patched)

### 1. Kernel pointer parameters are emitted `.ptr .align 1`

Consequence: every `u64` global access through a `&[u64]` kernel parameter
becomes two `ld.global.u32` + `shl.b64` + `or.b64` (stores:
`st.global.u32` + `shr.b64` + `st.global.u32`). Shared memory is unaffected —
`ld.shared.u64` is emitted correctly.

* `crates/rustc_codegen_gpu/src/context/to_mir_func.rs:481-509` — kernel
  functions are created here (`gpu.func` at 500-508); **no parameter alignment
  attribute is attached**. The signature comes from `fn_abi_to_fn_type`
  (`to_mir_func.rs:538-545`). NVPTX prints `.align 1` when the LLVM param
  `align` attribute is absent.
* `crates/rustc_codegen_gpu/src/builder/mod.rs:1046-1067` (`mlir_load`) — the
  `llvm.ptr` path attaches `align`, but the memref path (lines 1064-1066) calls
  `melior::dialect::memref::load(ptr, indices, loc)` and **drops the `align`
  argument entirely**. Same for the store path in `store_with_check`
  (`builder/mod.rs:1272-1305`).
* `crates/rustc_codegen_gpu/src/context/ty.rs:291-315` (`pointer_to_mlir_type`)
  types slice params as `memref<Nxi8>`; `type_memref` (`ty.rs:128-147`) has no
  alignment field. `mlir_memref_view` (`builder/mod.rs:890-975`) carries no
  alignment metadata through the retype at 969-975.

**Smallest recommended fix:** propagate `align` on the memref load/store ops at
`builder/mod.rs:1064-1066` and `builder/mod.rs:~1305`, and/or attach a param
`align` attribute in `to_mir_func.rs:481-509`. Experiment C is the empirical
proof that this is the mechanism: switch the element type to a 16-byte-aligned
`U32_4` and the backend immediately emits `ld.global.u64` and
`ld.global.v4.u32`.

### 2. `U32_4` vectorises only for index expressions written textually in a kernel body

The identical expression inside an `#[inline(always)] #[gpu::device]` helper
lowers to four scalar `ld.global.u32`. Workaround used here: `macro_rules!`
instead of device functions (experiment C).

* `crates/rustc_codegen_gpu/src/builder/vector.rs:24-52` — the vector
  load/store path attaches alignment; the generic memref path does not.
* `crates/rustc_codegen_gpu/src/context/ty.rs:430-445` — `mlir_type` treats
  `BackendRepr::SimdVector` as `todo!()`.

### 3. Redundant `ld.global.u64` alongside every `ld.global.v4.u32`

The backend re-reads lanes 0–1 of a `U32_4` it has just loaded (28 wasted loads
per tile kernel). A CSE/forwarding failure in the vector load path. Experiment D
shows working around it from Rust costs more than it saves.

### 4. No read-only / non-coherent load path

SeGuRu never emits `ld.global.nc` / `LDG...CONSTANT`; there is no `__restrict__`
equivalent in `crates/gpu`. Since a `&[T]` kernel parameter is immutable and
Rust's aliasing rules already guarantee it cannot alias any `&mut [T]`
parameter, the backend could emit `ld.global.nc` for every `&[T]` parameter
unconditionally, with no new user-facing API and no new soundness obligation.
Note that measurement F puts the value of this at only 0–5 % *on these kernels*
— it is free correctness-wise but should not be expected to close the gap.

### 5. No way to reinterpret a `[u64]` device buffer as `[U32_4]`

`TensorView::flatten()` (`crates/cuda_bindings/src/mem.rs:268,325`) only goes
vector → scalar (`[U32_4] → [u32]`). There is no inverse. This is what blocks
applying the experiment-C trick to the *data* buffers, which is where the
remaining 16 split `LDG` + 16 split `STG` per thread live — i.e. the most
likely cause of the residual gap. An un-flatten (or a `view_as::<U32_4>()`
returning `Option`, checked for length and alignment) would unblock it.

---

## SeGuRu API notes / limitations hit

1. **`u128` is fully supported in device code.** Barrett and Shoup lower to the
   expected `mul.hi.u64` PTX. No limb decomposition needed.
2. **Shared-memory writes require a static `reshape_map!`.** This rules out the
   textbook variable-distance butterfly and forced the register-resident
   radix-8 formulation — which turned out to be *faster* anyway (3 barriers
   instead of 12), so the safety rule pushed the design in the right direction.
   It is a genuine binding constraint on the kernel structure, but it is **not**
   the reason for the performance gap.
3. **No `u64` vector type** in `gpu::vector` (only `Float2/4/8`, `U32_2/4/8`).
   Combined with issue 1 above, this is why every `u64` global access costs two
   32-bit accesses plus a merge.
4. **Bounds checks were ~⅓ of the tile kernel's PTX** at baseline (216 `setp` /
   288 `selp` vs CUDA's 144 / 144). The polybench mask trick removed all of the
   twiddle-side ones from safe Rust (experiment A). Accesses whose index comes
   from a `reshape_map!` are already free.
5. **`U32_2` is not in the `gpu::*` prelude** — it needs
   `use gpu::vector::U32_2;`. Only `U32_4` is re-exported.
6. **`TensorViewMut` ping-pong across launches must be done by value**
   (`core::mem::swap` / `Option::take`), since views are not `Copy` and the
   launch closures capture them.

## Correctness

11 tests, all exact integer equality (this is exact arithmetic; no tolerances):

* `barrett_and_shoup_match_reference` — device Barrett/Shoup vs the CPU oracle
  over adversarial operands including `q-1`.
* `modulus_constants_are_consistent` — `mu`, `bit`, `pow`, `inv` invariants.
* `cpu_reference_ntt_roundtrips` — the oracle validates itself.
* `gpu_forward_ntt_matches_cpu`, `gpu_inverse_ntt_matches_cpu` — all five ring
  sizes against the CPU NTT.
* `ntt_roundtrip_is_identity` — `INTT(NTT(x)) == x`.
* `ntt_batched_matches_single` — batching does not change results.
* `negacyclic_convolution_matches_schoolbook` — NTT-based product vs
  `O(N^2)` schoolbook multiplication mod `x^N + 1`.
* `gpu_elementwise_matches_cpu`, `gpu_ciphertext_add_sub_negate`,
  `gpu_cipher_plain_mul_matches_cpu`.

No test was weakened or removed during this exercise; all 11 pass in the final
state.

## Running

```bash
source /home/ziqiaozhou/seguru/casestudies-by-agent-v2/env.sh
cd /home/ziqiaozhou/seguru/casestudies-by-agent-v2

cargo test --release -p heongpu-gpu --lib -- --test-threads=1
cargo run  --release -p heongpu-gpu --features bench --bin heongpu-bench
```

Reproducing the PTX / SASS evidence:

```bash
source /home/ziqiaozhou/seguru/casestudies-by-agent-v2/env.sh
cd /home/ziqiaozhou/seguru/casestudies-by-agent-v2
cargo build --release -p heongpu-gpu
ls -t target/release/deps/gpu/heongpu_gpu-*.ptx | head -1     # SeGuRu PTX
nvcc  -arch=sm_80 -O3 -ptx heongpu/cuda/heongpu_ref.cu -o cuda.ptx
ptxas -arch=sm_80 -O3 -v <ptx> -o out.cubin && cuobjdump -sass out.cubin
```

(`ncu` is unusable on this machine — `ERR_NVGPUCTRPERM`. Profiling was done with
`nsys profile --trace cuda` and `nsys stats --report cuda_gpu_kern_sum`.)

The `bench` feature turns on `build.rs`, which compiles `cuda/heongpu_ref.cu`
with `nvcc -O3 -lineinfo -arch=native` into a static library. Every CUDA
runtime call in the reference goes through a `CUDA_CHECK` macro that aborts on
failure — without it a rejected launch silently leaves zeros behind and the
comparison would be meaningless.
