# Proposals: improving `reshape_map!` for per-thread amortization

*Companion to `codegen-address-arithmetic-investigation.md`.*
*Status: design sketches, not implementation. All require codegen work.*

## The tension to resolve

`reshape_map!` composes two concerns into one index expression:
1. **Warp coalescing layout** — threads-adjacent in memory
2. **Per-thread tile addressing** — the 8×8 work items per thread

Today, every `y_thread[(i, j)] = v` re-derives the full linear offset,
producing `mul.wide + add.s64` pairs for each of 64 unrolled stores. nvcc
amortizes the thread portion once and uses immediate offsets for the tile,
but that requires per-thread stride to be 4 bytes — incompatible with
reshape_map's coalescing layout.

Below are four proposals, rankable by impact × effort.

---

## Proposal 1 — Stride-stepping accessor (*best-value, codegen-focused*)

**Idea:** split reshape_map access into a "thread setup once" step and a
"per-tile access" step whose strides are compile-time constants.

```rust
// At kernel entry — invoked once per thread:
let mut y_tile = chunk_mut(y, out_map).open_tile::<TM, TN>();
// y_tile internally carries:
//   - base ptr = y + thread_portion_of_offset (computed once)
//   - stride_i0 = CONST_I0_STRIDE  (known at macro expansion)
//   - stride_i1 = CONST_I1_STRIDE

// Per-store — lowers to `st.global.f32 [base + C0*i + C1*j], v`:
unroll! { for i in 0..8 {
    unroll! { for j in 0..8 {
        y_tile.store(i, j, v);
    }}
}}
```

**Codegen requirement:** `y_tile.store(i, j, v)` must lower to a GEP whose
index is `(i * const_stride_i + j * const_stride_j)` with **no thread-index
recomputation**. The thread-portion offset is baked into `base` once.

**Expected PTX (GEMM epilogue):** 1 mul.wide + 1 add.s64 at entry (base),
then 64 × `st.global.f32 [base + imm]` where `imm` is known at compile time
since i, j, and strides are all compile-time constants.

**Wins:** Matches nvcc's epilogue pattern exactly. Works for ANY kernel
using reshape_map, not just GEMM. Doesn't change layout, so warp coalescing
is preserved.

**Cost:** Moderate. Needs:
- Macro extension to expose per-dim stride as const generics or associated
  consts.
- New `Tile<T, TM, TN>` type in `crates/gpu/src/chunk.rs` with an `index_mut`
  that emits the simple affine form.
- Verification that MLIR preserves the "base + const*i + const*j" structure
  through to LLVM (likely needs a small `expand-strided-metadata` tweak or a
  custom affine op).

**Risk:** Low. Additive API; existing reshape_map usage unchanged.

**Estimated speedup:** 1.5×–2× on pure-GEMM kernels (closes most of the
0.42× → 0.94× gap).

---

## Proposal 2 — Two-pass smem epilogue (*standard CUDA idiom*)

**Idea:** write the 8×8 tile to shared memory in **per-thread-contiguous**
order (stride-4 within thread, amortizes perfectly), then drain smem to
global memory in **warp-coalesced** order.

```rust
let mut smem_tile = gpu::GpuShared::<[f32; 128*128]>::zero();
// ...main matmul fills acc[i][j]...

// Phase 1: thread writes own 8×8 to smem with stride 1 per store
let smem_base = ty * TM * 128 + tx * TN;
unroll! { for i in 0..8 {
    unroll! { for j in 0..8 {
        smem_tile[smem_base + i * 128 + j] = acc[i][j] + bias_reg[j];
    }}
}}
sync_threads();

// Phase 2: warp-coalesced copy smem → y (chunk_mut as today)
let out_map = reshape_map!(...);  // existing coalesced layout
let mut y_thread = chunk_mut(y, out_map);
// stride-1 reads from smem + stride-huge writes to gmem
// smem reads are free; gmem writes are coalesced
```

**Wins:**
- 64 stores to smem become `st.shared.f32 [ptr+0..+28]` (immediate
  offsets, stride 4). Per-thread: 1 base computation + 8 row advances.
- gmem writes still coalesced by reshape_map (unchanged).
- Standard CUDA pattern — cuBLAS, CUTLASS all do this.

**Cost:**
- Extra ~64 KB smem usage for a 128×128 block tile (may exceed per-block
  limit; need fp16 or smaller block).
- Extra `sync_threads()` (cheap: ~20 cycles).

**Risk:** Medium. Needs careful sizing so smem stays under 48 KB / 96 KB
per-SM limit. Half-warp variant (drain 32 rows at a time) avoids this.

**Estimated speedup:** 1.3×–1.8× on GEMM epilogues.

**Meta:** This can be shipped TODAY at the skill-doc level as a recommended
kernel pattern, with no compiler changes. Proposal 1 makes it unnecessary;
proposal 2 works around the codegen limitation.

---

## Proposal 3 — Fast path for "dense-tile-of-known-thread-offset" (*most complex*)

**Idea:** extend `MapReshape` with a compile-time flag
`dense_inner_tile: [TM, TN]` that signals "my last K dimensions form a
contiguous-per-thread tile." When set, the lowering emits:
1. `base = base_of(y) + thread_offset` at kernel entry.
2. `inner_stride_i, inner_stride_j` as `const` from the layout permutation.
3. Accesses via a custom MLIR op that the MLIR→LLVM pass lowers to a pure
   GEP with constant last-two indices.

**Wins:** Fully automatic — user writes `y_thread[(i, j)]` as today, but the
macro rewrites it into a "tile view" when the layout supports it.

**Cost:** High. Requires:
- Macro complexity to detect when the condition holds (last two layout
  positions are `i0, i1` with no thread dims between them).
- Custom MLIR op + lowering.
- Fallback path when the condition doesn't hold.

**Risk:** High. Cross-cutting code changes in macro, IR builder, MLIR
pipeline.

**Estimated speedup:** Same as Proposal 1, but transparent to users.

---

## Proposal 4 — Manual loop peeling + constant folding hint (*lightweight workaround*)

**Idea:** offer an `unroll_tile!` companion macro that pre-computes the 64
offsets as compile-time `const` values rather than as symbolic index
expressions through reshape_map.

```rust
unroll_tile! { y_thread, TM, TN, |i, j| {
    let v = acc[i][j] + bias_reg[j];
    if v < 0.0 { 0.0 } else { v }
}}
```

Internally expands to 64 lines of `y_thread.store_const_offset::<I, J>(v)`
where `I, J` are const generics and the method's MLIR lowering can embed
the strides as literals.

**Wins:** Minimal new codegen work — uses existing APIs, just adds a macro
layer. Could prototype in a week.

**Cost:** Needs `store_const_offset<const I: u32, const J: u32>` method on
`GlobalGroupChunk`.

**Risk:** Low. Additive.

**Estimated speedup:** 1.3×–1.6× (mostly from eliminating redundant thread
offset computation; strides still emerge from macro expansion but LLVM may
fold better with const parameters).

---

## Recommendation

1. **First**, ship Proposal 2 as a skill-doc pattern — zero compiler work,
   validates the upper-bound win. If the measured speedup matches nvcc
   (0.85×+), that's evidence for investing in Proposal 1.
2. **Then**, build Proposal 1 as the principled long-term fix. It
   generalizes beyond GEMM and removes a cross-cutting inefficiency.
3. Skip Proposal 3 unless Proposal 1 ships and proves insufficient.
4. Consider Proposal 4 as a stopgap if Proposal 1 stalls — the sugar is
   useful even after Proposal 1 lands.

## Validation plan (applies to all proposals)

1. Pick `gemm_add_relu` as the canary (well-studied PTX signature).
2. Measure PTX instruction counts before/after.
3. Run full kernelbench-c compare.py to confirm no regressions on the other
   11 problems.
4. Check nvcc parity: raw CUDA avg is 0.77× torch; SG must not over-optimize
   for this specific kernel at the cost of others.
