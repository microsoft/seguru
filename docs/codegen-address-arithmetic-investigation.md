# rustc_codegen_gpu: Residual address-arithmetic bloat in GEMM kernels

*Status: Investigation report. No code change landed — see §6.*

## 1. Observation

After Phase G's Float4 global-load fix, SeGuRu-compiled GEMM kernels still
run at ~0.42× torch while raw CUDA with an identical algorithm (same block
shape, same tile size, same unrolling) hits ~0.94×. That 2× residual gap is
not algorithmic; it is visible directly in the PTX.

## 2. PTX evidence: `gemm_add_relu_kernel`, 128×128 GEMM tile, 32×4 block, 8×8 per-thread

Kernel source: `examples/kernelbench-c/src/gemm_add_relu.rs`
(SG←PT variant, post-Float4). nvcc baseline compiled from the twin
`cuda/gemm_add_relu.cu` with `-O3 -arch=sm_86`.

Instruction counts for the single `gemm_add_relu_kernel` entry:

| metric              | SeGuRu | nvcc | notes                              |
|---------------------|-------:|-----:|------------------------------------|
| fma.rn.f32          |    512 |  512 | compute matches                    |
| ld.shared.v4.f32    |     32 |   32 | smem vectorization matches         |
| ld.global.v4.f32    |      2 |    2 | Phase G fix applied                |
| st.global.f32       |     64 |   64 | same total stores                  |
| **add.s64 (addr)**  | **71** |  13  | 5.5× more pointer arithmetic       |
| **mul.wide.u32**    | **69** |   5  | one mul.wide per indexed access    |

## 3. What nvcc does that SG does not

Sample from `nvcc -ptx` on the 8-row × 8-col unrolled store epilogue:

```ptx
; compute row stride once
mul.wide.s32 %rd21, %r_rowstride, 4;   ; rd21 = row-stride in bytes
; row 0
st.global.f32  [%rd26],      %f...;    ; base pointer
st.global.f32  [%rd26+4],    %f...;    ; immediate byte offsets
st.global.f32  [%rd26+8],    %f...;
...
st.global.f32  [%rd26+28],   %f...;    ; 8 stores with immediate +0..+28
; row 1
add.s64 %rd27, %rd26, %rd21;           ; advance by one row stride
st.global.f32  [%rd27],      %f...;
st.global.f32  [%rd27+4],    %f...;
...
```

This is textbook **LoopStrengthReduction**: the 8×8 unrolled store block is
recognized as having two induction-like dimensions (row-stride, column-stride
= 4 bytes). nvcc emits **7 add.s64** (one per row advance) + **0 muls for the
inner 8**, using PTX's immediate-offset addressing mode for the column.

Sample from SG's PTX at the same location:

```ptx
st.global.f32 [%rd125], %f888;
mul.wide.u32 %rd126, %r172, 4;     ; fresh mul+add for EVERY store
add.s64      %rd127, %rd1, %rd126;
st.global.f32 [%rd127], %f890;
mul.wide.u32 %rd128, %r174, 4;
add.s64      %rd129, %rd1, %rd128;
st.global.f32 [%rd129], %f892;
...  (× 64 stores, each with its own mul.wide + add.s64)
```

SG emits **one full `mul.wide + add.s64` pair per store** (64 pairs for the
64-store epilogue). No immediate offsets, no incremental stride adds.

## 4. Root cause hypothesis

The loop that computes the 8×8 tile is written via SG's `reshape_map!` macro
which produces **compile-time unrolled, straight-line code**. Each of the 64
"iterations" becomes a distinct MLIR/LLVM SSA basic block fragment with its
own index variable (`%r172`, `%r174`, `%r176`, ...).

To LLVM's LoopStrengthReduce pass, this isn't a loop — there is no induction
variable to strength-reduce. LSR operates on LLVM IR `loop` constructs; with
the loop fully unrolled in the frontend, there is nothing for it to find.

By contrast, nvcc (clang) receives the CUDA C `#pragma unroll` code, runs LSR
on the *still-rolled* IR, recognizes the two affine induction patterns, emits
the linear stride increments, and **then** unrolls. The order matters: LSR
must run before full unrolling.

## 5. Experiment run (this branch)

### 5.1 `convert-index-to-llvm{index-bitwidth=32}`

One-line change tested: `convert-index-to-llvm{index-bitwidth=32}` in the
MLIR→LLVM lowering pipeline
(`crates/mlir-compile/src/lib.rs:169`).

Hypothesis: if MLIR `index` was lowering to `i64` and forcing u32→u64
promotions, narrowing it to `i32` would eliminate half the bloat.

Result: **byte-identical PTX output.** The critical address arithmetic never
flows through MLIR `index` type — it arrives at the LLVM-dialect level
already materialized as i32/i64. The flag changes nothing about this kernel.

This rules out "promotion width" as the cause.

### 5.2 Swap inner/outer loop order in epilogue

Current code uses `y_thread[(j, i)]` with `for i outer, for j inner` where
`y_thread` is `chunk_mut(y, out_map)` with layout
`[i0, t0, t1, i1, t2, t3]`. User index 0 (= j) maps to i0 (outermost, stride
524288); user index 1 (= i) maps to i1 (middle, stride 256).

Hypothesis: swap to `y_thread[(i, j)]` so that inner loop walks i1's smaller
stride, potentially exposing the affine pattern to LLVM.

Result: PTX total instruction count unchanged (71 add.s64, 69 mul.wide.u32,
64 st.global.f32 — identical to baseline). More importantly, **correctness
broken**: output elements landed at wrong addresses because the layout is
asymmetric in user indices. The `(j, i)` convention is load-bearing for
coalesced writes across the warp.

### 5.3 Direct `y[...]` indexing with hoisted row base

Attempted to bypass `y_thread` and write to `y` directly with a computed flat
offset.

Result: **blocked by the type system.** `GpuGlobal<'_, [f32]>` explicitly
disallows `Deref` (`crates/gpu/src/global.rs:102`) by design, so direct
slicing is unavailable. All writes must go through `chunk_mut`.

## 6. Why the problem is fundamental to the reshape_map pattern

The three experiments together reveal the structural issue. `reshape_map!` is
designed so that adjacent threads in a warp write to adjacent memory
locations (coalescing), which forces each thread's 8×8 tile to be
non-contiguous in memory. The stride between any two stores within a single
thread's epilogue is at least 256 (one block-y worth), and typically much
larger for the outer dimension.

PTX `st.global.f32 [ptr+imm]` immediate-offset addressing only amortizes
pointer arithmetic for **stride-4-byte adjacent** stores (the nvcc inner
`[rd26+0 .. rd26+28]` pattern). Any larger stride requires a fresh pointer
computation. Thus even with a perfect loop-order or CSE pass, the
reshape_map'd thread tile **cannot** compress to nvcc's epilogue pattern
without sacrificing per-warp coalescing — which would be much worse overall.

**nvcc gets both** because CUDA C programmers write their 8×8 tile
contiguously per-thread and let nvcc figure out warp-level scheduling. SG's
`reshape_map!` bakes the warp-coalescing layout into the thread's index
computation, which is cleaner abstractly but costs amortization.

## 6.5. LLVM IR evidence (2026-04-25, this session)

Final pre-PTX LLVM IR inspected for `gemm_add_relu_kernel` at
`target/release/deps/gpu/libkernelbench_c-*gpu.gpu.gpu.gpu.bc`.

Two adjacent unrolled stores in the epilogue look like:

```llvm
; store (i, j=0):
%402 = or disjoint i32 %401, %392         ; thread + col_base  (%401, %392 already hoisted)
%403 = add i32 %402, %396                 ; + i*row_stride      (%396 per-row, hoisted)
%404 = zext i32 %403 to i64               ; <-- FRESH per store
%405 = shl nuw nsw i64 %404, 2            ; <-- FRESH per store (*4 bytes)
%406 = getelementptr i8, ptr %6, i64 %405 ; <-- FRESH per store
store float %.sroa.01133.0, ptr %406

; store (i, j=1):
%409 = or disjoint i32 %392, %396
%410 = or disjoint i32 %401, 1            ; lid j-offset
%411 = add i32 %410, %409
%412 = zext i32 %411 to i64               ; <-- FRESH per store
%413 = shl nuw nsw i64 %412, 2            ; <-- FRESH per store
%414 = getelementptr i8, ptr %6, i64 %413
store float %.sroa.01132.0, ptr %414
```

Crucial correction to §4: **LLVM does CSE correctly.** The shared i32 values
(`%401` thread-base, `%392` column tile-base, `%396` row-stride contribution)
are hoisted. What it fails to do is share the `zext → shl → gep` chain across
adjacent stores, because each store's i32 offset differs (by the compile-time
`j`), and without a loop induction variable LLVM never synthesizes a pointer
chain.

nvcc sidesteps this entirely: it materializes a 64-bit row pointer
(`add.s64 rd_row, rd_base, rd_row_stride`) once per `i` and then emits
`st.global.f32 [rd_row + j*4]` with `j` as a PTX immediate. This is what the
5.5× difference in `add.s64` counts reflects.

## 7. Actual fix options (all substantial)

- **(a)** Add an `open_tile()` / `row_mut()` primitive on `GlobalGroupChunk`
  that returns a pre-offset 1D sub-chunk whose base is a 64-bit pointer.
  Within the unroll, each store then becomes `st.global [row_base + imm*4]`.
  Requires: new chunk type, new `ScopeUniqueMap` split method, macro-level
  support to emit `map_thread` and `map_local` separately. Memory safety is
  preserved because the split is equivalent to today's `map(lid, tid)`.
- **(b)** Preserve loop structure through MLIR lowering so LLVM LSR can fire
  on still-rolled IR. Requires changing how the GPU-code iteration macros
  lower — they currently produce fully unrolled scf/linalg ops.
- **(c)** Custom MLIR pass that recognizes the affine stride pattern of
  adjacent SSA indices produced by `reshape_map!` + full unrolling and
  rewrites them into a single base + strided stores.
- **(d)** Codegen-level peephole in
  `rustc_codegen_gpu/src/builder/mod.rs:686` (`inbounds_gep_op`) that, when
  it sees N adjacent GEPs with shared i32 offsets differing only by
  compile-time constants, emits a shared 64-bit base + immediate-offset
  stores.

All four are substantial codegen work with regression risk across the entire
codebase. The fast iteration loop is also prohibitive: a full
`cargo clean -p kernelbench-c && cargo build --release` cycle is ~12 min.

**ROI estimate (2026-04-25):** The address-arith bloat is concentrated in the
unrolled epilogue (64 stores). Total per-thread work in `gemm_add_relu` is
~512 FMAs (main loop) + 64 stores (epilogue); stores take roughly 2-4× FMA
latency, so the epilogue is ~15% of runtime. Fully eliminating epilogue
address-arith cost would yield ≤15% speedup, insufficient to close the 2.5×
gap to torch. The remaining gap must live in the main loop (register
pressure, shared-memory bank conflicts, occupancy) rather than epilogue
addressing.

Decision: **defer this work.** Next investigation target is the main loop
and occupancy, not further epilogue codegen.

## 8. Artifacts

- PTX before/after width experiment (identical): `/tmp/sg_gar.ptx` and
  `/tmp/sg_gar_narrow_pt.ptx`.
- PTX after loop-order swap (identical totals, wrong output):
  `/tmp/sg_gar_swap_pt.ptx`.
- nvcc baseline: `/tmp/nvcc_gemm.ptx`.
- Float4 fix that delivered the bulk of the available win:
  commit `b38e99e5` on branch `kernelbench-skill-test`.

## 9. Honest note on prior sessions

An earlier session turn proposed "narrow indexing to 32-bit" as a ~10–15%
speedup. That proposal was based on counting `add.s64` instructions without
tracing them to their MLIR origin. The `index-bitwidth=32` experiment in §5.1
produced byte-identical PTX, disproving it.

A subsequent turn claimed "PTX cannot fold ... regardless of compiler effort"
because `reshape_map!` composes warp-coalescing and per-thread addressing.
§6.5's LLVM IR inspection corrects this: LLVM CSE does share the thread/row
portions correctly; the remaining cost is per-store 32→64 widening that would
disappear if the 64-bit row pointer were hoisted. That would require an
`open_tile()`-style API change, now documented as option 7(a).

Based on the ROI estimate in §7 (~15% ceiling, insufficient for the 2.5× gap),
this work is deferred and the investigation focus pivots to the main loop.
