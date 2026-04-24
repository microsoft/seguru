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

One-line change tested: `convert-index-to-llvm{index-bitwidth=32}` in the
MLIR→LLVM lowering pipeline
(`crates/mlir-compile/src/lib.rs:169`).

Hypothesis: if MLIR `index` was lowering to `i64` and forcing u32→u64
promotions, narrowing it to `i32` would eliminate half the bloat.

Result: **byte-identical PTX output.** The critical address arithmetic never
flows through MLIR `index` type — it arrives at the LLVM-dialect level
already materialized as i32/i64. The flag changes nothing about this kernel.

This rules out "promotion width" as the cause.

## 6. Why a fix is out of scope for this workstream

A correct fix requires one of:

- **(a)** Preserve loop structure through MLIR lowering so LLVM LSR can fire
  on still-rolled IR. Requires changing how `reshape_map!` and the GPU-code
  iteration macros lower — they currently produce unrolled scf/linalg ops.
- **(b)** Add a custom LSR-like pass that operates on unrolled straight-line
  code, recognizing the affine stride pattern of adjacent SSA indices and
  rewriting them into base+immediate-offset form. This is a nontrivial new
  pass to maintain.
- **(c)** Change codegen in `rustc_codegen_gpu/src/builder/mod.rs:686`
  (`inbounds_gep_op`) to emit shared base-pointer computations where the
  index expressions share a common term (manual CSE before lowering).

All three are substantial codegen work with regression risk across the entire
codebase. The fast iteration loop is also prohibitive: a full
`cargo clean -p kernelbench-c && cargo build --release` cycle is ~12 min.

Given the current performance floor (0.42× torch for pure GEMM) is within
3× of hand-written CUDA and the Float4 fix already closed the largest gap,
further codegen-level optimization is deferred until there is dedicated
compiler-team bandwidth.

## 7. Artifacts

- PTX before/after (identical): `/tmp/sg_gar.ptx` and
  `/tmp/sg_gar_narrow_pt.ptx` (regenerable by rebuilding `kernelbench-c`).
- nvcc baseline: `/tmp/nvcc_gemm.ptx` (from
  `nvcc -ptx examples/kernelbench-c/cuda/gemm_add_relu.cu`).
- Float4 fix that delivered the bulk of the available win:
  commit `b38e99e5` on branch `kernelbench-skill-test`.

## 8. Honest note on prior session

An earlier session turn proposed "narrow indexing to 32-bit" as a ~10–15%
speedup. This report supersedes that: the proposal was based on counting
`add.s64` instructions without tracing them to their MLIR origin. Once the
experiment was run, PTX was unchanged. The real gap is LSR-not-firing on
fully-unrolled code, which is a much larger structural change than a single
pipeline flag.
