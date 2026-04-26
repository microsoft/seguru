# CUDA to SeGuRu Porting Progress

This document keeps the design rationale, implementation progress, and
empirical benchmark history for CUDA-to-SeGuRu porting. Keep reusable rules and
copyable implementation recipes in `docs/cuda-to-seguru-porting-skill.md`; keep
measured results, phase history, and current optimization targets here.

## Branch status

- `agent-poc-v2` is the slim branch. It was reset to the pre-invasive
  `open_tile` base and currently keeps the source-level retained changes:
  Float4 global-load ports, K-major shared-memory layout for KernelBench-C
  GEMM/matmul kernels, checked Float4 tensor views, and
  `nvvm_launch_bound(16, 16, 1, 2)` annotations matching the 16x16 launch
  geometry.
- `codegen-i32-addr-arith` preserves the full reference branch with the
  experimental codegen work and refreshed benchmark snapshots.
- The active slim branch intentionally does not restore the open-tile proposal,
  row-view API, generated row-offset traits, or codegen investigation docs.

Fresh slim-branch KernelBench-C check on A100 80GB (CUDA 13.2,
`DISABLE_GPU_BOUND_CHECK=true`) after restoring K-major shared layout without
restoring open-tile/core row-view APIs:

| Problem | PyTorch eager | Raw CUDA | SeGuRu | SeGuRu-from-CUDA |
|---|---:|---:|---:|---:|
| gemm_mul_lrelu | 8433.9 us | 8955.6 us | 9082.2 us | 9083.5 us |
| conv_relu_hardswish | 6560.1 us | 6661.6 us | 6598.4 us | 8662.9 us |
| matmul_mish_mish | 8527.0 us | 9102.9 us | 9077.7 us | 9078.0 us |
| matmul_scale_resadd | 30832.5 us | 32767.6 us | 33215.9 us | 33389.9 us |
| gemm_scale_htanh_gelu | 15411.4 us | 17594.8 us | 17661.2 us | 17662.6 us |
| matmul_sigmoid_sum | 18721.0 us | 2093111.6 us | 1863824.2 us | 1863721.2 us |
| gemm_relu_div | 8641.5 us | 8961.8 us | 9156.2 us | 9152.9 us |
| conv_relu_biasadd | 7759.3 us | 103038.4 us | 136556.0 us | 136575.6 us |
| matmul_sub_mul_relu | 8654.6 us | 8962.1 us | 9096.9 us | 9084.9 us |
| gemm_add_relu | 8556.6 us | 8961.1 us | 9026.5 us | 9082.9 us |
| matmul_div_gelu | 8572.7 us | 8975.0 us | 9095.9 us | 9096.7 us |
| matmul_min_subtract | 4744.8 us | 7524.9 us | 7555.1 us | 7550.7 us |

Summary: all three implementation arms are 12/12 correct. Average speedup vs
PyTorch eager is 0.76x for SeGuRu, 0.77x for raw CUDA, and 0.74x for
SeGuRu-from-CUDA. The corrected ablation is: `open_tile().row_mut()` was
performance-neutral in isolation, but K-major shared-memory layout was not; the
branch must keep K-major layout while still avoiding the invasive core row-view
APIs.

## Reference benchmark snapshot

The broader benchmark refresh below comes from the preserved full reference
branch and is retained as historical progress context. Ratios are SeGuRu / raw
CUDA; lower is better.

| Suite | Reference | Scope | Result |
|---|---|---|---|
| PolyBenchGPU (19 overlapping kernels) | `benchmarks/run_polybench_comparison.sh` | classic kernels | corrected transfer contracts; raw geomean **1.172x**; launch-normalized geomean **1.052x**; normalized 12/19 no more than 5% slower, 12/19 no more than 10% slower |
| KernelBench-B full | raw custom CUDA | 20 L1 kernels | 20/20 correct; geomean **1.190x** hand SeGuRu, **1.008x** SeGuRu-from-CUDA |
| KernelBench-C full | raw custom CUDA | 12 fused GEMM/conv/epilogue kernels | 12/12 correct; geomean **0.995x** hand SeGuRu, **1.005x** SeGuRu-from-CUDA |

Original tracked summaries from that refresh:

- `benchmarks/polybench_comparison_results.txt`
- `examples/kernelbench-b/results/reported_comparison.txt`
- `examples/kernelbench-c/results/reported_comparison.txt`

## Design and implementation progress

- **Core feasibility:** CUDA-to-SeGuRu translation is partly mechanical
  (kernel signatures, intrinsics, launch plumbing) and partly semantic
  (`chunk_mut` mappings, reduction strategy, shared-memory layout, vector width).
- **Automation direction:** the most reliable workflow is to write or identify a
  raw CUDA kernel first, then translate that algorithm and launch geometry into
  SeGuRu. Direct PyTorch-to-SeGuRu can work, but tends to miss vectorization,
  row-reduction parallelism, or GEMM tiling unless those choices are explicit.
- **Benchmark contract corrections:** LU and GramSchmidt PolyBench comparisons
  were corrected so timings no longer measure mismatched host/device transfers.
- **Re-port refresh:** PolyBench, KernelBench-B, and KernelBench-C were refreshed
  with the current rules. KernelBench-B/C SeGuRu-from-CUDA ports are now nearly
  raw-CUDA parity overall; remaining gaps are concentrated in reductions/norms
  and a few PolyBench multi-kernel pipelines.

## Remaining parity targets

| Suite | Problem | Current ratio vs CUDA | Note |
|---|---|---:|---|
| PolyBench | gramschm | 1.384x | launch-normalized; multi-kernel orthogonalization pipeline |
| PolyBench | corr | 1.325x | launch-normalized; normalization + covariance-style reduction |
| PolyBench | covar | 1.297x | launch-normalized; covariance-style reduction |
| PolyBench | lu | 1.297x | launch-normalized; sequential panel/trailing-update pipeline |
| PolyBench | atax | 1.239x | launch-normalized; matrix-vector with column-stride traversal |
| KernelBench-B | l2_norm | 1.164x | norm/reduction gap |
| KernelBench-B | l1_norm | 1.160x | norm/reduction gap |
| KernelBench-B | mse_loss | 1.077x | remaining Float4 reduction gap |
| KernelBench-B | sum_dim | 1.065x | reduction gap |
| KernelBench-B | layer_norm | 1.059x | norm gap |
| KernelBench-C | gemm_mul_lrelu | 1.030x | minor GEMM epilogue/codegen gap |
| KernelBench-C | matmul_scale_resadd | 1.028x | minor GEMM epilogue/codegen gap |
| KernelBench-C | matmul_min_subtract | 1.025x | minor GEMM epilogue/codegen gap |
| KernelBench-C | gemm_relu_div | 1.023x | minor GEMM epilogue/codegen gap |
| KernelBench-C | gemm_add_relu | 1.021x | minor GEMM epilogue/codegen gap |

## Historical notes moved from the skill doc

The sections below are retained for provenance. They are intentionally not in the
skill reference because they describe experiments, measurements, and project
progress rather than rules future porting agents should load by default.

## Case Study: PyTorch LayerNorm (algorithm > idioms)

Porting PyTorch's `vectorized_layer_norm_kernel` revealed a critical nuance about
how SeGuRu reaches CUDA parity. Three SeGuRu variants were written from scratch
against a hand-tuned CUDA reference on M=8192, N=1024:

| Variant | Algorithm | SeGuRu idioms | Time | vs CUDA |
|---|---|---|---|---|
| CUDA reference | Fused Welford + float4 + warp shfl | (hand-written CUDA) | 62.7 µs | 1.00× |
| SeGuRu vectorized | Fused stats + Float4 | `reshape_map`, `ThreadWarpTile::redux`, subslice | **65.9 µs** | **1.05×** |
| SeGuRu naive | 3-pass (mean, var, out), scalar, block-per-row | tree reduction only | 82.3 µs | 1.31× |
| SeGuRu "idiomatic" | 3-pass scalar, warp-per-row | `reshape_map`, `redux`, `ldcs` | 89.7 µs | 1.43× |

### The surprise: idioms ≠ performance

The "idiomatic" variant — using every SeGuRu performance pattern (warp-cooperative
reduction, `ldcs`, `reshape_map` strided output, subslice rows) — was **slower than
the naive 1:1 port**. Both run the same 3-pass algorithm; the idiomatic version
loses because one warp per row gives only 1024 blocks of parallelism vs the naive's
8192 blocks.

### The rule

**Port the algorithm first, then the idioms.** A clean SeGuRu rendering of a bad
algorithm will not reach parity. What unlocks parity is:

1. **Fuse passes** — compute mean and variance in a single traversal (local `s` +
   `sq` accumulators, two `warp.redux` calls). Halves global memory traffic for
   memory-bound kernels.
2. **Vectorize loads** — use `Float4` for contiguous reads/writes. Host side builds
   `Vec<Float4>`; kernel declares `x: &[Float4]`; index as `x[i][k]` for lanes.
3. **Then apply SeGuRu idioms** — `reshape_map!` for per-thread output slots,
   `ThreadWarpTile::redux` for cross-lane reductions, subslice for row access.

Only when the algorithm is right do SeGuRu's safety abstractions become free.

### Implication for automated porting

A mechanical CUDA→SeGuRu translator will land at 1.3–2.0× overhead even with
perfect idiom usage, because the algorithmic opportunities (pass fusion, vector
width selection, Welford online updates) require **semantic** rewrites, not
syntactic translation. Automation should:

- Detect multi-pass statistics (mean+var, max+sum) and fuse them.
- Detect contiguous float loads of stride 4/8 and lift them to `Float4`/`Float8`.
- Detect warp-reducible patterns and emit `ThreadWarpTile::redux`.
- Leave block/grid geometry choices to a tuning pass — warp-per-row vs block-per-row
  is workload-dependent.

### Fused-stats warp kernel template

```rust
#[gpu::cuda_kernel]
pub fn layernorm_vectorized(
    x: &[Float4], gamma: &[Float4], beta: &[Float4], y: &mut [Float4],
) {
    let warp = ThreadWarpTile::<32>;
    let warps_per_block = warp.meta_group_size();
    let row = block_id::<DimX>() * warps_per_block + warp.subgroup_id();
    let lane = warp.thread_rank();

    const N4: u32 = N / 4;
    let x_row = &x[(row * N4) as usize..((row + 1) * N4) as usize];

    // ONE pass: accumulate sum and sumsq together.
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    let mut i = lane;
    while i < N4 {
        let v: Float4 = x_row[i as usize];
        for k in 0..4 { let vk = v[k]; s += vk; sq += vk * vk; }
        i += warp.size();
    }
    let sum = warp.redux(ReduxAdd, s);
    let sumsq = warp.redux(ReduxAdd, sq);
    let inv_n = 1.0 / (N as f32);
    let mean = sum * inv_n;
    let rstd = (sumsq * inv_n - mean * mean + 1e-5).rsqrt();

    // Strided Float4 output via reshape_map.
    let mut y_chunk = chunk_mut(y, reshape_map!(
        [N4 / 32] | [32, warps_per_block * grid_dim::<DimX>()] => layout: [t0, i0, t1]
    ));
    let mut slot = 0u32;
    let mut i = lane;
    while i < N4 {
        let v = x_row[i as usize];
        let g = gamma[i as usize];
        let b = beta[i as usize];
        let mut out = Float4::new([0.0; 4]);
        for k in 0..4 { out[k] = (v[k] - mean) * rstd * g[k] + b[k]; }
        y_chunk[slot] = out;
        i += warp.size();
        slot += 1;
    }
}
```

Source: `examples/bench-layernorm/` (all three variants + host harness); CUDA
reference at `benchmarks/cuda/layernorm_pytorch.cu`.

## Case Study: KernelBench L1 skill-doc stress test

Five KernelBench Level-1 problems were ported to SeGuRu using **only patterns
documented above**. All five compiled on the first attempt and produced correct
output. Performance vs PyTorch (A100):

| Problem | Shape | SeGuRu | PyTorch | Ratio |
|---|---|---|---|---|
| 19_ReLU | 4096×16384 | 350 µs | 325 µs | **1.08×** |
| 21_Sigmoid | 4096×16384 | 353 µs | 325 µs | **1.09×** |
| 23_Softmax (dim=1) | 4096×4096 | 208 µs | 98 µs | 2.12× |
| 40_LayerNorm (vectorized variant) | 8192×1024 | 66 µs | — | 1.05× vs CUDA ref |
| 1_SquareMatmul | 4096×4096 | 31510 µs | 7308 µs | 4.31× |

### What worked

- **Elementwise kernels reach PyTorch parity** with the documented
  `chunk_mut(MapContinuousLinear::new(1))` + bounds-guarded global-thread-id
  pattern. No shared memory, no vectorization, no idiom tricks — just the
  basic elementwise template.
- **All correctness checks passed** on the first build — sync analysis,
  bounds guards, and chunk-based writes caught nothing the author had missed,
  meaning the documented rules suffice to produce compilable code for these
  patterns.

### Gaps surfaced

**Gap 1 — fused multi-pass reductions** (softmax at 2.12×, layernorm-idiomatic
at 1.43×): the doc now has a "port the algorithm first" note in the LayerNorm
case study, but the general lesson is worth stating once as a Golden Rule:
*any reduction pattern (max+sum, mean+var, logsumexp) should be fused into a
single pass over global memory*. The 3-pass "natural translation" landing
around 2× is predictable.

**Gap 2 — register tiling for GEMM** (matmul at 4.31×). The skill doc's
GEMM-family guidance stops at the 16×16 shared-memory tile pattern. Reaching
competitive GEMM performance requires:
- Larger block tiles (e.g., 128×128) with each thread computing a **register
  tile** (e.g., 8×8 outputs held in registers across the K-loop).
- Double-buffered shared-memory loads (software pipelining).
- For f32, `tf32` via WMMA/tensor cores to approach cuBLAS.

None of this is in the doc. The existing `bench_gemm_tiled` in
`examples/bench/src/main.rs` has a comment: *"Register tiling (each thread
computes NxM outputs) is the next optimization axis"* — that axis is
undocumented.

**Practical rule for now:** For large dense f32 matmul, the doc already points
at cuBLAS via `cublasSgemm_v2`. Use it. Treat user-written SeGuRu matmul as
appropriate only for small sizes or non-standard shapes where cuBLAS doesn't
fit. The 4.3× gap is structural, not a skill-doc failure — but the doc should
say so explicitly instead of implying a 1.7-2.0× tile-only ratio is tight.

### Revised Golden Rule

Promoted to the top-level rules list as **Rule #9** ("Port the algorithm before the idioms"). See the top of this doc.

### Reproducing

- SeGuRu: `cargo run --release -p kernelbench`
- PyTorch: `python3 examples/kernelbench/python/run_torch_baseline.py`

Both crates at `examples/kernelbench/`.

## Case Study: KernelBench L1 (phase B — LLM-driven generation)

Three fresh LLM sub-agents (Claude Sonnet, one-shot, access only to this skill
doc + the phase-A examples) were asked to port:

- **LeakyReLU** — elementwise with scalar param
- **Tanh** — elementwise with intrinsic
- **RMSNorm** — strided reduction over 4D `(B, C, H, W)` tensor along dim=1

Results (after a 1-line host-side fix noted below):

| problem | torch | LLM-generated SeGuRu | speedup |
|---------|-------|----------------------|---------|
| LeakyReLU (4096×393216) | 7.68 ms | 8.35 ms | 0.92× |
| Tanh (4096×393216)      | 7.67 ms | 8.54 ms | 0.90× |
| RMSNorm (112×64×512×512)| 24.2 ms | 16.5 ms | **1.47×** |

All 3 compiled first try. All 3 kernels produced correct output. The LLM
actually *beat* PyTorch on RMSNorm by picking a sensible two-kernel
decomposition (one reduction kernel writing an `inv_rms` auxiliary buffer,
one elementwise apply kernel) — faster than PyTorch's fused-but-temporary path.

### The one observed LLM failure mode: host-side `copy_to_host`

**2 of 3 sub-agents produced a silent correctness bug**: after running the
kernel, they dropped the `TensorViewMut` *without* calling
`d_out.copy_to_host(&mut h_out)`. This yielded all-zeros output without any
warning. The skill doc does show the right pattern in "Host-Side Patterns /
Basic launch" but it's a one-line comment-less call in a code block; LLMs
focused on kernel design missed it.

**Golden Rule #8 (host-side):**

> After a kernel writes to a `TensorViewMut<[T]>` backed by a host vector,
> you **must** call `d_out.copy_to_host(&mut h_out).unwrap()` before reading
> or persisting `h_out`. `new_tensor_view` snapshots the host data to device
> at construction; there is no automatic readback on drop. This is the single
> most frequent mistake observed in LLM-generated ports — both Tanh and
> LeakyReLU agents hit it while writing otherwise-correct kernels.

### Takeaway for phase B

Even with a thorough skill doc, the *host plumbing* is what trips up LLMs,
not the GPU code itself. Kernel authorship is well-covered; the missing
piece is a "host recipe" section with explicit `read → device → launch →
sync → copy_to_host → write` scaffolding that the LLM can copy verbatim.

### Phase B.full results — safety vs. raw CUDA (same LLM)

Same model (Claude Sonnet sub-agent, one-shot) was asked to port the
same three problems to (a) SeGuRu using this skill doc and (b) raw CUDA
with `torch::Tensor` + `PYBIND11_MODULE`. Both arms ran against the
same PyTorch reference on identical input tensors. All six generated
kernels use float4 vectorization + grid-stride + (SeGuRu: `reshape_map!` /
`chunk_map`; CUDA: explicit `float4*` reinterpret).

| Problem   | PyTorch    | SeGuRu         | Raw CUDA       | overhead |
|-----------|-----------:|---------------:|---------------:|---------:|
| leaky_relu| 7.68 ms    | 8.35 ms (0.92×)| 8.05 ms (0.95×)|    −3.7% |
| tanh      | 7.67 ms    | 8.55 ms (0.90×)| 7.97 ms (0.96×)|    −7.3% |
| rms_norm  | 24.25 ms   |16.50 ms (1.47×)|13.27 ms (1.83×)|    −24%  |

Correctness: **3/3 for both arms** (max-abs-err ≤ 8e-6). On memory-bound
elementwise ops the safety layer is invisible; on reductions SeGuRu
currently pays ~20% vs. hand-managed shared memory in raw CUDA, but
*both* arms still beat PyTorch. The conclusion for KernelBench-style
LLM codegen: SeGuRu is a viable safe target whose `fast_N` scores
should track raw CUDA closely on memory-bound L1, with a measurable
but not disqualifying gap on reduction kernels.

### Phase B.10 results — scaled to 10 KernelBench L1 problems

Same setup, expanded to ten problems spanning elementwise / reduction /
softmax / norm categories. Same Claude Sonnet sub-agent ports each problem
to both SeGuRu (with this skill doc) and raw CUDA (with the symmetric
`docs/cuda-raw-kernel-skill.md`). Driver: `examples/kernelbench-b/python/compare2.py`.

| Problem    | PyTorch eager | SeGuRu           | Raw CUDA         | SeGuRu←CUDA      |
|------------|--------------:|-----------------:|-----------------:|-----------------:|
| leaky_relu |       640.9µs |   698.1µs (0.92×)|   668.3µs (0.96×)|   642.0µs (1.00×)|
| tanh       |       654.0µs |   707.2µs (0.92×)|   663.7µs (0.99×)|   637.5µs (1.03×)|
| relu       |       653.7µs |   696.9µs (0.94×)|   669.1µs (0.98×)|   642.3µs (1.02×)|
| sigmoid    |       655.5µs |   703.2µs (0.93×)|   664.5µs (0.99×)|   641.0µs (1.02×)|
| gelu       |       663.6µs |   728.9µs (0.91×)|   665.7µs (1.00×)|   641.2µs (1.04×)|
| softmax    |       229.5µs |   324.8µs (0.71×)|   244.6µs (0.94×)|   308.6µs (0.74×)|
| layer_norm |       261.5µs |   338.8µs (0.77×)|   215.3µs (1.21×)|   242.6µs (1.08×)|
| rms_norm   |     24269.7µs | 16574.7µs (1.46×)| 13268.3µs (1.83×)| 13342.5µs (1.82×)|
| sum_dim    |       177.5µs |   225.2µs (0.79×)|   152.9µs (1.16×)|   163.4µs (1.09×)|
| l2_norm    |       452.9µs |   299.5µs (1.51×)|   208.4µs (2.17×)|   242.8µs (1.87×)|

Aggregate (`fast_N` = pct of problems with speedup ≥ N× vs PyTorch eager):

| Arm          | source          | correct | fast_1 | fast_2 | avg speedup |
|--------------|-----------------|--------:|-------:|-------:|------------:|
| SeGuRu       | PyTorch ref     |   10/10 |    20% |     0% |       0.99× |
| Raw CUDA     | PyTorch ref     |   10/10 |    40% |    10% |       1.22× |
| SeGuRu←CUDA  | the `.cu` files |   10/10 |    80% |     0% |       1.17× |

Three ports of the same 10 problems, same LLM (Claude Sonnet sub-agents,
one-shot, in parallel):
- **SeGuRu** from the PyTorch reference + this skill doc.
- **Raw CUDA** from the PyTorch reference + `docs/cuda-raw-kernel-skill.md`.
- **SeGuRu←CUDA** mechanically translates each `.cu` kernel to SeGuRu
  (see `examples/kernelbench-b/src/from_cuda/*.rs`) using this skill doc
  only as a reference for how to spell CUDA primitives in SeGuRu; no
  redesign.

Key takeaways:
- **Correctness parity holds across all three arms** — 10/10 one-shot.
- **`SeGuRu←CUDA` essentially matches raw CUDA** (1.17× vs 1.22× avg, 80%
  beat-PyTorch vs 40%). SeGuRu's safety layer costs very little when the
  LLM mirrors a good CUDA kernel — the big gaps in the first SeGuRu
  column came from the LLM picking a worse strategy when designing
  from PyTorch, not from SeGuRu's runtime.
- **`sum_dim` and `l2_norm` row-reduction improvement from the skill
  doc update**: the first-pass SeGuRu port produced `sum_dim=2981.8µs`
  (0.06× PyTorch) and `l2_norm=1675.3µs` (0.27×). After adding the
  "Row-Reduction Strategy" section explicitly calling out the
  1-thread-per-row pitfall, a re-port produced `sum_dim=225.2µs`
  (13.2× speedup) and `l2_norm=299.5µs` (5.6× speedup). The skill doc
  meaningfully changes LLM codegen behavior.
- **The residual gap on `softmax` / `layer_norm`** is in the SeGuRu
  runtime's cross-warp reduce scaffolding, not strategy — `SeGuRu←CUDA`
  shows `layer_norm` matches raw CUDA within ~13% (1.08× vs 1.21×).

### Recommended automation pipeline: two-stage PyTorch → CUDA → SeGuRu

The `SeGuRu←CUDA` column above is not an accident of evaluation order —
it is the output of a **two-stage LLM pipeline**:

1. Stage 1 (design): LLM reads PyTorch reference + `docs/cuda-raw-kernel-skill.md`
   → emits a raw-CUDA `.cu` file. This is where the hard decisions
   happen: thread geometry, vectorization width, reduction strategy,
   block/grid shape.
2. Stage 2 (translation): LLM reads the Stage-1 `.cu` + this skill doc
   → emits a SeGuRu `.rs`. This is mechanical: `float4*` ↔ `Float4`,
   `__shfl_down_sync` ↔ `warp.redux(ReduxAdd,·)`, `__shared__` ↔
   `GpuShared<[f32; N]>`, `__syncthreads()` ↔ `sync_threads()`.

Stage 1 is the same task as the "Raw CUDA" column. Stage 2 is the same
task as the "SeGuRu←CUDA" column. End-to-end, the pipeline matches Raw
CUDA's avg speedup within 5% (1.17× vs 1.22×) while producing safe code.

**Why this beats the direct PyTorch → SeGuRu route (1.17× vs 0.99×):**
The CUDA intermediate pins down three decisions an LLM regularly
under-thinks when designing a SeGuRu kernel from scratch:

1. **Vectorization**. The direct route wrote scalar `fn kernel(x: &[f32])`
   for every elementwise problem; the translate route faithfully copies
   the `.cu`'s `float4` reinterpret into `Float4` + scalar tail, which
   is ~60µs faster on memory-bound elementwise at [2048, 65536].
2. **Thread geometry for reductions**. First-pass direct-SeGuRu
   `sum_dim` was 1-thread-per-row (2981µs / 0.06×). The CUDA arm was
   always 1-block-per-row with warp-shuffle reduce; translating that
   landed at 163µs (1.09×) on the first attempt.
3. **Launch config**. The `.cu` files pin block size and grid clamp;
   the translate route copies these rather than guessing.

The direct PyTorch → SeGuRu route can close most of the gap once the
skill doc explicitly names the pitfalls it tends to fall into (the
"Row-Reduction Strategy" section moved `sum_dim` from 0.06× → 0.79×
on its own). But as a default automation recipe, the two-stage
pipeline is less sensitive to skill-doc gaps: the intermediate CUDA
source acts as a compact, unambiguous specification of the desired
kernel, making Stage 2 a near-deterministic mapping.

Recommended usage:
- **For greenfield SeGuRu codegen from ML framework ops**: use the
  two-stage pipeline. Treat the intermediate `.cu` as a disposable
  artifact (you can delete it after Stage 2 completes — it's only a
  prompt for the translator). This gets you ~1.17× PyTorch eager
  with 80% beat-rate on L1 and 10/10 correctness.
- **For porting existing CUDA codebases to SeGuRu**: Stage 2 alone
  suffices. The `.cu` IS your intermediate.
- **For direct PyTorch → SeGuRu** (simpler setup, one LLM call): expect
  ~1.0× PyTorch on average; works well for elementwise, lags on
  reductions unless the skill doc explicitly addresses strategy.

Methodology note: `torch.compile` baseline disabled in this run (TypeError
on lambda wrap under our torch version). Re-enable in a follow-up by
wrapping problems in `nn.Module` instead of `lambda`.

### CUDA gotcha learned: `__shfl_down_sync` partial masks deadlock

While porting `l2_norm` the LLM wrote a cross-warp reduce of the form
`if (tid < BLOCK/32) { acc += __shfl_down_sync(0xffffffff, acc, o); }`.
On A100 (sm_80) this hangs the kernel forever (100% GPU util, never
returns). The mask `0xffffffff` declares all 32 lanes participating, but
only `BLOCK/32` lanes entered the branch — undefined behavior.

**Fix pattern**: have all 32 lanes of warp 0 enter the branch, with
inactive lanes contributing 0:

```cpp
if (threadIdx.x < 32) {
    float acc = (threadIdx.x < BLOCK/32) ? warp_sum[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if (threadIdx.x == 0) /* write */;
}
```

Now documented in `docs/cuda-raw-kernel-skill.md` so future raw-CUDA
ports avoid this pitfall.

### Isolated per-launch overhead: SeGuRu matches raw CUDA

To answer whether SeGuRu's FFI path is a constant tax on benchmarks that
dominate via many small launches, we measured empty-kernel launch cost
on A100 (block=1, thread=1, 1000 iterations, median):

| Launch path | Per-launch |
|---|---|
| `torch.cpp_extension` (raw CUDA) → `cuLaunchKernel` | **5.08 µs** |
| SeGuRu (Rust) → `cuLaunchKernel` | **4.48 µs** |
| `torch.nn.functional.relu` (reference) | 6.36 µs |

SeGuRu's FFI path is actually **0.6 µs faster** than torch's C++ extension
path — both are essentially the driver's `cuLaunchKernel` cost, with no
measurable Rust safety-wrapper tax. This rules out "per-launch FFI
overhead" as an explanation for the SeGuRu-vs-CUDA gap on the softmax-class
problems. Any residual gap is in kernel-body codegen (register pressure,
`__expf` vs `GPUDeviceFloatIntrinsics::exp`, compiler vectorization) or in
algorithm, **not** in the launch path.

The measurement harness lives at `examples/kernelbench-b/python/launch_overhead.py`
(SeGuRu dispatch) + `examples/kernelbench-b/src/empty.rs` and
`examples/kernelbench-b/cuda/empty.cu` (empty kernels). Re-run when
changing `crates/gpu_host`, Rust-side launch macros, or cuLaunchKernel
config plumbing.

### Phase C results — KernelBench Level 2 (fused operators)

Same methodology extended to 8 Level 2 problems (fused GEMM/Conv +
elementwise epilogue). Three sub-agents per problem, dispatched in
parallel, one-shot — all 8 ported by different agents independently.
Driver: `examples/kernelbench-c/python/compare.py`. All timings on
A100 80GB, 50 iter median, µs.

| Problem               | PyTorch eager | CUDA raw     | SeGuRu      | SeGuRu←CUDA |
|-----------------------|--------------:|-------------:|------------:|------------:|
| gemm_mul_lrelu        |        8520.6 |       8952.0 |    134460.1 |     41161.6 |
| matmul_mish_mish      |        8554.7 |       9093.7 |    134460.7 |     41395.8 |
| gemm_relu_div         |        8637.3 |       8964.5 |    134458.1 |     41171.2 |
| gemm_scale_htanh_gelu |       15430.4 |      17577.4 |    268304.6 |     82267.7 |
| matmul_scale_resadd   |       30923.9 |      32938.1 |    536396.1 |    156872.5 |
| conv_relu_hardswish   |        6561.6 |       6706.3 |     19216.1 |     19271.7 |
| conv_relu_biasadd     |        7771.5 |     102937.2 |    336531.9 |    336534.0 |
| matmul_sigmoid_sum    |       18686.1 |    2094149.5 |   1831788.1 |   1822243.5 |

Aggregate:

| Arm          | source          | correct | avg speedup vs PyTorch |
|--------------|-----------------|--------:|-----------------------:|
| SeGuRu       | PyTorch ref     |    8/8  |                 0.09×  |
| Raw CUDA     | PyTorch ref     |    8/8  |                 0.72×  |
| SeGuRu←CUDA  | the `.cu` files |    8/8  |                 0.17×  |

Key takeaways:
- **Correctness holds at Level 2 too** — 8/8 across three arms on fused
  GEMM/Conv+elementwise. The automation pipeline is viable for
  multi-op fusion tasks.
- **Hand-written raw CUDA matches cuBLAS TF32 within ~6% on the five
  GEMM problems.** This is surprising only if you assume PyTorch eager
  GEMM is using tensor cores at FP32 — it is not; `nn.Linear` on FP32
  tensors dispatches to cuBLAS with TF32 acceleration, which loses
  precision but only ~2× the FLOPS of a well-tuned FP32 SGEMM.
- **SeGuRu←CUDA gives a consistent 3.2–3.4× speedup over
  SeGuRu←PyTorch on the GEMM class** (134→41 ms, 268→82 ms, 536→156 ms).
  Same story as Phase B: the CUDA-intermediate route pins down tile
  geometry (BM=BN=128, BK=8, 8×8 register tile, `#pragma unroll`),
  which an LLM designing directly from `x @ W.T` tends to under-think.
  The direct SeGuRu arm defaults to 1-output-per-thread 16×16 tiling.
- **Convolution is still the weak spot**. On `conv_relu_hardswish` the
  two SeGuRu arms are within 1% — when the inner loop is small (8 Cin ×
  3×3 = 72 FMA) the port is memory-bound and SeGuRu keeps up with raw
  CUDA (6.7 ms vs 19.2 ms — within 2.9×). On `conv_relu_biasadd` with
  64 Cin, both SeGuRu arms hit 336 ms because both agents chose the
  same 1-output-per-thread direct convolution without shared-memory
  input tiling; raw CUDA at 103 ms pays for proper tiling. This is
  an LLM strategy gap, not a SeGuRu ceiling — add a "Convolution
  Tiling" section to this doc to close it.
- **`matmul_sigmoid_sum` is pathological for the two-stage pipeline.**
  The CUDA arm is *slower* than PyTorch here (2.1s vs 18.7ms) because
  PyTorch decomposes to `@` (cuBLAS-TF32) + pointwise + sum, hitting
  tensor cores. The hand-written fused kernel (K=N=32768, M=128) pays
  for non-coalesced W reads that cuBLAS's specialized kernels avoid.
  Lesson: **fusion is not always the right strategy**; for
  small-M / huge-K-N GEMMs with a row-reduce epilogue, a two-kernel
  (cuBLAS + reduce) plan wins. A future skill-doc update should name
  this anti-pattern.
- **Absolute vs PyTorch eager**: the average `SeGuRu←CUDA` speedup of
  0.17× looks bad on Level 2 compared to Phase B's 1.17×, but this is
  a Level 2 characteristic: Level 2 problems are GEMM-dominated, and
  any FP32 GEMM that isn't using TF32 tensor cores loses ≈3–4× against
  cuBLAS automatically. The meaningful comparison is **SeGuRu←CUDA vs
  Raw CUDA** — both written by the same LLM with the same GEMM
  algorithm, varying only the safety layer. That ratio is 4.6× on
  GEMM (41 ms vs 9 ms) — SeGuRu's shared-memory tiling currently pays
  a real cost for strided-write limitations (see "Honest limitation on
  COMPUTE phase" section earlier in this doc). Closing this gap
  requires a skill-doc expansion around SeGuRu shared-memory access
  patterns for register-tiled GEMM; the CUDA reference kernels in
  `examples/kernelbench-c/cuda/*.cu` are the target to match.

Artifacts:
- 8 raw-CUDA `.cu` files in `examples/kernelbench-c/cuda/`
- 8 SeGuRu←PyTorch kernels in `examples/kernelbench-c/src/*.rs`
- 8 SeGuRu←CUDA ports in `examples/kernelbench-c/src/from_cuda/*.rs`
- PyTorch sources in `examples/kernelbench-c/problems/*.py`
- Driver: `examples/kernelbench-c/python/compare.py`

### Phase C.2: skill-doc intervention — direct SeGuRu arm catches up

After adding the "## GEMM / Matmul Recipe" and "## Convolution Recipe"
sections above (prescribing BM=BN=128 / BK=8 / 8×8 register tile for
GEMM, shared-mem input tile for Conv), we re-dispatched 6 parallel
LLM sub-agents to re-port the direct SeGuRu-from-PyTorch arm only
(no change to the raw CUDA or SeGuRu←CUDA arms). The before/after:

| Problem               | SeGuRu v1 | SeGuRu v2 | improvement | SeGuRu←CUDA |
|-----------------------|----------:|----------:|------------:|------------:|
| gemm_mul_lrelu        |  134460.1 |   41168.4 |       3.27× |     41168.6 |
| matmul_mish_mish      |  134460.7 |   41397.7 |       3.25× |     41402.7 |
| gemm_relu_div         |  134458.1 |   41163.6 |       3.27× |     41163.9 |
| gemm_scale_htanh_gelu |  268304.6 |   82262.0 |       3.26× |     82263.1 |
| matmul_scale_resadd   |  536396.1 |  156859.2 |       3.42× |    156855.5 |
| conv_relu_biasadd     |  336531.9 |  268559.1 |       1.25× |    268562.8 |

Aggregate after intervention:

| Arm          | correct | avg speedup vs PyTorch | delta |
|--------------|--------:|-----------------------:|------:|
| SeGuRu v1    |    8/8  |                 0.09×  |       |
| SeGuRu v2    |    8/8  |                 0.17×  | +1.9× |
| SeGuRu←CUDA  |    8/8  |                 0.17×  |    —  |
| Raw CUDA     |    8/8  |                 0.72×  |    —  |

Key finding: **the direct SeGuRu-from-PyTorch arm now matches the
two-stage SeGuRu←CUDA arm** on all 5 GEMM problems (differences < 1%).
The CUDA intermediate (Stage 1) is no longer needed for the GEMM class
once the skill doc contains a prescriptive tile recipe — LLMs reliably
pick the right tile geometry when it's spelled out explicitly.

For convolution (historical C.2): the shared-mem input tile closed 25%
of the `conv_relu_biasadd` gap (336→268 ms), but still lost badly to raw
CUDA. This was later superseded by Phase R: matching raw CUDA's direct
16x16 one-output-per-thread geometry plus row-sliced 3x3 FMAs reached
raw-CUDA parity.

Methodology note: no other code changes between v1 and v2. Same
cuBLAS/PyTorch versions, same CUDA arm, same SeGuRu←CUDA arm, same
hardware. The delta is entirely attributable to the skill-doc
additions driving different LLM code choices.

Takeaway for the two-stage pipeline thesis: **a sufficiently
prescriptive skill doc collapses the two-stage pipeline into a
single-stage one for well-known compute patterns** (GEMM, and
presumably conv with the next iteration). The two-stage route
remains valuable as a discovery mechanism — the CUDA-written
kernels tell you what recipe to prescribe — but once the recipe
is in the doc, Stage 1 becomes redundant.

### Phase C.3: skill-doc intervention — reduction-class

Same methodology applied to the KernelBench Phase B reduction-class
problems (`softmax`, `layer_norm`, `sum_dim`, `l2_norm`). New sections
added to this doc:

- `## Softmax Recipe` — single-kernel fused online softmax, explicit
  step-by-step template, list of anti-patterns (2-kernel stats+apply).
- `## Row-Reduction Strategy` → "Always vectorize when D % 4 == 0
  (Float4 loads)" subsection with before/after timings per kernel.
- `## Row-Reduction Strategy` → "one scalar per block output" pitfall
  (Grid→Block→Thread scope chain for per-block scalar writes).
- `## Launch Config & Occupancy`, `## Shared-Memory Bank Conflicts`,
  `## Warp Divergence`, `## Debugging Checklist` — systematic
  coverage imported from the KrxGu CUDA skill repo.

Re-dispatched one agent per problem; each agent read ONLY the updated
skill doc and the PyTorch source spec (not the `from_cuda/` reference).

| Problem    | SeGuRu baseline | SeGuRu v3 | speedup | SeGuRu←CUDA |
|------------|----------------:|----------:|--------:|------------:|
| softmax    |         324.3 µs|  308.1 µs |   1.05× |    308.1 µs |
| layer_norm |         337.9 µs|  243.3 µs |   1.39× |    243.1 µs |
| sum_dim    |         225.4 µs|  163.6 µs |   1.38× |    163.6 µs |
| l2_norm    |         299.6 µs|  243.1 µs |   1.23× |    243.4 µs |

Aggregate direct-SeGuRu speedup vs PyTorch eager: **1.02× → 1.19×**,
exactly matching the SeGuRu←CUDA arm (1.19×). The two-stage pipeline
has now been collapsed to one stage for the entire reduction class,
reproducing the Phase C.2 result seen earlier for GEMM.

Key driver of the reduction improvements: **Float4 loads**. All
from_cuda reduction ports used Float4; none of the direct-PyTorch
ports did until the doc made it the default. Softmax is the one
exception where Float4 doesn't help (the online max+sum state update
is sequential per element), which is why softmax closed the 5% gap
with just the Softmax Recipe while the other three needed Float4.

Residual gap (raw CUDA still ~15–20% faster than SeGuRu): intrinsic
SeGuRu codegen overhead (u32 vs usize, ptxas optimizations, etc.),
not an LLM/skill-doc issue.

### Phase T refresh: corrected raw-CUDA parity status

Refreshed benchmark command set:

- PolyBench: `benchmarks/run_polybench_comparison.sh`
- KernelBench L1: `examples/kernelbench/python/driver.py`
- KernelBench-B: `examples/kernelbench-b/python/compare2.py`
- KernelBench-C: `examples/kernelbench-c/python/compare.py`

**PolyBenchGPU (19 overlapping CUDA/SeGuRu kernels)**:

| Metric | Result |
|---|---:|
| Correctly run rows | 19/19 |
| Raw geomean SeGuRu/CUDA | 1.172x |
| Launch-normalized geomean SeGuRu/CUDA | 1.052x |
| Raw no more than 5% slower than CUDA | 8/19 |
| Raw no more than 10% slower than CUDA | 10/19 |
| Launch-normalized no more than 5% slower than CUDA | 12/19 |
| Launch-normalized no more than 10% slower than CUDA | 12/19 |

Tracked raw outputs: `benchmarks/cuda_results.txt` and
`benchmarks/seguru_results.txt`; summary table:
`benchmarks/polybench_comparison_results.txt`. The summary now reports both raw
wall-clock ratios and a launch-normalized long-running-kernel proxy using
side-specific empty-launch measurements. Normalized ratios below 1.0x mean
launch overhead dominates enough that subtraction can flip the ratio; the raw
table remains the authoritative wall-clock result for those workloads. The LU
and GramSchmidt transfer contracts are corrected, so the remaining normalized
gaps are no longer dismissed as host-copy mismatches.

**KernelBench-B/C after the from-CUDA refresh**:

| Suite | Correct | SeGuRu/CUDA geomean | SeGuRu-from-CUDA/CUDA geomean |
|---|---:|---:|---:|
| KernelBench-B | 20/20 | 1.190x | 1.008x |
| KernelBench-C | 12/12 | 0.995x | 1.005x |

**Remaining parity targets after launch-normalized PolyBench update**:

PolyBench ratios below are launch-normalized; KernelBench ratios are raw
from-CUDA ratios because those suites are already long-running/fused enough that
launch overhead is not the dominant signal.

| Suite | Problem | Current ratio vs CUDA | Note |
|---|---|---:|---|
| PolyBench | gramschm | 1.384x | multi-kernel orthogonalization pipeline |
| PolyBench | corr | 1.325x | normalization + covariance-style reduction |
| PolyBench | covar | 1.297x | covariance-style reduction |
| PolyBench | lu | 1.297x | sequential panel/trailing-update pipeline |
| PolyBench | atax | 1.239x | matrix-vector with column-stride traversal |
| KernelBench-B | l2_norm | 1.164x | norm/reduction gap |
| KernelBench-B | l1_norm | 1.160x | norm/reduction gap |
| KernelBench-B | mse_loss | 1.077x | remaining Float4 reduction gap |
| KernelBench-B | sum_dim | 1.065x | reduction gap |
| KernelBench-B | layer_norm | 1.059x | norm gap |
| KernelBench-C | gemm_mul_lrelu | 1.030x | minor GEMM epilogue/codegen gap |
| KernelBench-C | matmul_scale_resadd | 1.028x | minor GEMM epilogue/codegen gap |
| KernelBench-C | matmul_min_subtract | 1.025x | minor GEMM epilogue/codegen gap |
| KernelBench-C | gemm_relu_div | 1.023x | minor GEMM epilogue/codegen gap |
| KernelBench-C | gemm_add_relu | 1.021x | minor GEMM epilogue/codegen gap |

The biggest automation lesson from the refresh is conservative: leave kernels
that are already near parity alone, and only re-port a kernel when a repeatable
recipe applies. For KernelBench-B that meant Float4 row/reduction paths; for
KernelBench-C it meant replacing the stale 14x14 convolution patch tile with the
direct 16x16 raw-CUDA geometry.
