# Cost of building these case studies with AI agents

This folder was produced end to end by GitHub Copilot CLI agents. This file records
what that cost, so the exercise can be evaluated as an engineering economics
question and not just a technical one.

Regenerate the numbers at any time with:

```bash
./cost.sh            # this session
./cost.sh <SESSION>  # any other session id
```

`cost.sh` reads the Copilot CLI local session store
(`~/.copilot/session-store.db`, table `assistant_usage_events`), which records one
row per model request with token counts and the billed cost in **nano-AIU**
(AI Units). `cost-agents.tsv` maps opaque sub-agent ids to readable names.

Column meanings:

| column | meaning |
| --- | --- |
| `reqs` | model requests (one per assistant turn, including tool-call turns) |
| `fresh_in` | prompt tokens that were **not** served from the prompt cache |
| `cache_read` | prompt tokens served from cache (20x cheaper than fresh input) |
| `cache_write` | prompt tokens written into the cache (1.25x the price of fresh input) |
| `output_tok` | generated tokens, including reasoning |
| `reasoning` | the subset of `output_tok` spent on hidden reasoning |
| `aiu` | billed cost in AI Units |
| `model_min` | wall-clock minutes spent inside model calls |

Note that `fresh_in` is tiny relative to `cache_read`. Long agent sessions are
dominated by re-sending a growing conversation on every turn; the prompt cache is
what makes them affordable, and it is why cost grows roughly with *number of
turns*, not with lines of code produced.

## Model

All agents ran on `claude-opus-5`. A single interactive "lead" agent did the
toolchain work and two case studies itself, and delegated the other three case
studies plus documentation to background sub-agents running concurrently.

## Incremental record

### Snapshot 1 — four of five case studies landed

Taken after: LLVM 20 + `rustc-gpu` installed, a compiler bug fixed, AES and
GPUSorting written and benchmarked by the lead, PolyBench and HEonGPU delivered by
sub-agents, KernelBench still blocked.

| agent | scope | reqs | fresh_in | cache_read | output_tok | AIU | model_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lead (interactive) | toolchain, AES, GPUSorting, orchestration | 184 | 574 821 | 20 572 152 | 183 060 | 1 845.5 | 38.7 |
| polybench-port | 19 PolyBench kernels | 85 | 216 572 | 8 337 816 | 113 238 | 835.3 | 22.3 |
| kernelbench-port | KernelBench (blocked) | 74 | 200 203 | 8 327 968 | 85 389 | 755.0 | 17.0 |
| heongpu-port | NTT + modular arithmetic | 56 | 200 347 | 5 603 932 | 108 735 | 677.2 | 20.9 |
| docs-infra | README, TUTORIAL, verify.sh | 21 | 76 480 | 1 240 430 | 9 359 | 131.4 | 2.2 |
| **total** | | **421** | **1.27 M** | **44.27 M** | **0.50 M** | **4 256.5** | **101.3** |

Wall-clock elapsed at this point was far less than 101 model-minutes of work,
because the four sub-agents ran concurrently: PolyBench took 1 616 s, KernelBench
1 512 s and HEonGPU 1 443 s of wall clock, all overlapping, while the lead worked
on GPUSorting in the same window.

Observations at this snapshot:

* **Delegation paid off on breadth, not on depth.** The three parallel sub-agents
  cost 2 267 AIU combined and delivered PolyBench (19 kernels, 19/19 tests) and
  HEonGPU (11 tests, full benchmark) — but one of the three (KernelBench) burned
  755 AIU and got stuck on a toolchain linking error it could not diagnose without
  the wider context the lead had.
* **The lead's cost is dominated by iteration, not authoring.** Of the lead's 184
  requests, a large fraction went into debugging two runtime faults
  (`InvalidDiversedData` in AES, `CUDA_ERROR_ILLEGAL_ADDRESS` in the sort's
  `reshape_map!`) and into three measured optimisation experiments on the sort.
* **Cost per useful artefact so far:** roughly 4 250 AIU for ~6 kLOC of verified,
  benchmarked, `unsafe`-free GPU Rust plus a compiler fix — about 0.7 AIU per line,
  which is a bad metric, since almost all the cost is in the verify-and-fix loop
  rather than in generating the lines.

### Snapshot 2 — all five case studies complete

Taken after: KernelBench unblocked and finished (22 tests), the PolyBench CUDA
baseline added, three compiler/library bugs fixed, and all documentation written.

| agent | scope | reqs | fresh_in | cache_read | output_tok | AIU | model_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lead (interactive) | toolchain, 3 bug fixes, AES, GPUSorting, orchestration, docs | 228 | 890 278 | 24 016 813 | 228 945 | 2 329.6 | 48.8 |
| kernelbench-port | 21 NN operators + found 2 compiler bugs | 140 | 359 702 | 13 554 421 | 125 720 | 1 216.8 | 26.5 |
| polybench-port | 19 PolyBench kernels | 85 | 216 572 | 8 337 816 | 113 238 | 835.3 | 22.3 |
| heongpu-port | NTT + modular arithmetic | 56 | 200 347 | 5 603 932 | 108 735 | 677.2 | 20.9 |
| polybench-bench | CUDA baseline for 14 kernels | 55 | 216 807 | 5 689 829 | 89 463 | 643.6 | 16.3 |
| docs-infra | README, TUTORIAL, verify.sh | 28 | 94 159 | 1 891 525 | 26 993 | 219.1 | 5.7 |
| **total** | | **592** | **1.98 M** | **59.09 M** | **0.69 M** | **5 921.7** | **140.6** |

Delta from snapshot 1: **+171 requests, +1 665 AIU, +39 model-minutes** to unblock
KernelBench, add the PolyBench CUDA baseline, fix three toolchain bugs and finish
all documentation.

### Snapshot 3 — final

Taken after: all five compiler/library fixes verified by the agents that found
them, the bounds-check tax measured, and all documentation complete.

| agent | reqs | fresh_in | cache_read | output_tok | AIU | model_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lead (interactive) | 238 | 1 001 532 | 24 936 534 | 235 457 | 2 461.4 | 50.4 |
| kernelbench-port | 140 | 359 702 | 13 554 421 | 125 720 | 1 216.8 | 26.5 |
| polybench-bench | 85 | 263 420 | 7 916 084 | 121 426 | 864.0 | 23.1 |
| polybench-port | 85 | 216 572 | 8 337 816 | 113 238 | 835.3 | 22.3 |
| heongpu-port | 56 | 200 347 | 5 603 932 | 108 735 | 677.2 | 20.9 |
| docs-infra | 28 | 94 159 | 1 891 525 | 26 993 | 219.1 | 5.7 |
| **total** | **632** | **2.14 M** | **62.24 M** | **0.73 M** | **6 273.8** | **148.9** |

### Cost per case study

Attributing the lead's shared overhead (toolchain, orchestration, docs) separately
rather than smearing it across the five:

| Case study | Agent(s) | AIU | Tests | Outcome |
| --- | --- | ---: | ---: | --- |
| PolyBench | polybench-port + polybench-bench | 1 699.3 | 19 | Parity on GEMM; measured the bounds-check tax |
| KernelBench | kernelbench-port | 1 216.8 | 22 | 95-100% of HBM roof; found 3 toolchain bugs |
| HEonGPU | heongpu-port | 677.2 | 11 | Parity element-wise, 1.3-1.6x on NTT |
| AES | lead | ~500 (est.) | 6 | Parity with CUDA; retracted the old 13% claim |
| GPUSorting | lead | ~700 (est.) | 9 | 2.2x off CUB, cause measured |
| Shared overhead | lead + docs-infra | ~1 480 (est.) | - | Toolchain, 5 bug fixes, docs, cost tracking |

Totals: **632 requests, 6 274 AIU, 148.9 model-minutes** for ~8 900 lines of
verified, benchmarked, `unsafe`-free GPU Rust, 67 passing tests, five CUDA
baselines, and five toolchain bugs found (three fixed, two reported with a
standing reproducer).

## What this cost profile actually says

* **Cost tracks turns, not lines.** 59.1 M of the 61.1 M prompt tokens were cache
  reads — re-sending a growing conversation. The expensive activity is the
  verify-fix-remeasure loop, not code generation. Anything that shortens that loop
  (a faster test suite, better error messages) is worth more than a model that
  writes better first drafts.
* **The two most valuable results were the most expensive to reach.** KernelBench
  cost the most of any sub-agent (1 217 AIU) and produced no headline speedup — but
  it found two compiler bugs, one of which caused silent wrong answers. Judging a
  port agent on benchmark numbers alone would have scored it lowest.
* **Delegation scaled breadth well and depth badly.** Five sub-agents delivered
  three case studies and a CUDA baseline concurrently. But every sub-agent that got
  genuinely stuck (KernelBench's linker error, PolyBench's `const_alloc`
  collision) needed the lead's cross-cutting context to get unstuck, because both
  were compiler bugs invisible from inside a single crate. Sub-agents were good at
  bounded work and bad at recognising that the toolchain, not their code, was
  wrong.
* **A shared `target/` directory is a real serialisation point.** Concurrent agents
  spent measurable wall-clock time blocked on the cargo file lock.

## What the money bought beyond the benchmarks

* A real bug fix in the SeGuRu compiler: `ctlz`/`cttz` were missing from the
  intrinsic table in `crates/rustc_codegen_gpu/src/builder/intrinsic.rs`, so
  `leading_zeros()`/`trailing_zeros()` panicked the compiler. The previous
  generation of this benchmark worked around it with a 32-iteration serial loop
  executed once per key in the radix sort's hot path.
* Two corrections to the previous generation's published claims (see the AES
  section of `README.md`).
* Quantified answers to "what does SeGuRu's safety cost?", measured rather than
  asserted — see `gpusorting/README.md`.

## Phase 2: optimisation (PTX-guided)

After the suite was complete and correct, three benchmarks still trailed their
CUDA baselines. A second phase attacked those by diffing generated PTX against
the CUDA original — two background agents (`opt-polybench`, `opt-heongpu`) plus
the lead on the radix sort.

Cost of phase 2 alone (delta from the end of the build phase):

| | build phase | +optimisation | delta |
| --- | ---: | ---: | ---: |
| requests | 632 | 965 | **+333** |
| AIU | 6 273.8 | 9 152.0 | **+2 878.2** |
| model-minutes | 148.9 | 210.4 | **+61.5** |

So optimisation cost **46% as much as building the entire five-benchmark suite
from scratch**. That ratio is the headline number of this exercise for anyone
budgeting agent work: getting code correct is cheaper than making it fast, and
not by a small margin.

### Was it worth it?

| Benchmark | Before | After | Cost |
| --- | ---: | ---: | ---: |
| PolyBench `conv3d` | 1.88x | **1.05x** | 832.9 AIU |
| PolyBench stencils (all) | 1.05-1.88x | **1.02-1.13x** | (same agent) |
| HEonGPU forward NTT | 1.52-1.58x | **1.44-1.47x** | 968.4 AIU |
| HEonGPU inverse NTT | 1.29-1.35x | **1.10-1.32x** | (same agent) |
| Radix sort (<=1 Mi keys) | 1.38-1.45x | **1.21-1.31x** | lead time |
| Radix sort (256 Mi keys) | 2.22x | 2.24x | (unchanged) |

PolyBench was the clear win: one agent, 833 AIU, turned the worst result in the
suite into parity. HEonGPU cost *more* and delivered *less* in speed — but it
produced the diagnosis of bug 6 (alignment dropped on the memref path, doubling
every `u64` global access), which is the most valuable single finding in
`FINDINGS.md`. Judged on the ratio alone that agent underperformed; judged on
what it learned, it did not.

### Cost observations specific to optimisation work

* **Negative results are most of the spend.** Of nine experiments across the three
  benchmarks, **five were reverted**. That is not waste — "masking does not work on
  dynamic shared memory" and "const-generic specialisation regressed 9%" are
  written down in the READMEs precisely so nobody spends the tokens again — but it
  means optimisation budgets should assume a majority-failure rate. Build-phase
  work had almost no reverts.
* **The cheapest diagnostic was the one used last.** The radix sort's two failed
  experiments (D and E) were driven by a PTX instruction histogram and cost real
  tokens. A single `ptxas -arch=sm_80 -O3 -v` — which prints the register count in
  one line — would have shown immediately that occupancy was register-limited and
  pointed straight at the change that actually worked. Read the cheap summary
  before building the expensive tool.
* **Agents held their claims to the right standard.** Both optimisation agents
  volunteered caveats against their own headline numbers: `opt-polybench` flagged
  its 0.79x as "not a codegen win" after checking achieved bandwidth, and
  `opt-heongpu` reported that it had refuted its own leading hypothesis about
  `__restrict__`. Every claim in this phase was independently re-run by the lead
  before being written down, and all of them reproduced.

## Phase 3: toolchain fixes, a safe shared-memory constructor, and the no-`unsafe` goal

Phase 2 left the suite fast but not clean: the compiler bugs it uncovered were
still open, `GpuShared::zero()` did not zero (bug 2), and the workaround for it
put 23 `unsafe` blocks into supposedly-safe kernels. Phase 3 closed all of that.

Cost of phase 3 alone (delta from the end of optimisation):

| | +optimisation | +phase 3 | delta |
| --- | ---: | ---: | ---: |
| requests | 965 | 1 271 | **+306** |
| AIU | 9 152.0 | 12 246.4 | **+3 094.4** |
| model-minutes | 210.4 | 263.6 | **+53.2** |

Phase 3 cost roughly what phase 2 did (+3 094 vs +2 878 AIU), and about half of
the original build. Almost all of it was lead time: the work was compiler
changes in `rustc_codegen_ssa` and `rustc_codegen_gpu`, which need one context
holding the MIR, the analysis and the case studies at once, and did not
parallelise into background agents the way porting did.

### The one place agents were used, and why

Removing the radix sort's `Atomic` scatter needed a hand-written
`ScopeUniqueMap`, which is an `unsafe trait` — the exact situation where a wrong
answer is expensive and hard to detect. Rather than decide alone, four agents on
four different models were asked independently whether the safe `chunk!` macro
could express the scatter:

| agent | reqs | AIU | verdict |
| --- | ---: | ---: | --- |
| chunk-consensus (opus-4.5) | 13 | 75.2 | CANNOT |
| chunk-consensus (gpt-5.6) | 12 | 37.9 | CANNOT |
| chunk-consensus (sonnet-4.5) | 13 | 41.0 | CANNOT |
| chunk-consensus (gemini-3.1-pro) | 11 | 17.2 | CANNOT |
| **total** | **49** | **171.3** | unanimous |

**171 AIU — 1.4% of the session — for unanimity on a soundness question.** All
four also independently confirmed the scatter's writes form a genuine
permutation, which is the fact the `unsafe impl` rests on. Cheap insurance, and
a pattern worth reusing: poll several models before writing an `unsafe impl`,
not after.

### What phase 3 bought

| Result | Cost |
| --- | --- |
| 3 compiler bugs fixed (`ctlz`/`cttz`, symbol collisions, `BASE_THREAD_MASK`) | lead |
| bug 2 fixed: safe `GpuShared::init`, incl. an NRVO fix in `rustc_codegen_ssa` | lead |
| `unsafe` in the case studies: 23 -> **0**, at no measurable runtime cost | lead |
| `ret_sync_data` honoured through diverged arguments | lead |
| radix-sort `Atomic` removal | unfinished |

The last row is the honest one: the scatter still uses `Atomic`. The consensus
and the analysis fix landed, but `chunk_mut` still rejects the map on its
receiver, so the ~40% of sort time that the change targets is still on the table.

### Cost observation: the cheap experiment that saved a false result

A benchmark sweep taken while another process held 17% of the GPU showed the NTT
regressing 40% and the sort dropping from 2.11x to 2.60x against CUB. Writing
that up as a regression would have cost far more than measuring it did — the
disproof was two commands: `nvidia-smi --query-compute-apps`, and one re-run on
an idle GPU, which reproduced every baseline number within 0.4%. The generalised
lesson from this session is the same one phase 2 recorded about the AES harness:
**when a number moves, check the measurement before believing the code.** Both
times the measurement was at fault.
