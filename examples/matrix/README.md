### Matrix Multiply

### Runtime costs (TODO: update the table)

| Setup arg(2048, 16)           | Time(s)       |Relative overhead|
|-------------------------------|---------------|--------|
|v1 + no-bound-check	        |1.557330589    |baseline|
|v1 + no-bound-check + fastmath |1.544111987    |--      |
|v1 + select-bound-check	    |1.574405571	|1%      |
|v1 + if-else-trap-bound-check	|6.143659647	|294%    |
|v1 + arith-immed-bound-check	|4.360865864	|180%    |
|v2 + no-bound-check	        |1.64882646	    |6%      |
|v2 + select-bound-check	    |1.648791268	|6%      |
|v2 + if-else-trap-bound-check	|6.081392385	|291%    |
|v2 + arith-immed-bound-check	|9.079779658	|483%    |
|v3 + no-bound-check	        |1.621740153    |3%      |
|v3 + select-bound-check	    |1.578538085	|1%      |
|v3 + if-else-trap-bound-check	|6.155008352	|295%    |
|v3 + arith-immed-bound-check	|3.802663921	|144%    |

### if-else-trap vs arith-immed vs select 

* Both if-else-trap and arith-immed are far more expensive than select-based inplace bound-checking. 

* `if-else-trap` one has a relative stable overhead for both v1 and v2.
This might due to that if-else-trap logic is easier to optimize out and so inner loop does not have bound checks when using if-else-trap.

* arith-immed bound check has a unstable overhead depending on the implementation.
arith-immed-bound-check is never optimized out in PTX optimization. When using arith-immed-bound-check v3 outperforms v2 and v1 since v3 will skip the inner bound check since Rust will skip bound check for the inner loop.


### inner_product_kernel(v1) vs inner_product_kernel(v2)

In theory, if we change inner_product_kernel(v1) to inner_product_kernel(v2), Rust will skip bound checking and so we might get a better performance. However, in practice, due to PTX optimization, both versions will skip bound checking for inner loop. inner_product_kernel(v2) is even more expensive than v1.

v2 will generate less-optimized ptx code, while v1 will generate highly-optimized ptx code.

Though ptx1 uses more registers, the parallelism benefits often outweigh the cost, ptx1 unrolls the loop partially (processing 4 elements per iteration).

In both cases, we did not see the bound-checking for inner loop when calculating `sum` and they are generated in MLIR but erased after some optimization.


## GEMM: head-to-head with cuda-oxide

`sgemm_naive_kernel` and `sgemm_tiled_kernel` in `matmul-gpu` are ports of the
`gemm` and `tiled_gemm` examples from [cuda-oxide][co] (commit `97f8b2b`), kept
at the same shape (1024x1024x1024, `alpha = 1`, `beta = 0`), the same 16x16
tiling, the same launch geometry (grid 64x64, block 16x16) and the same input
fill patterns. `cargo run --release --bin gemm_bench` runs both against cuBLAS
and checks all three against a CPU reference.

[co]: https://github.com/NVlabs/cuda-oxide

### Safety

Both systems type-check the launch and keep global memory race-free, so the
naive kernel is safe in both. They differ on shared memory:

| | cuda-oxide `tiled_gemm` | `sgemm_tiled_kernel` |
|---|---|---|
| tile declaration | `static mut TILE_A: SharedArray<f32, 256>` | `GpuShared::<[f32; 256]>::init(0.0)` |
| cooperative tile load | inside `unsafe` | `chunk_mut`, one element per thread |
| tile read after barrier | inside `unsafe` | shared borrow of the tile |
| `unsafe` blocks in the kernel | 2 | 0 |
| missing / divergent barrier check | none | compile-time |

cuda-oxide's `SharedArray` documents why the `unsafe` is unavoidable there:
"All shared memory access requires `unsafe` because ... multiple threads access
concurrently (potential races)." Its `sync_threads()` is a safe function that
documents, but does not enforce, "all threads in the block must reach the same
barrier".

Chunking removes both `unsafe` blocks. The load phase takes a `chunk_mut` that
hands each of the 256 threads exactly one of the 256 elements, so the writes
cannot overlap; the borrow ends before `sync_threads()`, after which the tile is
read through a shared borrow by the whole block.

### Performance

A100 80GB PCIe, CUDA 13.0, idle GPU, clocks at the 1410 MHz default (not
locked). Both sides use the same window: 10 warm-up launches, then 100 timed
launches per repetition. cuda-oxide's examples ship with a 1-launch warm-up and
5 (`gemm`) or 10 (`tiled_gemm`) timed iterations; widening them to match moved
the result by under 1%.

| kernel | this repo | cuda-oxide | |
|---|---|---|---|
| naive | 2.393 ms / 897 GFLOP/s | 1.804 ms / 1190 GFLOP/s | 1.33x slower |
| tiled | 0.470 ms / 4565 GFLOP/s | 0.487 ms / 4403 GFLOP/s | 1.04x faster |

cuBLAS `cublasSgemm_v2` is 0.129 ms / 16591 GFLOP/s, so the tiled kernel reaches
27.5% of cuBLAS -- the expected ceiling for a plain 16x16 tiled GEMM with no
register blocking, vectorized loads, or tensor cores. All three agree with the
CPU reference to 9.3e-8 (cuBLAS to 9.0e-6, it reassociates).

Two things are worth reading off this table.

**The tiled kernel is the one that matters, and safety is free there.** The
`chunk_mut` version is marginally faster than the `unsafe` one while proving
what the `unsafe` one only asserts. This depends on the shared indices being
visibly in range: `tx` and `ty` are masked to `TILE`, which makes
`ty * TILE + i < TILE * TILE` provable and lets the bounds checks fold. Without
the mask the checks survive in the innermost loop and the kernel drops to
1.709 ms -- 3.6x slower, and clearly behind cuda-oxide. The generated PTX tells
the whole story: with the mask the 16-iteration dot product is fully unrolled to
17 `fma.rn` with no `setp` guards; without it, every `ld.shared` carries a
`setp.lt.u64 ..., 256`.

**The naive kernel still pays for bounds checking.** Its inner loop keeps the
select-based guard on each global load (one `setp` and two `selp` per access),
which costs about a third of its runtime. Hoisting the A row into a checked-once
subslice and walking B with `step_by` does not help -- it measures 3.403 ms,
slower still, for the same reason `v2` loses to `v1` in the table above. The
tiled kernel sidesteps this by reading global memory 16x less often.
