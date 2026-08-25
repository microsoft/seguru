# PolyBench/GPU on SeGuRu

A from-scratch port of the 19 [PolyBench/GPU](https://github.com/sgrauerg/polybenchGpu)
benchmarks to [SeGuRu](../../), the safe-Rust GPU toolchain. Everything lives in a
single crate (`polybench-gpu`, lib name `polybench_gpu`), one module per kernel.

**There is no `unsafe` anywhere in this crate.**

## Running the tests

```bash
source ../env.sh                       # LLVM 20 + CUDA on PATH
cd ..                                  # casestudies-by-agent-v2
cargo test --release -p polybench-gpu --lib
```

Every module has a `#[cfg(test)]` test that runs the GPU implementation and a
plain-Rust CPU reference on the same input and compares them with a *relative*
error tolerance (see `common::assert_close`). The whole suite runs in ~11 s.

```
running 19 tests
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.10s
```

## Kernels

| Module | Benchmark | Test size | Tol. | Strategy |
| --- | --- | --- | --- | --- |
| `gemm` | `C = alpha*A*B + beta*C` | 256×192×320 | 1e-4 | shared-memory tiled, 64×64 CTA tile, 4×4 register micro-tile |
| `twomm` | `E = A*B; F = E*D` | 192×256×128×192 | 1e-4 | two `gemm_kernel` launches |
| `threemm` | `E=A*B; F=C*D; G=E*F` | 192/192/128/256/128 | 5e-4 | three `gemm_kernel` launches |
| `doitgen` | `A[r][q][p] = sum_s A[r][q][s]*C4[s][p]` | 32×32×128 | 1e-4 | `(r,q)` flattened → one `gemm_kernel` |
| `syrk` | `C = alpha*A*A^T + beta*C` | n=256, m=320 | 1e-4 | tiled, NT operand loads |
| `syr2k` | `C = alpha*A*B^T + alpha*B*A^T + beta*C` | n=m=256 | 1e-4 | tiled, four shared tiles |
| `covar` | column means → centre → `D^T D` | 512×256 | 1e-4 | thread-per-variable reductions + tiled TN product |
| `corr` | mean/stddev → standardise → Gram | 512×256 | 1e-4 | reuses `covar_symmat`, unit diagonal kernel |
| `atax` | `y = A^T (A x)` | 1024×768 | 1e-4 | warp-per-row `Float4` reduction, then thread-per-column |
| `bicg` | `q = A p`, `s = r^T A` | 1024×768 | 1e-4 | same two-phase shape as `atax` |
| `mvt` | `x1 += A y1`, `x2 += A^T y2` | n=1024 | 1e-4 | warp-per-row + thread-per-column |
| `gesummv` | `y = alpha*A*x + beta*B*x` | n=1024 | 1e-4 | one fused warp-per-row reduction over both matrices |
| `conv2d` | 3×3 stencil | 1024×512 | 1e-5 | 4 rows per thread, sliding register window |
| `conv3d` | 3×3×3 stencil | 128³ | 1e-5 | one point per thread, 3D grid |
| `jacobi1d` | 2 × `tsteps` sweeps | n=8192, t=20 | 1e-4 | step + copy kernels |
| `jacobi2d` | 2 × `tsteps` 5-point sweeps | 512², t=10 | 1e-4 | step + copy kernels |
| `fdtd2d` | 3 kernels × `tmax` | 512², t=20 | 1e-4 | `ey`/`ex`/`hz` launches per step |
| `lu` | LU without pivoting | n=512 | 1e-4 | 2 launches per `k`: row/col snapshot + rank-1 update |
| `gramschm` | modified Gram-Schmidt QR | 512×256 | 1e-3 | 4 launches per `k` |

## Design decisions

### One crate, shared building blocks

The previous port used 19 separate crates with no code sharing. Here everything is
one crate, so the pieces that actually matter are written once:

* `lib.rs::warp_sum` — a five-step butterfly `gpu::shuffle!(xor, ..)` reduction, used
  by every mat-vec kernel and by `gramschm`.
* `gemm::gemm_kernel` — launched directly by `twomm`, `threemm` and `doitgen`.
* `covar::covar_symmat` — launched by `corr`.
* `common.rs` — deterministic input generation (`seq`), 2D/1D zero padding
  (`pad2`/`unpad2`/`pad1`) and the relative-error comparator.

### Tiled GEMM geometry

The tiled kernels use a 16×16 block (256 threads) owning a 64×64 output tile, i.e. a
4×4 register micro-tile per thread, with a K-slab of 16 staged through two
`GpuShared<[f32; 16*64]>` buffers. That gives 8 flops per shared-memory load and
keeps register pressure low enough to stay off the spill path.

Three shared-tile load variants are needed, and they differ *only* in the
`reshape_map!` used for the staging chunk:

* **NN** (`gemm`) — `A` rows and `B` rows, both read along the unit-stride axis.
* **NT** (`syrk`, `syr2k`) — both operands are rows of the same `N×M` matrix.
* **TN** (`covar`, `corr`) — both tiles are columns of an `N×M` matrix, so the
  staging loads run along `M` (unit stride) and the transpose happens in shared memory.

This is the main reason the shared-memory tiling is cheap to reuse: the compute
loop is identical in all three, only the loader map changes.

### Padding vs. tail predicates

Wherever the arithmetic is affine and zero-safe (all the GEMM-family kernels), the
host driver pads the operands to exact multiples of the tile geometry with
`common::pad2` and crops afterwards with `unpad2`. The grid then maps exactly onto
the data and there are no tail predicates in the kernel at all.

Padding is *not* usable for two classes of kernel:

* **Stencils** — a zero halo would move the boundary and change the answer, so the
  drivers `assert!` that the sizes are exact multiples of the block geometry and the
  boundary condition is folded into the *stored value* rather than into control flow
  (`out[0] = if interior { v } else { out[0] }`).
* **`covar`/`corr`** — padded rows would be centred to `-mean[j]` and would then
  contribute to the symmetric matrix. Exact multiples are required instead.

### `chunk_mut` and divergence

SeGuRu rejects `chunk_mut` inside divergent control flow. Two idioms handle
everything here:

1. **Predicate the store, not the chunk.** `chunk_mut` is executed by all threads;
   only the assignment is guarded, either with a `if cond { out[0] = v }` or, more
   often, by selecting the value (`out[0] = if cond { v } else { out[0] }`) so the
   store itself is unconditional too.
2. **Shrink the map instead.** For "one thread of the warp/CTA writes one scalar",
   the map's *target* dim for the thread axis is set to 1 — e.g.
   `reshape_map!([1] | [(32, 1), 1, 8, gy] => layout: [i0, t0, t1, t2, t3])` — so only
   lane 0 owns a slot, and the actual store is wrapped in `if tx == 0`.

### Sequential outer loops (`lu`, `gramschm`)

Both have a sequential `k` loop, and both need values that live in *other* threads'
chunks (the pivot row/column, the current `R[k][*]`). SeGuRu will not let a kernel
hold the same buffer as `&[f32]` and `&mut [f32]`, which is exactly the aliasing
that the naive one-kernel formulation requires. Routing the shared values through
small read-only scratch buffers filled by a preceding launch removes the aliasing
*and* the race, at the cost of one extra launch per step:

* `lu`: `lu_rowcol` (snapshot pivot row/column) → `lu_update` (rank-1 update, with
  the pivot-row write-back folded in).
* `gramschm`: `gs_norm` → `gs_q` → `gs_r` → `gs_update`.

`gs_norm` also shows the offset form of the macro, writing a single scalar to
`R[k][k]`:
`reshape_map!([1] | [(256, 1), 1] => layout: [i0, t0, t1], offset: k * m + k)`.

### Vectorisation and unrolling

Row reductions over the unit-stride axis (`atax`, `bicg`, `mvt`, `gesummv`) load
`gpu::Float4` so each warp issues 128-byte transactions. All loops over local arrays
use `crunchy::unroll!`, and all indices are `u32`.

### Indexing that LLVM can prove in range

Memory-bound kernels that index a `&[f32]` parameter directly (`conv2d`,
`conv3d`, `jacobi1d`, `jacobi2d`, `mvt`, `atax`, `bicg`) open with

```rust
let total = ni * nj;                                  // u32, exact extent
if total == 0 || a.len() < total as usize { return; }
let a = &a[..total as usize];
```

and then index through `crate::ix(idx, total - 1)`. This is what makes the
per-access bounds check fold away without any `unsafe`; the reasoning, the
variants that did *not* work, and the measured effect are in the optimisation
log below. The clamp never changes a result — every index these kernels produce
is already in range — and the benchmark verifies that element-wise against CUDA
on every run.

### Numerical tolerances

The GPU code is built with fast-math and FTZ enabled, and the reduction orders
differ from the CPU reference by construction, so exact equality is not meaningful.
Single GEMMs and stencils match to 1e-4/1e-5 relative error; `threemm` needs 5e-4
because three chained f32 GEMMs compound the reassociation error, and `gramschm`
needs 1e-3 after 256 sequential orthogonalisation steps.

## Performance vs hand-written CUDA

`cuda/polybench_ref.cu` contains a hand-written CUDA implementation of 14 of the
19 kernels. Each one mirrors its Rust counterpart exactly: same algorithm, same
tile sizes, same block shape, same work per thread, same shared-memory layout —
including the 16-way bank conflict in the GEMM-family staging store, which is a
property of the `reshape_map!` layout and was reproduced deliberately so that the
comparison isolates *codegen* rather than *algorithm choice*. Where the CUDA side
deviates it says so in a comment.

### Methodology

* **Kernel-only timing.** Host allocation and H2D/D2H copies are excluded. The
  CUDA side brackets a loop of *N* launches with `cudaEvent`s; the SeGuRu side
  brackets the same loop with `Instant` + `ctx.sync()`. Both therefore amortise
  launch overhead identically.
* **Warm-up then measure.** Every kernel gets warm-up iterations, then *N* timed
  iterations (*N* scaled by problem size, from 200 down to 3); the reported number
  is the mean per-iteration time.
* **Correctness gates the timing.** Before any time is printed, the kernel is
  re-run from freshly uploaded inputs on both sides and the outputs are compared
  with a relative infinity-norm error, `max|sg - cuda| / max(max|cuda|, 1)`. A row
  whose error exceeds its tolerance prints `MISMATCH` instead of a time and the
  binary exits non-zero. All 42 rows below verified; the worst observed error is
  4.0e-6 (`syr2k` at 2048), and 19 rows are bit-identical.
* **Same fast-math on both sides.** SeGuRu emits PTX with `USE_FAST`/`USE_FTZ`, so
  the CUDA reference is compiled `-O3 -arch=native --use_fast_math`. Not doing this
  would hand SeGuRu a free win.
* **Tile-aligned sizes.** All sizes are exact multiples of the tile geometry, so
  the `pad2`/`unpad2` paths are the identity and both sides do identical work.
* **CPU baseline** is the crate's existing scalar reference, run at the smallest
  size of each sweep only (it takes minutes at the larger ones).
### Results

A100 80GB PCIe, CUDA 13.3, driver-default clocks. `SeGuRu/CUDA` below 1.00
means SeGuRu is faster. `before` is the ratio measured on the first version of
this port, prior to the optimisations documented in the log below; the CUDA
column is unchanged between the two (same binary, same baseline, re-measured on
an idle GPU).

| Kernel | Size | SeGuRu (us) | CUDA (us) | SeGuRu/CUDA | before | CPU (us) | GPU vs CPU | max rel err |
|---|---|---|---|---|---|---|---|---|
| gemm | 512^3 | 63.6 | 57.5 | 1.11x | 1.10x | 226931 | 3569x | 0.0e0 |
| gemm | 1024^3 | 287.6 | 277.7 | 1.04x | 1.04x | - | - | 0.0e0 |
| gemm | 2048^3 | 1775.2 | 1755.9 | 1.01x | 1.01x | - | - | 0.0e0 |
| gemm | 4096^3 | 12933.6 | 13077.2 | 0.99x | 0.99x | - | - | 0.0e0 |
| twomm | 512^3 x2 | 129.7 | 116.4 | 1.11x | 1.12x | 433358 | 3340x | 0.0e0 |
| twomm | 1024^3 x2 | 592.6 | 567.8 | 1.04x | 1.04x | - | - | 0.0e0 |
| twomm | 2048^3 x2 | 3546.4 | 3513.9 | 1.01x | 1.01x | - | - | 0.0e0 |
| threemm | 512^3 x3 | 193.8 | 174.9 | 1.11x | 1.11x | 650802 | 3358x | 0.0e0 |
| threemm | 1024^3 x3 | 903.8 | 870.5 | 1.04x | 1.04x | - | - | 0.0e0 |
| threemm | 2048^3 x3 | 5316.2 | 5286.3 | 1.01x | 1.01x | - | - | 0.0e0 |
| syrk | 512^3 | 71.3 | 67.0 | 1.06x | 1.06x | 129605 | 1817x | 1.1e-7 |
| syrk | 1024^3 | 352.3 | 339.9 | 1.04x | 1.04x | - | - | 1.1e-7 |
| syrk | 2048^3 | 2205.7 | 2147.8 | 1.03x | 1.03x | - | - | 1.1e-7 |
| syrk | 4096^3 | 16600.4 | 16153.8 | 1.03x | 1.03x | - | - | 1.1e-7 |
| syr2k | 512^3 | 134.0 | 149.6 | 0.90x | 0.89x | 134389 | 1003x | 1.5e-6 |
| syr2k | 1024^3 | 699.7 | 792.1 | 0.88x | 0.88x | - | - | 2.1e-6 |
| syr2k | 2048^3 | 4565.0 | 5085.0 | 0.90x | 0.90x | - | - | 4.0e-6 |
| atax | 2048^2 | 225.1 | 126.5 | 1.78x | 2.83x | 34901 | 155x | 4.2e-7 |
| atax | 4096^2 | 496.4 | 638.2 | 0.78x | 1.20x | - | - | 4.4e-7 |
| atax | 8192^2 | 1069.2 | 1360.7 | 0.79x | 1.18x | - | - | 6.3e-7 |
| bicg | 2048^2 | 225.1 | 119.1 | 1.89x | 3.01x | 4452 | 20x | 1.6e-7 |
| bicg | 4096^2 | 495.5 | 637.8 | 0.78x | 1.20x | - | - | 2.4e-7 |
| bicg | 8192^2 | 1069.8 | 1360.4 | 0.79x | 1.18x | - | - | 4.1e-7 |
| gesummv | 2048^2 x2 | 15.8 | 19.8 | 0.80x | 0.76x | 4177 | 264x | 1.8e-7 |
| gesummv | 4096^2 x2 | 84.7 | 86.5 | 0.98x | 0.98x | - | - | 3.0e-7 |
| gesummv | 8192^2 x2 | 310.3 | 316.1 | 0.98x | 0.98x | - | - | 4.0e-7 |
| mvt | 2048^2 | 227.0 | 119.6 | 1.90x | 3.00x | 27830 | 123x | 2.8e-7 |
| mvt | 4096^2 | 499.8 | 640.9 | 0.78x | 1.19x | - | - | 2.6e-7 |
| mvt | 8192^2 | 1076.1 | 1362.5 | 0.79x | 1.18x | - | - | 3.7e-7 |
| conv2d | 2048^2 | 27.8 | 26.3 | 1.06x | 1.21x | 5798 | 209x | 0.0e0 |
| conv2d | 4096^2 | 99.9 | 96.3 | 1.04x | 1.18x | - | - | 0.0e0 |
| conv2d | 8192^2 | 387.0 | 375.0 | 1.03x | 1.16x | - | - | 0.0e0 |
| conv3d | 128^3 | 22.8 | 21.5 | 1.06x | 1.80x | 4539 | 199x | 1.1e-7 |
| conv3d | 256^3 | 162.7 | 155.0 | 1.05x | 1.85x | - | - | 1.2e-7 |
| conv3d | 384^3 | 539.3 | 514.4 | 1.05x | 1.88x | - | - | 1.1e-7 |
| jacobi1d | 1048576 x t20 | 260.0 | 243.8 | 1.07x | 1.16x | 11427 | 44x | 0.0e0 |
| jacobi1d | 4194304 x t20 | 967.0 | 909.3 | 1.06x | 1.08x | - | - | 0.0e0 |
| jacobi1d | 16777216 x t20 | 3781.3 | 3689.9 | 1.02x | 1.05x | - | - | 0.0e0 |
| jacobi2d | 1024^2 x t10 | 154.6 | 136.6 | 1.13x | 1.56x | 7649 | 49x | 0.0e0 |
| jacobi2d | 2048^2 x t10 | 559.9 | 536.0 | 1.04x | 1.32x | - | - | 0.0e0 |
| jacobi2d | 4096^2 x t10 | 2111.1 | 2035.8 | 1.04x | 1.26x | - | - | 0.0e0 |
| fdtd2d | 1024^2 x t10 | 275.3 | 194.7 | 1.41x | 1.43x | 13022 | 47x | 3.1e-7 |
| fdtd2d | 2048^2 x t10 | 1237.4 | 1222.0 | 1.01x | 1.01x | - | - | 3.4e-7 |
| fdtd2d | 4096^2 x t10 | 4672.6 | 4605.3 | 1.01x | 1.01x | - | - | 3.6e-7 |

### Vendor-library reference

The mirrored CUDA baseline is a straightforward shared-memory tiled SGEMM, not a
tuned one — no double buffering, no `mma`, no vectorised shared loads, and it
inherits the bank conflict described above. For calibration, here is cuBLAS on
the same problem.

> **Both hand-written implementations are 1.8-2.2x slower than cuBLAS.** The GEMM
> parity result in the table above means "SeGuRu matches the CUDA you would write
> by hand", *not* "SeGuRu matches the best available GEMM". Neither this crate nor
> `cuda/polybench_ref.cu` is competitive with the vendor library, and neither is
> trying to be — the point of the comparison is to isolate the cost of SeGuRu's
> code generation against an identical algorithm.

| Kernel | Size | SeGuRu (us) | CUDA mirror (us) | cuBLAS (us) | SeGuRu/cuBLAS |
|---|---|---|---|---|---|
| gemm | 512^3 | 63.2 | 57.4 | 28.3 | 2.24x |
| gemm | 1024^3 | 287.3 | 277.5 | 129.8 | 2.21x |
| gemm | 2048^3 | 1775.6 | 1761.7 | 982.4 | 1.81x |
| gemm | 4096^3 | 12918.1 | 13063.0 | 7341.9 | 1.76x |

### Reading the numbers

* **GEMM family (`gemm`, `twomm`, `threemm`, `syrk`) — parity, untouched.**
  1.11x at 512 (where a ~5 us fixed overhead is still visible), converging to
  0.99-1.03x from 1024 upward. These kernels are register/shared-memory bound
  and both compilers produce essentially the same inner loop (40 registers each
  for `gemm`). Their loads go through shared memory inside a fully unrolled
  loop, where the bounds check is hoisted or folded already, so none of the
  optimisations below apply to them and none were made.
* **`syr2k` and `gesummv`: SeGuRu is 10-20% *faster*.** Unchanged from the
  first port, and still the result I trust least. The CUDA `syr2k` uses 48
  registers/thread and SeGuRu 71, and both use the same 16 KB of shared memory
  and 256 threads/block, so CUDA has the *higher* occupancy and is still
  slower. I did not manage to close the gap by hand-tuning the CUDA side, but I
  would not present this as "SeGuRu beats CUDA": it means my CUDA `syr2k` is
  leaving ~10% on the table.
* **Stencils — now 1.02-1.13x** (`conv3d` 1.05-1.06x, `conv2d` 1.03-1.06x,
  `jacobi2d` 1.04-1.13x, `jacobi1d` 1.02-1.07x), down from 1.05-1.88x. Two
  changes did all of it: making the slice bounds check provable (experiment A)
  and replacing the output `reshape_map!` with the plain linear map so the
  runtime `div.u32` disappears (experiment B).
* **Memory-bound mat-vec (`atax`, `bicg`, `mvt`) — 0.78-0.79x at 4096/8192,
  1.78-1.90x at 2048.** The 0.78x is *not* a SeGuRu win worth quoting; see the
  dedicated section below. The 2048 point is the same launch-configuration
  artifact as before (8 CTAs on 108 SMs), improved from 3.0x only because the
  kernel now issues fewer instructions in an issue-bound regime.
* **`fdtd2d` — untouched, 1.01x at scale.** It was already at parity at 2048
  and 4096; the 1.41x at 1024 is 3 short launches per time step against a
  273 us total, i.e. launch-latency dominated. Left alone deliberately: the
  brief was to fix the kernels that trail at scale.

### Do not read `atax`/`bicg`/`mvt` at 0.78x as "SeGuRu beats CUDA"

After experiment A the SeGuRu column pass is faster than the mirrored CUDA one,
which flips the ratio below 1.0. That is a real measurement — the outputs are
verified element-wise every run, worst error 6.3e-7 — but it says more about the
baseline than about SeGuRu. `nsys profile --trace cuda` at n=8192:

| kernel | SeGuRu | CUDA |
| --- | ---: | ---: |
| column pass (`mvt_x2` / `mv_col_acc`) | 931 us | 1218 us |
| row pass (`mvt_x1` / `mv_row_acc`) | 164 us | 166 us |

The column pass moves 268 MB. 931 us is **288 GB/s**; 1218 us is **220 GB/s**.
The A100's HBM peak is about 1555 GB/s, so *both* implementations are running at
14-19% of peak. One thread per output column with `n / 256` CTAs puts 32 blocks
on 108 SMs and cannot keep enough loads in flight; whichever side issues fewer
instructions per load wins, and after experiment A that is SeGuRu. The honest
statement is: **the mirrored CUDA column kernel is leaving 5x on the table, and
so is the SeGuRu one.** A properly blocked column sweep would beat both by a
wide margin. I did not "fix" the CUDA baseline, because changing it to make
SeGuRu look worse is as dishonest as weakening it to make SeGuRu look better;
the number is reported with this caveat instead.

## Optimisation log

Everything below is a change to the *Rust* side only. `cuda/polybench_ref.cu`
was not touched. Instruction counts are from the emitted PTX
(`target/release/deps/gpu/libpolybench_bench-*.gpu.gpu.ptx` vs
`nvcc -arch=sm_80 -O3 --use_fast_math -ptx cuda/polybench_ref.cu`), counted by
class rather than eyeballed; register counts are from `ptxas -arch=sm_80 -O3 -v`
(`ncu` cannot read counters on this machine, `ERR_NVGPUCTRPERM`).

### A. Make the slice bounds check provable, for arbitrary sizes — kept

**The problem.** SeGuRu emits, for every global load through a `&[f32]` kernel
parameter, a bounds check it never eliminates:

```
cvt.u64.u32     %rd9, %r14;
setp.gt.u64     %p3, %rd9, %rd2;      // idx > len?
selp.b64        %rd10, 0, %rd8, %p3;  // branchless: address or null
selp.b64        %rd11, 0, %rd10, %p3;
ld.global.f32   %f3, [%rd10];
```

Nothing bounds a thread-derived index, so LLVM cannot discharge it.
`src/bin/boundscheck.rs` had already *measured* the cost (29-31% of `conv3d`,
50-59% of the `mvt` column pass) using a compile-time power-of-two mask, which
only works for power-of-two sizes.

**The change.** A two-step idiom, in safe Rust, that works for any size —
documented on `crate::ix` in `src/lib.rs`:

```rust
let total = ni * nj * nk;                                // u32, exact extent
if total == 0 || a.len() < total as usize { return; }
let a = &a[..total as usize];                            // len == zext(total)
let last = total - 1;
...
a[ix(idx, last)]                                         // idx.min(last) as usize
```

The sub-slice is the load-bearing part. After it, `a.len()` is literally
`zext(total)`, so LLVM can prove `zext(umin(idx, total - 1)) < zext(total)` by
comparing the `u32` operands, and the check folds to a single `min.u32`. The
clamp is a no-op on the data — every index these kernels produce is already
`< total` — so the results are bit-identical, which the benchmark's
element-wise comparison against CUDA confirms on every run.

**Rejected sub-variants**, both tried and both worse:

| form | conv3d PTX | conv3d 384^3 | why |
| --- | ---: | ---: | --- |
| stock | 204 instrs, 12 `setp.gt.u64`, 24 `selp.b64` | 966 us (1.88x) | baseline |
| `a[idx.min(a.len() - 1)]`, guarded by `if a.is_empty()` inside the helper | 264 instrs, 18 branches | not measured | the `is_empty` early-return inlines 15 times and defeats CSE of the addresses (11 -> 15 `ld.global`) |
| `a[idx.min(a.len() - 1)]`, guard hoisted to the kernel | 183 instrs, 1 `setp.gt.u64`, 11 `min.u64`, 12 `cvt.u64.u32` | 881 us (1.70x) | provable, but only in 64 bits: `min.u64` is `ISETP`+`SEL` in SASS, so it gives back two thirds of the win |
| clamp in `u32` against `(a.len() - 1) as u32` | 204 instrs, checks **back** (12/24 again) | — | LLVM cannot see through `trunc`/`zext`, so `zext(umin(idx, trunc(len-1))) < len` is not provable |
| **sub-slice + `u32` clamp (kept)** | **176 instrs, 1 `setp.gt.u64`, 14 `min.u32`** | **668 us (1.30x)** | provable in 32 bits |

Applied to `conv3d`, `conv2d`, `jacobi1d` (step only), `jacobi2d`, `mvt`,
`atax`, `bicg`. Registers are essentially unchanged (`mvt_x2` 34 -> 32,
`conv3d` 32 -> 32), so the win is issue bandwidth, not occupancy — which is what
you would expect for kernels this far from the memory roofline.

`jacobi1d_copy` is the one place the idiom was applied and then **reverted**: it
has a single load already guarded by `interior`, and the entry guard grew the
kernel from 32 to 51 PTX instructions without removing any check. Reverted; the
shipped `jacobi1d_copy` is the original.

### B. Replace the output `reshape_map!` with `MapContinuousLinear` — kept

Reading the post-A PTX showed the *store* address, not the loads, was now the
biggest remaining block of integer work in `conv3d` and `jacobi2d`:

```
mad.lo.s32  %r39, %r5, %r6, %r4;   // flatten thread id
shr.u32     %r40, %r39, 5;
div.u32     %r41, %r40, %r2;       // <-- runtime integer division
mul.lo.s32  %r42, %r41, %r2;
sub.s32     %r43, %r40, %r42;      // ... and the matching modulo
shl.b32     %r44, %r43, 5;
or.b32      %r45, %r44, %r1;
and.b32     %r46, %r41, 7;
mad.lo.s32  %r47, %r46, %r6, %r45;
shr.u32     %r60, %r41, 3;
mad.lo.s32  %r48, %r60, %r7, %r47;
```

`reshape_map!` flattens the thread id into a single linear index and then
*un-flattens* it against the target dimensions — even when, as here, the layout
is the identity permutation of the hardware ids. Because `grid_dim::<DimX>()` is
a runtime value, the un-flattening emits `div.u32`, which sm_80 has no
instruction for; ptxas expands it to a long Newton-Raphson sequence.

Both kernels' grids map exactly onto their arrays, so
`MapContinuousLinear::new(1)` — which is just
`gid_x + (gid_z * gdim_y + gid_y) * gdim_x` — already computes the right
address: `j + i * n` for `jacobi2d`, `k + (i * nj + j) * nk` for `conv3d`.

| kernel | PTX instrs stock / after A / after B | CUDA | `div.u32` after B |
| --- | ---: | ---: | ---: |
| `conv3d_kernel` | 204 / 176 / **162** | 129 | 0 (was 2) |
| `jacobi2d_step` | 126 / 115 / **92** | 66 | 0 (was 2) |
| `jacobi2d_copy` | 82 / 88 / **65** | 32 | 0 (was 2) |

| kernel | stock | after A | after B | CUDA |
| --- | ---: | ---: | ---: | ---: |
| conv3d 384^3 | 966.6 us | 667.6 us | **539.3 us** | 514.4 us |
| jacobi2d 4096^2 x t10 | 2569.4 us | 2292.5 us | **2111.1 us** | 2035.8 us |

`Map2D::new(n)` was tried first for `jacobi2d` and also removes the division
(2118 us); `MapContinuousLinear` is marginally better and needs no `x_size`
argument, so that is what shipped.

`conv2d` still uses `reshape_map!` and still contains one `div.u32`: each
thread owns four *consecutive* rows, and neither `MapContinuousLinear` nor
`Map2D` can express a blocked local dimension (`Map2D`'s local index is
strided). Restructuring to strided rows would break the sliding register window
that makes `conv2d` load 18 values per 4 outputs instead of 36, which is a much
larger effect. Left as is; `conv2d` is at 1.03x anyway.

The `div.u32` is present in 19 of the emitted kernels, including the whole GEMM
family — there it is loop-invariant and amortised over the K loop, which is why
those kernels never showed it. **This is the single most valuable thing to fix
in `reshape_map!` itself:** when the `layout:` permutation is the identity over
the hardware id dimensions, the flatten/un-flatten round trip is a no-op and
should be elided.

### C. The residual `mvt` column-pass gap: `IMAD.WIDE.U32` — measured, not fixed

Re-running `polybench-boundscheck` after A and B:

```
| Kernel | Size | stock (us) | no-bounds-check (us) | tax | max rel err |
|---|---|---|---|---|---|
| conv3d | 128^3 | 23.0 | 27.7 | -20.4% | 0.0e0 |
| conv3d | 256^3 | 162.6 | 198.7 | -22.2% | 0.0e0 |
| mvt (column pass) | 2048^2 | 87.0 | 78.8 | 9.5% | 6.9e-8 |
| mvt (column pass) | 8192^2 | 916.4 | 716.5 | 21.8% | 1.2e-7 |
```

`conv3d` is now *negative*: the shipped kernel beats the power-of-two-masked
variant, because the masked variant still carries the `reshape_map!` division
that experiment B removed. But the mvt column pass still trails its masked twin
by 22%, and the PTX does not explain it — 99 instructions vs 98, 32 registers
vs 32, 9 `ld.global` each. The SASS does:

| variant | total SASS | `IMAD.WIDE.U32` | `IMNMX.U32` | `LOP3.LUT` | `UIADD3`/`ULOP3` |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped (`min` against a runtime bound) | 352 | 65 | 56 | 1 | 0 |
| masked (`and` against a compile-time constant) | 480 | 1 | 0 | 33 | 85 |

The shipped kernel has *fewer* SASS instructions and is still slower. Clamping
against a runtime bound leaves ptxas with an index of unknown range, so every
access needs a 32->64-bit widening multiply (`IMAD.WIDE.U32`, FMA pipe) plus the
`IMNMX`. Masking against a compile-time constant bounds the index to 28 bits, so
ptxas strength-reduces the whole address chain into 32-bit `LOP3`/`IADD3`, and —
because the loop counter is warp-uniform — pushes 85 of those onto the *uniform*
datapath, off the main issue path entirely.

I could not reproduce that in safe Rust without reintroducing a compile-time
size, and a power-of-two fast path was explicitly a last resort, so this is left
open. It is worth roughly 20% on `atax`/`bicg`/`mvt` at scale. Both the shipped
and the masked kernel are far off the memory roofline anyway (288 and 375 GB/s
of 1555), so the real fix for these three is a blocked column sweep, not more
address-arithmetic tuning.

### D. Secondary levers that were checked and found to be no-ops

* **`u32`/`i32` indices instead of `usize`/`u64`** — already done throughout the
  original port; experiment A is what finally lets the 32-bit arithmetic survive
  to the address computation instead of being widened for the check.
* **`crunchy::unroll!` on inner loops** — already applied everywhere it helps
  (`mvt`/`atax`/`bicg` column passes, `conv2d`'s row window). ptxas additionally
  unrolls the column passes 8x on its own (65 `LDG.E` for an 8-load body).
* **`#[gpu::device]` helper inlining** — `conv3d::at` and `conv2d::row3` are
  `#[inline(always)]` and do inline; the PTX shows no `call`.
* **Vector loads** — the row passes already use `Float4`. The column passes
  cannot: consecutive `j` belong to different threads, so a per-thread `float4`
  would read four *rows*, not four contiguous floats.

### PTX evidence, whole suite

Counted with a script over `.visible .entry` bodies, not by eye. `setp.gt.u64` +
`selp.b64` is the bounds-check signature (one `setp` and two `selp` per checked
access); `min.u32` is the clamp that replaces it.

| kernel | instrs before | instrs after | CUDA | `setp.gt.u64` before/after | `selp.b64` before/after | `div.u32` before/after |
|---|---:|---:|---:|---:|---:|---:|
| `conv3d_kernel` | 204 | 162 | 129 | 12 / 1 | 24 / 2 | 2 / 0 |
| `conv2d_kernel` | 298 | 253 | 192 | 22 / 4 | 44 / 8 | 1 / 1 |
| `jacobi2d_step` | 126 | 92 | 66 | 7 / 2 | 14 / 4 | 2 / 0 |
| `jacobi2d_copy` | 82 | 65 | 32 | 3 / 2 | 6 / 6 | 2 / 0 |
| `jacobi1d_step` | 61 | 55 | 34 | 5 / 2 | 10 / 4 | 0 / 0 |
| `jacobi1d_copy` | 32 | 32 (reverted) | 20 | 2 / 2 | 6 / 6 | 0 / 0 |
| `mvt_x2` | 111 | 93 | 61 | 9 / 1 | 18 / 2 | 0 / 0 |
| `atax_y` | 110 | 92 | 61 | 9 / 1 | 18 / 2 | 0 / 0 |
| `bicg_s` | 110 | 92 | 61 | 9 / 1 | 18 / 2 | 0 / 0 |
| `mvt_x1` | 68 | 70 | 68 | 3 / 1 | 6 / 2 | 0 / 0 |
| `atax_tmp` | 66 | 68 | 68 | 3 / 1 | 6 / 2 | 0 / 0 |

Load counts are identical before and after in every row (e.g. `conv3d` 11
`ld.global.f32`, `conv2d` 18, `mvt_x2` 9), which is the check that the clamp did
not accidentally change what is read. Two rows *grew*: `mvt_x1` and `atax_tmp`
pay the entry guard (`total`, the length test, the sub-slice) but only have one
checked access outside their unrolled body to save it on. Their hot loops did
lose the checks (3/6 -> 1/2), and both are ~20% of their benchmark's total time,
so they were kept; `jacobi1d_copy` had nothing to save at all and was reverted.

### The mat-vec ratio at n=2048 is still a launch-configuration artifact

`atax`, `bicg` and `mvt` show 1.78-1.90x at n=2048 but 0.78-0.79x at 4096 and
8192. That was ~3.0x before these optimisations and the explanation is unchanged
— it is the grid shape, not a codegen cliff at small sizes. The column kernel
launches `n / 256` blocks, so at n=2048 it puts **8 CTAs on 108 SMs**, 7% of the
machine, and at that occupancy the kernel is purely instruction-issue-bound.
Separately, CUDA's column pass costs 111 us at 2048 and 601 us at 4096 — a 5.4x
jump for 4x the work, because the 16 MB matrix fits in the A100's 40 MB L2 at
2048 and the 64 MB one does not at 4096.

Experiment A cut SeGuRu's per-load instruction count, which is exactly what an
issue-bound kernel is limited by, and the ratio fell from 3.0x to 1.9x. It did
not go to parity because the remaining gap is the `IMAD.WIDE.U32` chain of
experiment C, which is also pure issue cost. **Neither the 1.9x nor the 0.78x
should be quoted as "SeGuRu's mat-vec penalty":** the first is an
under-occupied launch on an L2-resident working set, the second is two
implementations racing at 15-19% of memory peak. The launch shape (one thread
per output column) is a property of the ported kernel, and it is identical on
both sides.

### Coverage

**Mirrored in CUDA (14):** `gemm`, `twomm`, `threemm`, `syrk`, `syr2k`, `atax`,
`bicg`, `gesummv`, `mvt`, `conv2d`, `conv3d`, `jacobi1d`, `jacobi2d`, `fdtd2d` —
i.e. all of the compute-bound BLAS-like kernels, all of the memory-bound
mat-vec kernels, and all of the stencils.

**Skipped (5):** `doitgen`, `covar`, `corr`, `lu`, `gramschm`. These are
multi-phase kernels (a covariance/correlation pipeline, a sequential-in-`k`
factorisation, and a 256-step orthogonalisation) whose cost is dominated by kernel
launch sequencing rather than by the inner loops, so they would mostly measure the
driver. They are covered by the correctness tests but not benchmarked.

### Reproducing

```bash
source /home/ziqiaozhou/seguru/casestudies-by-agent-v2/env.sh   # required, every time
cd /home/ziqiaozhou/seguru/casestudies-by-agent-v2

# main comparison
cargo build --release -p polybench-gpu --features bench --bin polybench-bench
./target/release/polybench-bench            # all kernels
./target/release/polybench-bench gemm       # one kernel

# bounds-check tax experiment
cargo build --release -p polybench-gpu --features bench --bin polybench-boundscheck
./target/release/polybench-boundscheck

# launch-configuration evidence for the n=2048 mat-vec outlier
nsys profile --trace cuda -o mvt --force-overwrite true ./target/release/polybench-bench mvt
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace mvt.nsys-rep

# correctness
cargo test --release -p polybench-gpu --lib -- --test-threads=1   # 19/19
```

`build.rs` compiles `cuda/polybench_ref.cu` with `nvcc -O3 -arch=native
--use_fast_math`. `-arch=native` is mandatory: without it the kernels fail to
launch silently and leave the output buffers zeroed.

`src/cuda_ffi.rs` is the only file in the crate containing `unsafe`, and it is
gated behind `#[cfg(feature = "bench")]`, so the default build and the test suite
remain 100% safe Rust.

## SeGuRu limitations hit

* **Codegen stack overflow.** `rustc-gpu` aborts with
  `thread 'opt cgu.0' has overflowed its stack` if a single kernel body unrolls too
  far. A fully unrolled 16-deep K loop over a 4×4 micro-tile triggers it. The fix
  was to split the unroll: `for kh in 0..(KTILE / 4) { unroll! { for kl in 0..4 { .. } } }`.
  (`RUST_MIN_STACK=134217728` also works but is not something a caller should have
  to know.)
* **No closures in kernels.** A `let f = |..| ..` inside a `#[gpu::cuda_kernel]`
  fails with `E0658: attributes on expressions are experimental`. `#[gpu::device]`
  free functions are the workaround (see `conv3d::at`).
* **`reshape_map!` thread dims must be literals.** Using a `const` such as
  `RED_BDIM` in the thread-dim list makes the proc macro fail to parse.
* **`layout:` and `offset:` need a comma between them** — `=> layout: [..], offset: e`.
* **No read+write aliasing of a buffer in one launch**, as described above.
* **Scalar kernel parameters are reordered** in the generated `launch` signature
  (u32s before f32s), which is easy to trip over when editing a signature.
* **Per-crate `const_alloc_<N>` naming collided at link time** (*fixed upstream
  during this work*). `rustc_codegen_gpu` named constant allocations from a
  per-crate counter, so a library and a binary that each emitted at least one GPU
  const alloc failed to link with
  `error: Linking globals named 'const_alloc_0': symbol multiply defined!`. The
  same class of bug had previously been fixed for `static_shared_*` only;
  `const_alloc_*` and `memory_alloc_*` now also get a crate-unique suffix
  (`crate_unique_suffix()` in `crates/rustc_codegen_gpu/src/context/const_static.rs`).
  With that fix the driver lives in `src/bin/bench.rs` in the natural layout and
  a second binary (`src/bin/boundscheck.rs`) links alongside it.
* **Slice bounds checks are not elided on kernel parameters.** Every global load
  through a `&[f32]` parameter emits `setp.gt.u64` + two `selp.b64`. Originally
  measured at 29-31% of conv3d's runtime and 50-59% of the mvt column pass's.
  This is a compiler limitation, not a language one — the indices are provably
  in range. The safe-Rust workaround (experiment A above) is a sub-slice to the
  exact extent plus a 32-bit clamp, `crate::ix`; it costs one `min.u32` per
  access instead of three instructions, and it recovers most but not all of the
  gap, because a runtime clamp bound still leaves ptxas emitting
  `IMAD.WIDE.U32` for every address (experiment C). A compiler that propagated
  the range fact itself would not need the clamp at all.
* **`reshape_map!` emits a runtime `div.u32` even when the layout is the
  identity permutation.** It linearises the hardware thread id and then
  un-flattens it against `grid_dim::<DimX>()`, which is a runtime value, so
  sm_80 gets an integer division it has no instruction for. Worth 24% of
  `conv3d` and 20% of `jacobi2d_step` on its own. Workaround:
  `MapContinuousLinear::new(1)` (or `Map2D::new(x_size)`) where the grid maps
  directly onto the array — see experiment B. `div.u32` still appears in 19 of
  the emitted kernels in this crate, including the GEMM family, where it is
  loop-invariant and therefore harmless; the stencils were the ones that paid
  for it per output element. This looks like the highest-value fix available on
  the SeGuRu side.
* **No map expresses a *blocked* local dimension.** `conv2d` gives each thread
  four consecutive rows; `Map2D`'s local index is strided and
  `MapContinuousLinear` has no local dimension, so `conv2d` had to keep its
  `reshape_map!` and its `div.u32`.
