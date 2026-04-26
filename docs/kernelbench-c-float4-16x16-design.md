# KernelBench-C Float4 and 16x16 GEMM Design

This document explains the retained KernelBench-C GEMM design choices on the
slim branch:

- `Float4` global-memory views for `x` and `W`;
- K-major shared-memory tile layout;
- 16x16 thread blocks with 8x8 per-thread register tiles;
- checked `Float4` vector views for any vectorized epilogue operands.

The goal is raw custom CUDA parity. PyTorch/cuBLAS numbers are useful context,
but these kernels are compared primarily against hand-written CUDA kernels with
the same algorithm and launch geometry.

## Scope

This design describes the current `agent-poc-v2` slim branch. It intentionally
does not depend on the open-tile prototype, row views, generated row-offset
traits, or core codegen experiments. K-major shared memory is retained because
the corrected ablation showed it is a source-level performance requirement, not
an invasive codegen feature.

The representative implementation is
`examples/kernelbench-c/src/gemm_add_relu.rs`; the same geometry and host-side
view pattern is applied across the fused GEMM/matmul KernelBench-C kernels.

## Kernel geometry

The dense fused GEMM kernels use these constants:

| Constant | Value | Meaning |
|---|---:|---|
| `BM` | 128 | output rows per block |
| `BN` | 128 | output columns per block |
| `BK` | 8 | K elements per shared-memory tile |
| `TM` | 8 | output rows per thread |
| `TN` | 8 | output columns per thread |
| `BDIM_X` | 16 | threads along output columns |
| `BDIM_Y` | 16 | threads along output rows |

The launch grid is:

```rust
let gx: u32 = nn / BN;
let gy: u32 = mm / BM;
let cfg = gpu_host::gpu_config!(gx, gy, 1, BDIM_X, BDIM_Y, 1, 0);
```

Each thread block therefore has 256 threads and owns one 128x128 output tile.
Each thread `(tx, ty)` computes an 8x8 register tile:

```text
output row = bid_y * 128 + ty * 8 + i,  i in 0..8
output col = bid_x * 128 + tx * 8 + j,  j in 0..8
```

This is why 16x16 and 8x8 are coupled: `16 * 8 = 128` in both output
dimensions. Changing either block dimension requires changing the per-thread
tile or block tile dimensions as a single design.

## Float4 global loads

`x` and `W` are logically row-major `f32` matrices:

```text
x: [M, K]
W: [N, K]
```

The kernels take them as `&[Float4]`:

```rust
pub fn gemm_add_relu_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[Float4],
    y: &mut [Float4],
    M: u32,
    N: u32,
    K: u32,
)
```

The host still allocates normal `f32` tensors. Before launch, it creates
checked zero-copy vector views:

```rust
let d_x4_view = d_x
    .try_cast_slice::<Float4>()
    .expect("x Float4 view requires 16-byte alignment and length divisible by 4");
let d_w4_view = d_w
    .try_cast_slice::<Float4>()
    .expect("w Float4 view requires 16-byte alignment and length divisible by 4");
let d_x4 = &d_x4_view;
let d_w4 = &d_w4_view;
```

Inside the kernel, `K` is converted to `Float4` units:

```rust
let k4 = K >> 2;
let k_base4 = tstep * (BK >> 2);
let a_col4 = a_col >> 2;
```

This turns four scalar global loads into one 128-bit global load:

```rust
let v: Float4 = x[((bm + a_row) * k4 + k_base4 + a_col4) as usize];
```

The loaded vector is then expanded into scalar shared memory:

```rust
ca[0] = v[0];
ca[1] = v[1];
ca[2] = v[2];
ca[3] = v[3];
```

This branch uses `Float4` only for global-memory traffic. Shared memory and the
register-tile compute loop remain scalar `f32`, matching the existing SeGuRu
chunk/map abstractions.

## Thread partition for tile loads

The 256 threads in a block cover one `128 x 8` tile of `x` and one `128 x 8`
tile of `W` per K step. The flat thread id partitions the tile:

```rust
let tid = ty * BDIM_X + tx; // 0..255
let a_row = tid >> 1;       // 0..127
let a_col = (tid & 1) << 2; // 0 or 4
```

For each row of the shared tile, two threads load the full 8-wide K slice:

```text
thread 2*r + 0 loads columns 0..3 as one Float4
thread 2*r + 1 loads columns 4..7 as one Float4
```

The shared-memory load map is K-major:

```rust
let load_map = reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0]);
```

For local lane `i0`, this maps to:

```text
offset = (a_col + i0) * 128 + a_row
```

So each thread still writes four disjoint shared-memory slots, and all 256
threads collectively fill exactly 1024 scalar slots (`128 * 8`) without
overlap. The difference from the original slim branch is the physical shared
layout: compute reads use CUDA-style `tile[k][row_or_col]`, which is much faster
than reading the old row-major shared tile as `tile[row_or_col][k]`.

## Register-tile compute loop

After both shared tiles are loaded, every thread computes 64 output values in
registers:

```rust
let mut acc = [[0.0f32; TN as usize]; TM as usize];
```

For each `kk` in the 8-wide K tile, the thread reads:

```rust
let row_off = (ty * TM) as usize;
let col_off = (tx * TN) as usize;

a_reg[ii] = tile_a[kk * BM as usize + row_off + ii];
b_reg[jj] = tile_b[kk * BN as usize + col_off + jj];
```

Then it performs the outer product:

```rust
acc[ii][jj] += a_reg[ii] * b_reg[jj];
```

The design keeps `TM` and `TN` at 8 because that matches the raw CUDA
register-tiled algorithm: enough work per thread to reuse shared-memory loads,
but still small enough to compile and run with the launch-bound register cap.

## Output mapping

Most GEMM/matmul epilogues write each thread's 8x8 register tile through
scalar `chunk_mut`:

```rust
let out_map = reshape_map!(
    [8, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
    => layout: [i0, t0, t1, i1, t2, t3]
);
let mut y_thread = chunk_mut(y, out_map);
```

The local write:

```rust
y_thread[(j as u32, i as u32)] = value;
```

corresponds to:

```text
y[(bid_y * 128 + ty * 8 + i) * N + (bid_x * 128 + tx * 8 + j)]
```

The important safety property is that `chunk_mut` receives a map whose local
8x8 tile is unique for each `(block, thread)` pair. The kernel code never
constructs raw mutable pointers for epilogue writes.

`gemm_add_relu` additionally uses a vectorized bias/output epilogue because the
preserved reference branch had that source-level change. The kernel signature is
`bias: &[Float4]` and `y: &mut [Float4]`, and the output map writes two
`Float4` values per row:

```rust
let out_map = reshape_map!(
    [2, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
    => layout: [i0, t0, t1, i1, t2, t3]
);
let mut y_tile = chunk_mut(y, out_map);

unroll! { for i in 0..8 {
    let mut o0 = Float4::default();
    let mut o1 = Float4::default();
    // fill o0/o1 from acc + bias
    y_tile[(0u32, i as u32)] = o0;
    y_tile[(1u32, i as u32)] = o1;
}}
```

The earlier reference branch spelled this with `open_tile().row_mut()`. The
current branch keeps the same vectorized output shape but uses direct
`chunk_mut` indexing, because isolated testing showed the row-view helper itself
was performance-neutral and would require restoring invasive core APIs.

## Launch-bound annotation

The slim branch uses:

```rust
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
```

This is not the launch configuration. The real launch geometry is still the
`gpu_config!(..., 16, 16, 1, ...)` call. The annotation tells NVVM/ptxas that
the kernel is launched with a 16x16x1 block and should target at least two
resident blocks per SM when possible.

The annotation must match the actual launch dimensions. If a kernel changes
`BDIM_X`, `BDIM_Y`, or `BDIM_Z`, update or remove the annotation in the same
patch. A mismatched launch-bound is a performance contract bug: it may force a
register allocation strategy for a block shape the kernel no longer uses.

## Safety invariants

The host-side `Float4` cast is safe only under these invariants:

1. the source and destination element types are non-zero-sized;
2. the source byte length is divisible by the destination element size;
3. the source device pointer satisfies `Float4` alignment;
4. the destination view length is rebuilt in `Float4` elements;
5. mutable casts require an original mutable tensor view.

`TensorView::try_cast_slice` and `TensorViewMut::try_cast_slice_mut` check the
size and alignment invariants. `Float4` is `#[repr(transparent)]` through
`gpu::VecType<T>`, and `VecType<T>` implements the `TensorViewCastElement`
marker used by the checked cast API.

The kernel also requires:

- `M % BM == 0`;
- `N % BN == 0`;
- `K % BK == 0`, which implies `K % 4 == 0` because `BK = 8`.

These are asserted in the host runner before launch.

## Why this branch stops here

The retained design is intentionally conservative:

- `Float4` global loads are a small source-level change that expresses the
  intended 128-bit global-load shape (`ld.global.v4.f32`) through a safe host
  abstraction.
- K-major shared memory is retained because it is required for the raw-CUDA
  access pattern in the compute loop; reverting it roughly doubled GEMM/matmul
  runtime on KernelBench-C.
- 16x16 block geometry is the raw-CUDA-compatible shape for these fused GEMM
  kernels and aligns with the 8x8 per-thread register tile.
- `nvvm_launch_bound(16, 16, 1, 2)` is tied directly to that launch geometry.

Invasive APIs such as open-tile row views and generated row-offset traits are
not part of this branch. The corrected conclusion is that K-major layout and
vectorized data shapes matter, while the `open_tile().row_mut()` helper itself
does not. The slim branch therefore keeps the existing safe map/chunk interface.

## Maintenance checklist

When adding or modifying a KernelBench-C fused GEMM kernel:

1. Keep `BDIM_X = 16`, `BDIM_Y = 16`, `TM = 8`, and `TN = 8` together unless
   changing the whole tiling design.
2. Use checked `try_cast_slice::<Float4>()` host views for `x` and `W`; do not
   add benchmark-local unsafe pointer casts.
3. Keep the K-major shared-memory map:
   `reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0])`.
4. Keep `K % BK == 0` assertions so `Float4` indexing is valid.
5. If an epilogue uses `Float4` for bias or output, create those views with
   checked `try_cast_slice::<Float4>()` / `try_cast_slice_mut::<Float4>()`.
6. Ensure `nvvm_launch_bound(16, 16, 1, 2)` matches the actual launch
   dimensions.
7. Use `chunk_mut` plus `reshape_map!` for output writes; do not introduce raw
   mutable epilogue pointers in benchmark code.
8. Do not reintroduce `.open_tile()` or `.row_mut()` for KernelBench-C unless the
   core API is separately justified and benchmarked.
