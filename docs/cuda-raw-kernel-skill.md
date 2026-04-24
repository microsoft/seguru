# Raw CUDA kernel skill (for KernelBench / PyTorch extensions)

Symmetric counterpart to `cuda-to-seguru-porting-skill.md`. Given a PyTorch
reference op, produce a single `.cu` file that:

1. Defines one or more `__global__` CUDA kernels.
2. Exposes a `torch::Tensor run(...)` wrapper callable from Python.
3. Binds it via `PYBIND11_MODULE` so `torch.utils.cpp_extension.load` works.

Target is Ampere (`sm_80`) and newer.

## Golden rules

1. **Grid-stride loops.** Always use
   `for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)`.
   This decouples launch geometry from problem size and keeps the kernel
   correct for any input shape.
2. **Vectorize elementwise with `float4`.** For contiguous fp32 problems where
   `n % 4 == 0`, cast pointers to `float4*` / `reinterpret_cast<float4*>` and
   process 4 elements per iteration. This halves or better the issue count
   on memory-bound kernels. **Also applies to row reductions** — see Row-Reduction
   Strategy below; measured 1.23–1.38× on `sum_dim` / `layer_norm` / `l2_norm`.
3. **Use `__ldg` for read-only loads** on older arches; on Ampere+ this is
   already implicit for `const __restrict__`, but `__ldg` is still safe.
4. **Stream awareness.** Always launch on
   `at::cuda::getCurrentCUDAStream()` so torch's stream machinery works.
   Include `<ATen/cuda/CUDAContext.h>`. Do NOT create your own stream.
5. **Block size = 256 or 512** unless you have a reason. Grid size =
   `min(div_ceil(n, block*vec), 65535)` or let grid-stride handle large n.
   See "Launch Config & Occupancy" for the full decision.
6. **Row reductions: pick parallelism per row, not per thread.** For
   `sum(x, dim=-1)`-style kernels on `[B, D]`: one block per row when `D ≥ ~1K`
   and `B ≤ ~10K`; one warp per row when `D ≤ 1024` and many rows; one
   thread per row **only** when rows are tiny (`D ≤ ~128`). Two-stage
   block reduction = per-thread accumulate → `__shfl_down_sync` within each
   warp → `__shared__` stage across warps. Put `__syncthreads()` between
   stages. See "Row-Reduction Strategy".
7. **fp32 accumulate** for any reduction, even if input/output are fp16/bf16.
   Cast to `float` inside the loop. Identity for `max` is `-FLT_MAX`, not 0.
8. **Guard tails.** After vectorized main loop, handle `n % vec` remainder
   in a scalar tail loop. Cheaper than padding.
9. **Don't allocate inside the kernel.** Use `torch::empty_like(x, x.options())`
   or `torch::empty({...}, x.options())` on the host wrapper.
10. **Assert contiguity + dtype** with `TORCH_CHECK(x.is_cuda())`,
    `TORCH_CHECK(x.is_contiguous())`, `TORCH_CHECK(x.dtype() == torch::kFloat32)`.
11. **Port the algorithm before the idioms.** Fuse multi-pass reductions
    (max+sum → single-pass online softmax; mean+var → Welford); pick
    vector width (float4) before writing the loop; pick the tile size with
    register tiling before writing a GEMM. A clean CUDA rendering of a bad
    algorithm will not reach parity. See Case Studies.

## File skeleton (copy-paste)

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void my_kernel(const float* __restrict__ x,
                          float* __restrict__ y,
                          int64_t n) {
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Vectorized main loop (4 elements per iter).
    int64_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);
    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        float4 r;
        r.x = /* op */ v.x;
        r.y = /* op */ v.y;
        r.z = /* op */ v.z;
        r.w = /* op */ v.w;
        y4[i] = r;
    }
    // Scalar tail.
    int64_t base = n4 * 4;
    for (int64_t i = base + tid; i < n; i += stride) {
        y[i] = /* op */ x[i];
    }
}

torch::Tensor run(torch::Tensor x /*, extra args */) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dtype() == torch::kFloat32);
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    int block = 256;
    int grid  = std::min<int64_t>((n + block*4 - 1) / (block*4), 65535);
    my_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "kernel");
}
```

## Row-Reduction Strategy

Before writing any row-wise reduction (sum, mean, max, argmax, norm,
softmax, layer_norm, rms_norm): pick the **parallelism per row** based on
`B` (rows) and `D` (row width). Getting this wrong is a 10–20× cliff, not
a percentage overhead.

| `B` (rows)    | `D` (row width) | Strategy                       | Launch                                         |
|---------------|-----------------|--------------------------------|------------------------------------------------|
| ≥ ~10× #SMs   | small (≤ ~128)  | 1 thread per row               | `grid = ceil_div(B, 256)`, `block = 256`       |
| moderate      | ≤ ~1024         | 1 warp per row                 | `grid = ceil_div(B, warps_per_block)`, `block = 32 * warps_per_block` |
| small (≤ ~1K) | large (≥ ~1K)   | **1 block per row**            | `grid = B`, `block = 256 or 512`               |
| small         | huge (≥ ~64K)   | 1 block per row + `float4`     | `grid = B`, `block = 512`                      |

Rule of thumb: you want ≥ ~4× SM count worth of *resident threads*. An A100
has 108 SMs, so aim for at least ~50K–100K concurrent threads.
`B × threads_per_row` should land in that range.

**The #1 LLM mistake here**: reusing the elementwise grid-stride template
for reductions:

```cpp
// ❌ WRONG on small-B reductions — only 128 threads run, GPU is ~1% busy
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < B) {
    float s = 0.0f;
    for (int i = 0; i < D; ++i) s += x[row*D + i];
    y[row] = s;
}
```

At `B=128, D=16384` this is ~17× slower than the 1-block-per-row template
below. Each row sequentially scans 16K floats on a single thread.

**Float4 in the inner loop** (Rule #2): for `D % 4 == 0`, load via
`float4* xr4 = reinterpret_cast<const float4*>(xr);` inside the per-thread
accumulate. Measured 1.23–1.38× speedup on `sum_dim`, `layer_norm`,
`l2_norm` at `D=16384`. The exception is online softmax: the `(local_max,
local_sum)` pair depends sequentially on each element, so the inner update
can't vectorize — only the outer component-wise `acc += v.x+v.y+v.z+v.w`
is vectorizable.

## Reduction skeleton (row-wise, e.g. softmax / rms_norm / layer_norm)

```cpp
template <int BLOCK>
__global__ void row_kernel(const float* __restrict__ x,
                           float* __restrict__ y,
                           int64_t D) {
    int row = blockIdx.x;
    const float* xr = x + row * D;
    float*       yr = y + row * D;

    // Pass 1: reduce.
    float acc = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        acc += v * v;  // rms_norm example
    }
    // Warp reduce.
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    __shared__ float warp_sum[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < BLOCK / 32) {
        acc = warp_sum[threadIdx.x];
        for (int o = (BLOCK / 32) >> 1; o > 0; o >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, o);
        if (threadIdx.x == 0) warp_sum[0] = acc;
    }
    __syncthreads();
    float scale = rsqrtf(warp_sum[0] / D + 1e-5f);

    // Pass 2: write.
    for (int i = threadIdx.x; i < D; i += BLOCK) yr[i] = xr[i] * scale;
}
```

Launch: `row_kernel<256><<<B, 256, 0, stream>>>(x, y, D)` with one block per row.

## Softmax Recipe

Row-wise softmax over the last dim (`[B, D]` → softmax over `dim=1`).
Applies to any row-wise normalising reduction (softmax, log_softmax,
masked attention softmax). Phase B.10 raw-CUDA result: **244 µs = 0.94×
PyTorch** on `[4096, 16384]` using the template below.

**Non-negotiables (correctness):**

1. **Subtract the row max before `__expf()`.** Never compute `__expf(x)`
   directly. Raw exp saturates to `+inf` for `x > 88.7` (fp32). The identity
   `softmax(x) = softmax(x − max(x))` is free and mandatory.
2. **Accumulate `max` and `sum` in fp32** even if input is fp16/bf16.
3. **Identity for max-reduction is `-FLT_MAX`**, not `0.0f`. Using `0` corrupts
   all-negative rows.
4. **Use the single-kernel online (Milakov-Gimelshein) formulation** — a
   two-kernel `stats_kernel` + `apply_kernel` split is ~1.3× slower due to
   the extra launch + the round-trip through a `row_max[]` / `row_sum[]`
   auxiliary buffer.

**Decomposition by row length:**

- `D ≤ 32`: one warp per row. `blockDim = (32, num_rows, 1)`. No smem.
- `32 < D ≤ 4096`: one block per row, `blockDim = 256`. Smem: one slot per warp.
- `D > 4096`: same single-block template with a stride loop — usually still fits.
- `D ≫ 64K`: split-row multi-block (rare).

**Canonical template (one block per row, `BLOCK=256`):**

```cpp
template <int BLOCK>
__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y, int D) {
    static_assert(BLOCK % 32 == 0, "BLOCK must be a multiple of 32");
    constexpr int NUM_WARPS = BLOCK / 32;

    const int row    = blockIdx.x;
    const int lane   = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;

    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;

    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];

    // Pass 1 — online max+sum, strided scan.
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, v);
        local_sum *= __expf(old_max - local_max);  // rescale running sum
        local_sum += __expf(v - local_max);
    }

    // Block-wide max: warp XOR-butterfly → smem → warp 0 butterfly → broadcast.
    float warp_max = local_max;
    for (int o = 16; o > 0; o >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, o));
    if (lane == 0) smem_max[warpid] = warp_max;
    __syncthreads();
    float block_max;
    if (warpid == 0) {
        float v = (lane < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (lane == 0) smem_max[0] = v;
    }
    __syncthreads();
    block_max = smem_max[0];

    // Rescale partial sums from local_max → block_max, then block-wide sum.
    local_sum *= __expf(local_max - block_max);
    float warp_sum = local_sum;
    for (int o = 16; o > 0; o >>= 1)
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, o);
    if (lane == 0) smem_sum[warpid] = warp_sum;
    __syncthreads();
    float block_sum;
    if (warpid == 0) {
        float v = (lane < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) smem_sum[0] = v;
    }
    __syncthreads();
    block_sum = smem_sum[0];

    const float inv_sum = 1.0f / block_sum;

    // Pass 2 — write normalized output. Recompute exp() rather than caching.
    for (int i = threadIdx.x; i < D; i += BLOCK)
        yr[i] = __expf(xr[i] - block_max) * inv_sum;
}
```

**Pitfalls:**

- Two-kernel `stats` + `apply` split: ~1.3× slower on `[128, 4096]`. Only
  split if a row must span blocks.
- Using `0.0f` as max identity → wrong answer on all-negative rows.
- Computing `1.0f / block_sum` in the pass-2 write loop: hoist outside.
  Scalar division is ~4× slower than multiply.
- **`__shfl_down_sync(mask, …)` with a partial-mask branch** hangs on
  sm_80+. Always have all 32 lanes of warp 0 enter the branch — see the
  "Common mistakes" section.
- **`log_softmax`**: drop the division, subtract `logf(block_sum)` instead.
  `y[i] = (x[i] - block_max) - logf(block_sum)`. Same structure.
- **Masked softmax, fully-masked row**: guard `inv_sum = block_sum > 0 ? 1/block_sum : 0` to avoid NaN.

Full reference: `examples/kernelbench-b/cuda/softmax.cu`.

## GEMM Recipe (register-tiled SGEMM)

For dense FP32 matmul `y[M, N] = x[M, K] · W[N, K]ᵀ` (PyTorch `nn.Linear`
convention). Phase C result: **hand-written raw CUDA ≈ cuBLAS-TF32 within 6%**
on `M=128, N=K=8192` using the recipe below. The short shared-memory-only
skeleton that used to be in this section landed at ~3–4× cuBLAS; the
difference is the 8×8 register tile.

**Tile parameters** (for `M, N, K` all multiples of 128 / 128 / 8):

```cpp
#define BM 128   // output tile rows per block
#define BN 128   // output tile cols per block
#define BK 8     // K chunk per shared-memory load
#define TM 8     // register tile rows per thread
#define TN 8     // register tile cols per thread
// block = 16 × 16 = 256 threads; each thread owns an 8×8 = 64-output sub-tile
```

**Skeleton** (abbreviated from `examples/kernelbench-c/cuda/gemm_mul_lrelu.cu`):

```cpp
__global__ void gemm_kernel(const float* __restrict__ X,   // [M, K]
                            const float* __restrict__ W,   // [N, K]
                            float* __restrict__ Y,         // [M, N]
                            int M, int N, int K) {
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
    const int thread_row = threadIdx.y;                      // 0..15
    const int thread_col = threadIdx.x;                      // 0..15

    __shared__ float As[BK][BM];   // transposed: As[k][m] — enables float4 load of inner dot
    __shared__ float Bs[BK][BN];   // Bs[k][n]

    // Each thread loads 1 float4 per outer K step into each tile.
    const int a_row = tid >> 1;                 // 0..127
    const int a_col = (tid & 1) << 2;           // 0 or 4 — 4-wide K slice

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
    }
    float a_reg[TM], b_reg[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // float4 gmem→smem (transposed store so inner dot is a float4 load).
        float4 va = *reinterpret_cast<const float4*>(&X[(bm + a_row) * K + k0 + a_col]);
        As[a_col+0][a_row] = va.x; As[a_col+1][a_row] = va.y;
        As[a_col+2][a_row] = va.z; As[a_col+3][a_row] = va.w;
        float4 vb = *reinterpret_cast<const float4*>(&W[(bn + a_row) * K + k0 + a_col]);
        Bs[a_col+0][a_row] = vb.x; Bs[a_col+1][a_row] = vb.y;
        Bs[a_col+2][a_row] = vb.z; Bs[a_col+3][a_row] = vb.w;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 8-wide register fan from smem via 2 float4 loads.
            float4 a0 = *reinterpret_cast<const float4*>(&As[k][thread_row*TM + 0]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[k][thread_row*TM + 4]);
            a_reg[0]=a0.x; a_reg[1]=a0.y; a_reg[2]=a0.z; a_reg[3]=a0.w;
            a_reg[4]=a1.x; a_reg[5]=a1.y; a_reg[6]=a1.z; a_reg[7]=a1.w;
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][thread_col*TN + 0]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][thread_col*TN + 4]);
            b_reg[0]=b0.x; b_reg[1]=b0.y; b_reg[2]=b0.z; b_reg[3]=b0.w;
            b_reg[4]=b1.x; b_reg[5]=b1.y; b_reg[6]=b1.z; b_reg[7]=b1.w;

            // 64 FMAs per k — the whole point of register tiling.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
            }
        }
        __syncthreads();
    }

    // Epilogue: vectorized float4 store (thread_col*TN is a multiple of 8).
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = bm + thread_row * TM + i;
        float4* yptr = reinterpret_cast<float4*>(&Y[row * N + bn + thread_col * TN]);
        yptr[0] = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
        yptr[1] = {acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
    }
}

// Host:
dim3 block(16, 16, 1);
dim3 grid(N / BN, M / BM, 1);
gemm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(X, W, Y, M, N, K);
```

**Non-negotiables:**

1. **Transposed `As[k][m]` layout** so the inner `a_reg` fan is a `float4`
   load of contiguous `As[k][thread_row*TM..]`. A non-transposed layout would
   force 8 scalar smem loads per `k`.
2. **`#pragma unroll` everywhere** — the k-loop, i-loop, j-loop. Without it,
   `acc[8][8]` spills to local memory and you lose 5–10×.
3. **Per-thread `float4` gmem load** (not scalar). Each thread loads exactly
   one `float4` for A and one for B per outer step — `256 × 4 = 1024 = BM·BK`.
4. **Epilogue `float4` store** — `thread_col * TN = 0, 8, 16, …, 120` is
   always 16-byte aligned.

**When to deviate:**

- `M, N, K` not multiples of 128/128/8: pad host-side or write a tail kernel.
- Smaller problems (`M, N ≤ 256`): fall back to a 32×32 tile + 4×4 register tile;
  128×128 wastes SMs when `gridDim < 2 × #SMs`.
- **Use cuBLAS for dense FP32 matmul at meaningful sizes.** cuBLAS with
  TF32 dispatches tensor cores, winning ~3–4× over any FP32 SGEMM. This
  recipe is competitive with cuBLAS *FP32* (no TF32), and is the right
  choice when you need precise FP32, a fused epilogue, or a non-standard
  shape.

**Fused epilogue**: put bias prefetch and activation (`relu`, `gelu`,
`leaky_relu`, `mish`, `hardswish`, `scale + res_add`, etc.) between the
`acc[]` computation and the `float4` store. Zero extra memory traffic.

## Convolution Recipe

**Key insight**: for raw CUDA, **direct convolution (no shared memory) often
wins** on small kernel sizes. Phase C measured the simple "one output per
thread, all Cin reads from gmem, `#pragma unroll` Kh/Kw" pattern at **103 µs**
on `[128, 64, 128, 128]` conv3x3, vs 336 µs for a shared-mem-tiled version.
L1/L2 serve the overlapping reads well enough that the cost of the smem
staging outweighs the win from shared-memory reuse at `Kh=Kw=3`.

This is opposite to the SeGuRu equivalent, which **needs** shared-mem tiling
because of safety-layer bounds-check overhead on gmem loads.

**Direct convolution skeleton (3×3, stride 1, no padding):**

```cpp
#define TH 16
#define TW 16

__global__ void conv_relu_biasadd_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const float* __restrict__ B1, const float* __restrict__ B2,
    float* __restrict__ Y,
    int B, int Cin, int H, int Wd, int Cout, int Kh, int Kw, int Ho, int Wo) {
    const int wo = blockIdx.x * TW + threadIdx.x;
    const int ho = blockIdx.y * TH + threadIdx.y;
    const int bc = blockIdx.z;   // 0 .. B*Cout
    const int bi = bc / Cout;
    const int co = bc - bi * Cout;
    if (wo >= Wo || ho >= Ho) return;

    const int xbs = Cin * H * Wd;
    const int wcs = Cin * Kh * Kw;

    float acc = 0.0f;
    const float* x_batch = X + bi * xbs;
    const float* w_chan  = W + co * wcs;

    for (int ci = 0; ci < Cin; ++ci) {
        const float* x_p = x_batch + ci * H * Wd + ho * Wd + wo;
        const float* w_p = w_chan  + ci * Kh * Kw;
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                acc += x_p[kh * Wd + kw] * w_p[kh * Kw + kw];
            }
        }
    }

    // Fused epilogue: conv_bias → ReLU → extra_bias.
    float v = acc + B1[co];
    v = v > 0.0f ? v : 0.0f;
    v += B2[co];
    Y[((bi * Cout + co) * Ho + ho) * Wo + wo] = v;
}

// Host:
dim3 block(TW, TH, 1);
dim3 grid((Wo + TW - 1)/TW, (Ho + TH - 1)/TH, B * Cout);
```

**Non-negotiables:**

1. **`#pragma unroll` kh and kw** — turns the inner loop into straight-line
   FMAs. Without it, the inner loop dominates.
2. **`blockIdx.z` packs `(batch, out_channel)`** — pulls Cout into the grid
   so each block's `w_chan` pointer is fixed and L1/L2 amortizes the weight
   reads across the `TH × TW` output tile.
3. **Contiguous output write per thread** — `wo` varies innermost so writes
   coalesce.

**When to use shared-mem tiling instead:**

- Very large `Kh × Kw` (depthwise conv with `Kh = Kw = 7` or more): the
  per-thread `Cin × Kh × Kw` FMA count dominates and smem reuse starts
  paying off.
- Memory-bound conv with small `Cin`: the pattern above reads each input
  pixel ~9 times from gmem; if L1 misses become the bottleneck, stage via
  smem.
- **Register tiling the output side** (each thread computes N×M output
  pixels): closes the remaining gap to cuDNN. This is the next
  optimization axis beyond this recipe.

Full reference: `examples/kernelbench-c/cuda/conv_relu_biasadd.cu`.

## Matmul skeleton (shared-memory tile, no register tile)

Kept for historical reference and as a starting point when register tiling
is overkill (small K, small `gridDim`, or debugging). Prefer the "GEMM
Recipe" above for performance.

For `C = A @ B`, `A:[M,K]`, `B:[K,N]`:

- `__shared__ float As[TM][TK];` `__shared__ float Bs[TK][TN];`
- Each block computes `[TM x TN]` tile of C. Each thread computes `TM*TN / (blockDim.x*blockDim.y)` outputs.
- Loop over K in chunks of TK: cooperative gmem→smem load, `__syncthreads`, inner dot.
- Typical: `TM=TN=64, TK=16`, `blockDim=(16,16)`, 4 outputs per thread.
- For best perf use `float4` vectorized loads of A/B into smem when K % 4 == 0.

Expect 0.25–0.35× the register-tiled GEMM recipe at N=2048 (i.e. 3–4×
slower). Use only when register tiling doesn't fit the problem shape.

## Launch Config & Occupancy

Same hardware constraints as the SeGuRu skill doc — the CUDA abstraction
doesn't change the occupancy math.

**Defaults that almost always work:**

- **1-D elementwise / reductions**: `blockDim = 256`, grid =
  `min(ceil_div(n, block*vec), 65535)` or grid-stride for large `n`.
  Move to 128 if the kernel spills; move to 512 only if measured better.
- **2-D tile kernels (GEMM, conv)**: `blockDim = (16, 16) = 256` or
  `(32, 8) = 256`. Innermost (`threadIdx.x`) must map to the stride-1
  memory dimension.
- **One warp per row**: only for rows ≤ 32. Above that, a full block
  (8 warps) gives much better latency hiding.
- **Avoid `blockDim = 1024`**: max 2 resident blocks/SM → scheduler has
  only 2 block-state machines → poor latency hiding on memory-bound
  kernels. Prefer 256 or 512.
- **Always a multiple of 32.** A 48-thread block wastes half of the last
  warp (fp32 throughput drops to 48/64 = 75%).

**Check register spills** (compile with `-Xptxas -v` or inspect PTX):

```bash
nvcc -Xptxas -v -arch=sm_80 my_kernel.cu 2>&1 | grep -E "registers|spill"
```

Any `stack frame` or `spill stores` bytes ≠ 0 = register spill. Fix by:
1. Reducing tile size (e.g., 128×128 → 64×64 for GEMM),
2. Reducing per-thread register footprint (fewer accumulators),
3. Splitting the kernel into two kernels.

Target ≤ 32 regs/thread for 100% occupancy at `blockDim=256`; ≤ 64 for 50%.
For the GEMM Recipe (64 `acc` + 8 `a_reg` + 8 `b_reg` = ~80 regs/thread),
occupancy drops to ~37% — that's fine; the register tile beats occupancy.

**Tail-block guard** — every elementwise kernel with grid-stride should
still be correct when `grid * block > n`. Grid-stride's `i < n` guard
already handles this; no extra branch.

**Tail effects (launch overhead)**: when `gridDim < 4 × #SMs`, the tail
block dominates total time. For small problems (`n < 100K` elements),
launch overhead (~5–6 µs on A100) may exceed kernel time — consider
fusing neighbouring kernels.

## Shared-Memory Bank Conflicts

CUDA shared memory has a 32-bank, 4-byte-wide layout. Bank index for
element `smem[i]` with `sizeof(T) = 4`: `i % 32`.

**When bank conflicts bite:**

- A 32×32 fp32 tile accessed column-wise: consecutive threads hit
  `i * 32 % 32 = 0` → 32-way conflict, **32× slowdown** for that access.
- GEMM kernels that store a transposed tile in smem and then column-read it.
- Histogram kernels with un-padded shared bins.

**Fix**: pad the row by 1 element.

```cpp
__shared__ float tile_a[TILE_M][TILE_K + 1];  // pad last dim by 1
//                                  ^^^^^^^^
```

Effect: element `tile_a[row][col]` maps to bank
`(row * (TILE_K + 1) + col) % 32`. When 32 threads access `[0..32][col]`
(column-wise), rows shift by `(TILE_K+1) % 32 ≠ 0`, breaking the conflict.

**When NOT to pad:**

- Row-major smem with stride-1 reads (the GEMM Recipe's `As[k][m]` inner
  `float4` load on `As[k][thread_row*TM..]` is stride-1 → no conflict,
  no padding needed).
- fp16 smem — two fp16 per bank → padding must be 2, not 1. Prefer
  storing `half2` pairs to avoid this.

**Verify**: `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`
should report ≤ 5% of smem transactions. If it doesn't, your pad is
wrong-dimension or wrong-size.

## Warp Divergence

SIMT: all 32 lanes of a warp execute the same instruction per cycle.
Divergent branches execute each path serially with inactive lanes masked —
cost = sum of both paths.

| Class | Example | Cost | Action |
|---|---|---|---|
| Warp-uniform | `if (blockIdx.x == 0)` | 0 | none |
| Geometry | `if (lane < NUM_WARPS)` | ≤ 1 cycle | none |
| Data-dependent | `if (x[i] > 0) { … } else { … }` | sum of both paths | consider restructure |
| Boundary | `if (idx < N)` (tail guard) | ≤ 1 cycle on non-tail warps | keep; correctness |

**Rule of thumb:** don't spend time on divergence in a memory-bound
kernel — memory latency dominates. Reduction/softmax kernels gain < 5%.
Compute-bound kernels (GEMM, elementwise transcendentals) can see 20–40%.

**Boundary divergence trick — loop peeling**: split into "full-warp"
iterations (no bounds check) and a tail (with check):

```cpp
int n_full = n - (n % 32);
for (int i = tid; i < n_full; i += gridDim.x * blockDim.x) { /* no guard */ }
for (int i = n_full + tid; i < n; i += gridDim.x * blockDim.x) { /* guard */ }
```

## Debugging Checklist

When a CUDA kernel produces wrong output:

1. **Reproduce on the smallest failing input.** Try sizes
   `{31, 63, 64, 65, 127, 128, 129}` — power-of-two hides boundary bugs.
2. **Classify the error pattern:**
   - All elements off by a constant factor → epilogue bug (wrong scale,
     missing eps, wrong dtype cast).
   - Last few rows/cols wrong → tail-block bounds guard missing.
   - Random wrong values → race condition (missing `__syncthreads()` after
     smem write).
   - NaN → division by zero (empty reduction, fully-masked softmax) or
     `sqrtf`/`logf` of negative.
   - Inf → forgot max-subtraction in softmax.
3. **Sentinel-fill the output buffer** before launch with `NAN`, not `0.0f`.
   Forgotten writes silently return 0 otherwise, hiding bugs.
4. **Single-block isolation**: `if (blockIdx.x != 0) return;` at kernel top.
   Removes inter-block ordering effects.
5. **Run under `compute-sanitizer`:**
   ```bash
   compute-sanitizer --tool memcheck ./my_extension_test.py
   ```
   Catches OOB reads/writes, uninitialized smem, misaligned vector loads.
   Mandatory for non-deterministic bugs.
6. **Compare against a minimal CPU reference** element-by-element on the
   smallest failing input. Use `fp64` CPU if numerical drift is suspected.
7. **After fix**: test at `{31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
   256, 257, 1023, 1024, 1025}` before declaring done.

**Common CUDA-specific traps:**

- `__shfl_*_sync` with a partial-mask branch → deadlock (see Common mistakes #9).
- `TORCH_CHECK(x.is_contiguous())` missing + user passed a transposed
  tensor → silently wrong answer.
- Dropped `AT_CUDA_CHECK(cudaGetLastError())` in the host wrapper → kernel
  errors swallowed until the next synchronization point.

## Case Studies: Phase B/C empirical results

These are the same benchmarks documented in `cuda-to-seguru-porting-skill.md`,
showing what the CUDA arm alone achieves.

**Phase B.10 — 10 KernelBench L1 problems, LLM-generated raw CUDA** (one-shot,
Claude Sonnet sub-agent using only this skill doc):

| Problem    | PyTorch eager | Raw CUDA           | correct |
|------------|--------------:|-------------------:|:-------:|
| leaky_relu |        640.9µs |   668.3µs (0.96×)  | ✓ |
| tanh       |        654.0µs |   663.7µs (0.99×)  | ✓ |
| relu       |        653.7µs |   669.1µs (0.98×)  | ✓ |
| sigmoid    |        655.5µs |   664.5µs (0.99×)  | ✓ |
| gelu       |        663.6µs |   665.7µs (1.00×)  | ✓ |
| softmax    |        229.5µs |   244.6µs (0.94×)  | ✓ |
| layer_norm |        261.5µs |   215.3µs (1.21×)  | ✓ |
| rms_norm   |      24269.7µs | 13268.3µs (1.83×)  | ✓ |
| sum_dim    |        177.5µs |   152.9µs (1.16×)  | ✓ |
| l2_norm    |        452.9µs |   208.4µs (2.17×)  | ✓ |

Aggregate: 10/10 correct, `fast_1 = 40%`, `fast_2 = 10%`, avg speedup
**1.22× PyTorch eager**.

Memory-bound elementwise sits at 0.96–1.00× PyTorch (~5% torch overhead
is basically noise). Reductions beat PyTorch once properly parallelized.

**Phase C — 8 KernelBench L2 problems, fused GEMM/Conv + epilogue** (same
methodology):

| Problem               | PyTorch eager |  Raw CUDA          | correct |
|-----------------------|--------------:|-------------------:|:-------:|
| gemm_mul_lrelu        |        8521µs |    8952µs (0.95×)  | ✓ |
| matmul_mish_mish      |        8555µs |    9094µs (0.94×)  | ✓ |
| gemm_relu_div         |        8637µs |    8965µs (0.96×)  | ✓ |
| gemm_scale_htanh_gelu |       15430µs |   17577µs (0.88×)  | ✓ |
| matmul_scale_resadd   |       30924µs |   32938µs (0.94×)  | ✓ |
| conv_relu_hardswish   |        6562µs |    6706µs (0.98×)  | ✓ |
| conv_relu_biasadd     |        7772µs |  102937µs (0.08×)  | ✓ |
| matmul_sigmoid_sum    |       18686µs | 2094150µs (0.01×)  | ✓ |

Aggregate: 8/8 correct, avg speedup **0.72× PyTorch eager**.

Notes:
- GEMM class (5 problems): raw CUDA ≈ cuBLAS-TF32 within 6%. This is
  what the GEMM Recipe above produces.
- `conv_relu_biasadd` at 0.08× is an outlier — PyTorch here uses cuDNN's
  implicit GEMM which beats direct conv; plan accordingly for shapes where
  cuDNN's dispatcher kicks in.
- `matmul_sigmoid_sum` (0.01×) — fusion is not always the right strategy.
  For small-M / huge-K-N GEMM with a row-reduce epilogue, the two-kernel
  (cuBLAS matmul + separate reduce) plan wins by ~100× because cuBLAS
  hits tensor cores and the hand-written fused kernel pays for non-coalesced
  W reads.

**Launch overhead** (measured on A100): the driver's `cuLaunchKernel`
costs ~5.08 µs per empty kernel via `torch.cpp_extension`. For problems
where `gridDim < 4 × #SMs`, tail effects dominate — fuse neighbouring
kernels to amortize.

## Common mistakes (seen in LLM-generated kernels)

1. **Missing `<ATen/cuda/CUDAContext.h>`** when using
   `at::cuda::getCurrentCUDAStream()`. Include it explicitly.
2. **Using `c10::cuda::getCurrentCUDAStream()`** → also works but requires
   `<c10/cuda/CUDAStream.h>`.
3. **Assuming `n % 4 == 0`** without a scalar tail loop.
4. **Block size > 1024** → launch fails silently.
5. **Forgetting `TORCH_CHECK(x.is_contiguous())`** → wrong answers on
   transposed inputs.
6. **Reducing in fp16** → catastrophic precision loss. Always accumulate in fp32.
7. **`x.data_ptr<float>()` on a non-contiguous tensor** → wrong answers.
   Either `.contiguous()` first or check.
8. **Launching with `grid=0`** when `n < block` → no kernel runs, no error.
   Use `max(1, grid)`.
9. **`__shfl_down_sync(0xffffffff, ...)` from a partial-mask branch** →
   *deadlocks the kernel forever* on sm_80+ (100% GPU util, never returns).
   The mask declares all 32 lanes participating, so all 32 must enter the
   branch. For cross-warp reduces, gate on `threadIdx.x < 32` (whole warp 0)
   and have inactive lanes contribute 0:
   ```cpp
   if (threadIdx.x < 32) {
       float acc = (threadIdx.x < BLOCK/32) ? warp_sum[threadIdx.x] : 0.0f;
       for (int o = 16; o > 0; o >>= 1)
           acc += __shfl_down_sync(0xffffffff, acc, o);
   }
   ```
   Do NOT write `if (threadIdx.x < BLOCK/32) { __shfl_down_sync(0xffffffff, ...); }`.
10. **Computing `1.0f / sum` inside the write loop** — hoist the reciprocal;
    scalar division is ~4× slower than multiply.
11. **Using `expf()` instead of `__expf()`** in softmax hot loops — the
    `__` form is ~3× faster with acceptable precision for softmax.
12. **Missing `-FLT_MAX` identity for max-reduction** — `0.0f` corrupts all-
    negative rows.
13. **`__syncthreads()` inside a divergent branch** — undefined behavior.
    All threads of the block must hit the same `__syncthreads`.
14. **Forgetting a `__syncthreads()` between a smem write and a smem read**
    by a different warp → race, non-deterministic wrong answers.
15. **Reading `float4` from a non-16-byte-aligned gmem address** — silently
    wrong on some GPUs, segfaults on others. Verify `addr % 16 == 0` or
    use scalar loads for the unaligned tail.
16. **Computing with fp16 input but storing the reduction result to fp16
    too early** — keep intermediate stats in fp32, cast only at the final
    write.

## Host-side recipe

```cpp
torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    x = x.contiguous();
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    constexpr int VEC = 4;
    constexpr int BLOCK = 256;
    int64_t grid = std::min<int64_t>((n + BLOCK*VEC - 1) / (BLOCK*VEC), 65535);
    if (grid < 1) grid = 1;
    my_kernel<<<grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}
```
