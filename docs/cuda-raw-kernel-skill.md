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
   on memory-bound kernels.
3. **Use `__ldg` for read-only loads** on older arches; on Ampere+ this is
   already implicit for `const __restrict__`, but `__ldg` is still safe.
4. **Stream awareness.** Always launch on
   `at::cuda::getCurrentCUDAStream()` so torch's stream machinery works.
   Include `<ATen/cuda/CUDAContext.h>`. Do NOT create your own stream.
5. **Block size = 256 or 512** unless you have a reason. Grid size =
   `min(div_ceil(n, block*vec), 65535)` or let grid-stride handle large n.
6. **Shared-memory reductions.** For row-wise reductions (softmax, rms_norm,
   layer_norm, sum_dim): one block per row, `blockDim.x` threads, two
   stages — per-thread accumulate, then block reduce via `__shfl_down_sync`
   within each warp followed by a `__shared__` stage across warps. Put
   `__syncthreads()` between the two stages.
7. **fp32 accumulate** for any reduction, even if input/output are fp16/bf16.
   Cast to `float` inside the loop.
8. **Guard tails.** After vectorized main loop, handle `n % vec` remainder
   in a scalar tail loop. Cheaper than padding.
9. **Don't allocate inside the kernel.** Use `torch::empty_like(x, x.options())`
   or `torch::empty({...}, x.options())` on the host wrapper.
10. **Assert contiguity + dtype** with `TORCH_CHECK(x.is_cuda())`,
    `TORCH_CHECK(x.is_contiguous())`, `TORCH_CHECK(x.dtype() == torch::kFloat32)`.

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

## Matmul skeleton (shared-memory tile)

For `C = A @ B`, `A:[M,K]`, `B:[K,N]`:

- `__shared__ float As[TM][TK];` `__shared__ float Bs[TK][TN];`
- Each block computes `[TM x TN]` tile of C. Each thread computes `TM*TN / (blockDim.x*blockDim.y)` outputs.
- Loop over K in chunks of TK: cooperative gmem→smem load, `__syncthreads`, inner dot.
- Typical: `TM=TN=64, TK=16`, `blockDim=(16,16)`, 4 outputs per thread.
- For best perf use `float4` vectorized loads of A/B into smem when K % 4 == 0.

Full double-buffered / tensor-core versions are out of scope for one-shot gen;
the above gets you 0.8–1.2× cuBLAS on square fp32 at N=2048.

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
