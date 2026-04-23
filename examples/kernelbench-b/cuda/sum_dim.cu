#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Row-wise sum: one block per row, warp-shuffle block reduction.
//
// Layout: x[B, D], y[B].
// Each thread accumulates D/BLOCK elements via float4 loads (D must be
// divisible by 4; D=16384 is). Then a two-stage block reduction:
//   1. Warp reduce with __shfl_down_sync.
//   2. Warp 0 reduces the per-warp partial sums stored in shared memory.
// Thread 0 writes y[blockIdx.x].
//
// CUDA raw-kernel-skill.md rules applied:
//   - Rule 4:  getCurrentCUDAStream().
//   - Rule 6:  warp shuffle + shared mem block reduction.
//   - Rule 7:  fp32 accumulation.
//   - Rule 10: TORCH_CHECK guards.

template <int BLOCK>
__global__ void sum_dim_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int64_t D) {
    static_assert(BLOCK % 32 == 0, "BLOCK must be a multiple of warp size");
    constexpr int WARPS = BLOCK / 32;

    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;

    // Accumulate partial sum using float4 loads.
    float acc = 0.0f;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    int64_t D4 = D / 4;
    for (int64_t i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = __ldg(xr4 + i);
        acc += v.x + v.y + v.z + v.w;
    }
    // Scalar tail for D % 4 != 0.
    for (int64_t i = D4 * 4 + threadIdx.x; i < D; i += BLOCK) {
        acc += __ldg(xr + i);
    }

    // Stage 1: warp-level reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Stage 2: collect per-warp sums in shared memory.
    __shared__ float warp_sums[WARPS];
    int lane  = threadIdx.x & 31;
    int warpid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warpid] = acc;
    __syncthreads();

    // Warp 0 reduces the WARPS partial sums.
    if (warpid == 0) {
        acc = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARPS >> 1; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) y[row] = acc;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [B, D]");

    int64_t B = x.size(0);
    int64_t D = x.size(1);

    auto y = torch::empty({B}, x.options());

    constexpr int BLOCK = 256;
    // One block per row; grid size = B (at most 65535 for B=4096, fine).
    int grid = (int)B;

    sum_dim_kernel<BLOCK><<<grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), D);
    AT_CUDA_CHECK(cudaGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "sum over last dim (row-wise)");
}
