// Row-wise masked cumulative sum over last dim: cumsum(x * mask, dim=-1)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>

template <int BLOCK>
__global__ void masked_cumsum_kernel(const float* __restrict__ x, const float* __restrict__ mask,
                                     float* __restrict__ y, int D) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int items = D / BLOCK;
    const int base = tid * items;
    const float* xr = x + (int64_t)row * D;
    const float* mr = mask + (int64_t)row * D;
    float* yr = y + (int64_t)row * D;

    float run = 0.0f;
    for (int j = 0; j < items; ++j) {
        run += xr[base + j] * mr[base + j];
        yr[base + j] = run;
    }

    smem[tid] = run;
    __syncthreads();
    for (int off = 1; off < BLOCK; off <<= 1) {
        const float v = (tid >= off) ? smem[tid - off] : 0.0f;
        __syncthreads();
        smem[tid] += v;
        __syncthreads();
    }
    const float offset = (tid == 0) ? 0.0f : smem[tid - 1];
    for (int j = 0; j < items; ++j) yr[base + j] += offset;
}

torch::Tensor run(torch::Tensor x, torch::Tensor mask) {
    TORCH_CHECK(x.is_cuda() && mask.is_cuda() && x.is_contiguous() && mask.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && mask.scalar_type() == torch::kFloat32 &&
                x.dim() == 2 && mask.dim() == 2);
    TORCH_CHECK(mask.sizes() == x.sizes());
    const int64_t B64 = x.size(0);
    const int64_t D64 = x.size(1);
    TORCH_CHECK(B64 > 0 && D64 > 0, "masked_cumsum requires non-empty B and D");
    constexpr int BLOCK = 256;
    const int64_t D = D64;
    TORCH_CHECK(D % BLOCK == 0, "D must be divisible by BLOCK");
    TORCH_CHECK(B64 <= std::numeric_limits<int>::max(),
                "masked_cumsum B exceeds int-indexed kernel limit");
    TORCH_CHECK(D64 <= std::numeric_limits<int>::max(),
                "masked_cumsum D exceeds int-indexed kernel limit");
    const int B = static_cast<int>(B64);
    const int D_i = static_cast<int>(D64);

    at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto y = torch::empty_like(x);
    masked_cumsum_kernel<BLOCK><<<B, BLOCK, 0, stream>>>(
        x.data_ptr<float>(), mask.data_ptr<float>(), y.data_ptr<float>(), D_i);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "masked cumsum over last dim"); }
