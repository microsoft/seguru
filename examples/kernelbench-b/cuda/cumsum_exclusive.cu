// Row-wise exclusive cumulative sum over last dim: [B, D] -> [B, D]
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <int BLOCK>
__global__ void cumsum_exclusive_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
    __shared__ float smem[BLOCK];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int items = D / BLOCK;
    const int base = tid * items;
    const float* xr = x + (int64_t)row * D;
    float* yr = y + (int64_t)row * D;

    float run = 0.0f;
    for (int j = 0; j < items; ++j) {
        yr[base + j] = run;
        run += xr[base + j];
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

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    const int64_t B = x.size(0), D = x.size(1);
    constexpr int BLOCK = 256;
    TORCH_CHECK(D % BLOCK == 0, "D must be divisible by BLOCK");
    auto y = torch::empty_like(x);
    cumsum_exclusive_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "exclusive cumsum over last dim"); }
