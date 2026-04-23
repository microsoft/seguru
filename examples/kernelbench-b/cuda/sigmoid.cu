#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void sigmoid_kernel(const float* __restrict__ x, float* __restrict__ y,
                                int64_t n) {
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        float4 r;
        r.x = 1.0f / (1.0f + __expf(-v.x));
        r.y = 1.0f / (1.0f + __expf(-v.y));
        r.z = 1.0f / (1.0f + __expf(-v.z));
        r.w = 1.0f / (1.0f + __expf(-v.w));
        y4[i] = r;
    }

    int64_t tail_start = n4 << 2;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        y[i] = 1.0f / (1.0f + __expf(-x[i]));
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;

    const int block = 256;
    int64_t n4 = n / 4;
    int64_t needed = (n4 + block - 1) / block;
    const int64_t max_blocks = 4096;
    int grid = (int)((needed < max_blocks) ? needed : max_blocks);
    if (grid < 1) grid = 1;

    sigmoid_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "sigmoid forward");
}
