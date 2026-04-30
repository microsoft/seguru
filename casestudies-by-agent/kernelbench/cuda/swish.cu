#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t n4 = n >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = __ldg(x4 + i);
        v.x = v.x * (1.f / (1.f + __expf(-v.x)));
        v.y = v.y * (1.f / (1.f + __expf(-v.y)));
        v.z = v.z * (1.f / (1.f + __expf(-v.z)));
        v.w = v.w * (1.f / (1.f + __expf(-v.w)));
        y4[i] = v;
    }
    int64_t tail = n4 << 2;
    for (int64_t i = tail + tid; i < n; i += stride) {
        float v = __ldg(x + i);
        y[i] = v * (1.f / (1.f + __expf(-v)));
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int block = 256;
    int64_t n4 = n >> 2;
    int64_t work = n4 > 0 ? n4 : n;
    int64_t grid64 = (work + block - 1) / block;
    int device = 0; cudaGetDevice(&device);
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
    int max_grid = sm * 32;
    int grid = (int)(grid64 < max_grid ? grid64 : (int64_t)max_grid);
    if (grid < 1) grid = 1;
    swish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "swish (x*sigmoid(x))"); }
