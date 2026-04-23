#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void leaky_relu_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int64_t n, float slope
) {
    int64_t n4 = n >> 2;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = __ldg(x4 + i);
        v.x = v.x > 0.f ? v.x : v.x * slope;
        v.y = v.y > 0.f ? v.y : v.y * slope;
        v.z = v.z > 0.f ? v.z : v.z * slope;
        v.w = v.w > 0.f ? v.w : v.w * slope;
        y4[i] = v;
    }

    // tail
    int64_t tail_start = n4 << 2;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        float v = __ldg(x + i);
        y[i] = v > 0.f ? v : v * slope;
    }
}

torch::Tensor run(torch::Tensor x, float slope) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;

    const int block = 256;
    int64_t n4 = n >> 2;
    int64_t work = n4 > 0 ? n4 : n;
    int64_t grid64 = (work + block - 1) / block;

    // Cap grid to keep grid-stride loop efficient; tuned for large n.
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int max_grid = sm_count * 32;  // plenty of waves
    int grid = (int)(grid64 < (int64_t)max_grid ? grid64 : (int64_t)max_grid);
    if (grid < 1) grid = 1;

    auto stream = at::cuda::getCurrentCUDAStream();
    leaky_relu_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n, slope);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "LeakyReLU forward");
}
