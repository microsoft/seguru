#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
static constexpr float GELU_COEFF     = 0.044715f;

__device__ __forceinline__ float gelu_elem(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int64_t n) {
    int64_t tid    = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        float4 r;
        r.x = gelu_elem(v.x);
        r.y = gelu_elem(v.y);
        r.z = gelu_elem(v.z);
        r.w = gelu_elem(v.w);
        y4[i] = r;
    }

    int64_t tail_start = n4 << 2;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        y[i] = gelu_elem(x[i]);
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    x = x.contiguous();
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return y;

    constexpr int BLOCK = 256;
    int64_t n4     = n / 4;
    int64_t needed = (n4 + BLOCK - 1) / BLOCK;
    const int64_t max_blocks = 4096;
    int grid = (int)((needed < max_blocks) ? needed : max_blocks);
    if (grid < 1) grid = 1;

    gelu_kernel<<<grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "gelu tanh-approx forward");
}
