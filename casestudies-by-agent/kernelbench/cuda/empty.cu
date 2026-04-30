// Empty kernel — measures raw torch.cpp_extension launch overhead.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void empty_kernel(const float* x, float* y, int n) {
    // empty
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    auto y = torch::empty_like(x);
    empty_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)x.numel());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "empty kernel");
}
