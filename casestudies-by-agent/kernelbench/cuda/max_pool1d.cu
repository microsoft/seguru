// MaxPool1d k=4 s=4 over last dim of [N, C, L] -> [N, C, L/4]
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void max_pool1d_k4s4_kernel(const float* __restrict__ x, float* __restrict__ y,
                                       int64_t N_total_out) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < N_total_out; i += stride) {
        int64_t base = i << 2;
        float a = __ldg(x + base);
        float b = __ldg(x + base + 1);
        float c = __ldg(x + base + 2);
        float d = __ldg(x + base + 3);
        y[i] = fmaxf(fmaxf(a, b), fmaxf(c, d));
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t N = x.size(0), C = x.size(1), L = x.size(2);
    TORCH_CHECK(L % 4 == 0, "L must be divisible by 4");
    const int64_t Lo = L / 4;
    auto y = torch::empty({N, C, Lo}, x.options());
    int64_t out_n = N * C * Lo;
    const int block = 256;
    int device = 0; cudaGetDevice(&device);
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
    int64_t grid64 = (out_n + block - 1) / block;
    int max_grid = sm * 32;
    int grid = (int)(grid64 < max_grid ? grid64 : (int64_t)max_grid);
    max_pool1d_k4s4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out_n);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "max_pool1d k=4 s=4"); }
