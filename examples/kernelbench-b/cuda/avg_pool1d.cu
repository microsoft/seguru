// AvgPool1d k=8 s=1 p=4 over last dim of [N, C, L] -> [N, C, L + 1].
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void avg_pool1d_k8s1p4_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int64_t total_out,
                                         int64_t L,
                                         int64_t Lo) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = tid; idx < total_out; idx += stride) {
        int64_t pos = idx % Lo;
        int64_t bc = idx / Lo;
        int64_t base = bc * L;
        int64_t start = pos - 4;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int64_t in_pos = start + k;
            if (in_pos >= 0 && in_pos < L) sum += __ldg(x + base + in_pos);
        }
        y[idx] = sum * 0.125f;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t N = x.size(0), C = x.size(1), L = x.size(2);
    const int64_t Lo = L + 1;
    auto y = torch::empty({N, C, Lo}, x.options());
    int64_t out_n = N * C * Lo;
    const int block = 256;
    int device = 0; cudaGetDevice(&device);
    int sm = 0; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
    int64_t grid64 = (out_n + block - 1) / block;
    int max_grid = sm * 32;
    int grid = (int)(grid64 < max_grid ? grid64 : (int64_t)max_grid);
    avg_pool1d_k8s1p4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out_n, L, Lo);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "avg_pool1d k=8 s=1 p=4"); }
