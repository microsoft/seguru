// Argmin over dim=1 for [B, D1, D2] -> [B, D2] (int64)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>
#include <limits.h>

template <int BLOCK>
__global__ void argmin_dim_kernel(const float* __restrict__ x, long long* __restrict__ y,
                                  int B, int D1, int D2) {
    const int out = blockIdx.x;
    const int tid = threadIdx.x;
    const int b = out / D2;
    const int k = out - b * D2;

    float best_v = FLT_MAX;
    int best_i = INT_MAX;
    for (int i = tid; i < D1; i += BLOCK) {
        const float v = x[((int64_t)b * D1 + i) * D2 + k];
        if (v < best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float s_val[BLOCK];
    __shared__ int s_idx[BLOCK];
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rv = s_val[tid + stride];
            const int ri = s_idx[tid + stride];
            const float lv = s_val[tid];
            const int li = s_idx[tid];
            if (rv < lv || (rv == lv && ri < li)) {
                s_val[tid] = rv;
                s_idx[tid] = ri;
            }
        }
        __syncthreads();
    }

    if (tid == 0) y[out] = (long long)s_idx[0];
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int B = (int)x.size(0), D1 = (int)x.size(1), D2 = (int)x.size(2);
    auto y = torch::empty({B, D2}, x.options().dtype(torch::kInt64));
    constexpr int BLOCK = 256;
    argmin_dim_kernel<BLOCK><<<B * D2, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), (long long*)y.data_ptr<int64_t>(), B, D1, D2);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "argmin over dim=1 (i64)"); }
