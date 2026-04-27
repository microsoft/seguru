#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <int BLOCK>
__global__ void group_norm_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int group_elems, float eps) {
    __shared__ float s_sum[BLOCK];
    __shared__ float s_sumsq[BLOCK];
    __shared__ float s_mean;
    __shared__ float s_rstd;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (int64_t)row * group_elems;
    float* yr = y + (int64_t)row * group_elems;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < group_elems; i += BLOCK) {
        float v = xr[i];
        local_sum += v;
        local_sumsq += v * v;
    }
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float inv_n = 1.0f / (float)group_elems;
        float mean = s_sum[0] * inv_n;
        float var = s_sumsq[0] * inv_n - mean * mean;
        var = var < 0.0f ? 0.0f : var;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean;
    float rstd = s_rstd;
    for (int i = tid; i < group_elems; i += BLOCK) {
        yr[i] = (xr[i] - mean) * rstd;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 4, "group_norm expects [B,C,H,W]");
    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    constexpr int GROUPS = 8;
    TORCH_CHECK(C % GROUPS == 0, "C must be divisible by 8 groups");
    int group_elems = (C / GROUPS) * H * W;
    auto y = torch::empty_like(x);
    constexpr int BLOCK = 256;
    group_norm_kernel<BLOCK><<<B * GROUPS, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), group_elems, 1e-5f);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "GroupNorm forward, no affine"); }
