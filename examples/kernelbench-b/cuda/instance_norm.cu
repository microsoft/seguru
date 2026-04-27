#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>

template <int BLOCK>
__global__ void instance_norm_stats_kernel(const float4* __restrict__ x4,
                                           float* __restrict__ mean,
                                           float* __restrict__ rstd,
                                           int hw4,
                                           float eps) {
    __shared__ float s_sum[BLOCK];
    __shared__ float s_sumsq[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float4* xr = x4 + (int64_t)row * hw4;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < hw4; i += BLOCK) {
        float4 v = xr[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
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
        float inv_n = 1.0f / (static_cast<float>(hw4) * 4.0f);
        float m = s_sum[0] * inv_n;
        float var = s_sumsq[0] * inv_n - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean[row] = m;
        rstd[row] = rsqrtf(var + eps);
    }
}

__global__ void instance_norm_apply_kernel(const float4* __restrict__ x4,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ rstd,
                                           float4* __restrict__ y4,
                                           int hw4,
                                           int total4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total4) return;

    int row = idx / hw4;
    float m = mean[row];
    float rs = rstd[row];
    float4 v = x4[idx];
    y4[idx] = make_float4(
        (v.x - m) * rs,
        (v.y - m) * rs,
        (v.z - m) * rs,
        (v.w - m) * rs);
}

torch::Tensor run(torch::Tensor x) {
    constexpr int BLOCK = 256;
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 4, "instance_norm expects [B,C,H,W]");
    const int64_t B64 = x.size(0);
    const int64_t C64 = x.size(1);
    const int64_t H64 = x.size(2);
    const int64_t W64 = x.size(3);
    TORCH_CHECK(B64 > 0 && C64 > 0 && H64 > 0 && W64 > 0,
                "instance_norm requires non-empty B, C, H, and W");
    const int64_t hw64 = H64 * W64;
    const int64_t total64 = x.numel();
    const int64_t rows64 = B64 * C64;
    TORCH_CHECK(hw64 % 4 == 0, "instance_norm spatial size must be divisible by float4 width");
    TORCH_CHECK(total64 % 4 == 0, "instance_norm total elements must be divisible by float4 width");
    TORCH_CHECK(hw64 / 4 <= std::numeric_limits<int>::max() - BLOCK,
                "instance_norm spatial size exceeds int-indexed stats kernel loop limit");
    TORCH_CHECK(total64 / 4 <= std::numeric_limits<int>::max(),
                "instance_norm input exceeds int-indexed kernel limit");
    TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(),
                "instance_norm row count exceeds int-indexed kernel limit");
    const int hw4 = static_cast<int>(hw64 / 4);
    const int total4 = static_cast<int>(total64 / 4);
    const int rows = static_cast<int>(rows64);

    at::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto mean = torch::empty({rows64}, x.options());
    auto rstd = torch::empty({rows64}, x.options());
    auto y = torch::empty_like(x);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(x.data_ptr<float>()) % alignof(float4) == 0,
                "x data pointer must be 16-byte aligned for float4 loads");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(y.data_ptr<float>()) % alignof(float4) == 0,
                "y data pointer must be 16-byte aligned for float4 stores");
    const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>());
    float4* y4 = reinterpret_cast<float4*>(y.data_ptr<float>());

    instance_norm_stats_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(
        x4, mean.data_ptr<float>(), rstd.data_ptr<float>(), hw4, 1e-5f);
    AT_CUDA_CHECK(cudaGetLastError());
    const int64_t grid_apply64 = (static_cast<int64_t>(total4) + BLOCK - 1) / BLOCK;
    TORCH_CHECK(grid_apply64 <= std::numeric_limits<int>::max(),
                "instance_norm apply grid exceeds int-indexed kernel limit");
    const int grid_apply = static_cast<int>(grid_apply64);
    instance_norm_apply_kernel<<<grid_apply, BLOCK, 0, stream>>>(
        x4, mean.data_ptr<float>(), rstd.data_ptr<float>(), y4, hw4, total4);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "InstanceNorm2d forward, no affine"); }
