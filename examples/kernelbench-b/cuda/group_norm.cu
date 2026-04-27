#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

template <int BLOCK>
__global__ void group_norm_stats_kernel(const float4* __restrict__ x4,
                                        float* __restrict__ mean,
                                        float* __restrict__ rstd,
                                        int group_elems4,
                                        float eps) {
    __shared__ float s_sum[BLOCK];
    __shared__ float s_sumsq[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float4* xr = x4 + (int64_t)row * group_elems4;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < group_elems4; i += BLOCK) {
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
        float inv_n = 1.0f / (float)(group_elems4 * 4);
        float m = s_sum[0] * inv_n;
        float var = s_sumsq[0] * inv_n - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean[row] = m;
        rstd[row] = rsqrtf(var + eps);
    }
}

__global__ void group_norm_apply_kernel(const float4* __restrict__ x4,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        float4* __restrict__ y4,
                                        int group_elems4,
                                        int total4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total4) return;

    int row = idx / group_elems4;
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
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 4, "group_norm expects [B,C,H,W]");
    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    constexpr int GROUPS = 8;
    TORCH_CHECK(C % GROUPS == 0, "C must be divisible by 8 groups");
    int group_elems = (C / GROUPS) * H * W;
    int total = B * C * H * W;
    TORCH_CHECK(group_elems % 4 == 0, "group elements must be divisible by float4 width");
    TORCH_CHECK(total % 4 == 0, "total elements must be divisible by float4 width");
    int group_elems4 = group_elems / 4;
    int total4 = total / 4;

    auto y = torch::empty_like(x);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(x.data_ptr<float>()) % alignof(float4) == 0,
                "x data pointer must be 16-byte aligned for float4 loads");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(y.data_ptr<float>()) % alignof(float4) == 0,
                "y data pointer must be 16-byte aligned for float4 stores");
    auto mean = torch::empty({B * GROUPS}, x.options());
    auto rstd = torch::empty({B * GROUPS}, x.options());
    const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>());
    float4* y4 = reinterpret_cast<float4*>(y.data_ptr<float>());

    constexpr int BLOCK = 256;
    group_norm_stats_kernel<BLOCK><<<B * GROUPS, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x4, mean.data_ptr<float>(), rstd.data_ptr<float>(), group_elems4, 1e-5f);
    AT_CUDA_CHECK(cudaGetLastError());
    int grid_apply = (total4 + BLOCK - 1) / BLOCK;
    group_norm_apply_kernel<<<grid_apply, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x4, mean.data_ptr<float>(), rstd.data_ptr<float>(), y4, group_elems4, total4);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "GroupNorm forward, no affine"); }
