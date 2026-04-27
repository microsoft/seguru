#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <int BLOCK>
__global__ void batch_norm_stats_kernel(const float* __restrict__ x,
                                        float* __restrict__ mean,
                                        float* __restrict__ rstd,
                                        int B, int C, int HW, float eps) {
    __shared__ float s_sum[BLOCK];
    __shared__ float s_sumsq[BLOCK];
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int count = B * HW;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < count; i += BLOCK) {
        int b = i / HW;
        int hw = i - b * HW;
        float v = x[((int64_t)b * C + c) * HW + hw];
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
        float inv_count = 1.0f / (float)count;
        float m = s_sum[0] * inv_count;
        float var = s_sumsq[0] * inv_count - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean[c] = m;
        rstd[c] = rsqrtf(var + eps);
    }
}

__global__ void batch_norm_apply_kernel(const float* __restrict__ x,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        float* __restrict__ y,
                                        int64_t N, int C, int HW) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int c = (int)((idx / HW) % C);
    y[idx] = (x[idx] - mean[c]) * rstd[c];
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 4, "batch_norm expects [B,C,H,W]");
    int B = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int HW = H * W;
    auto y = torch::empty_like(x);
    auto mean = torch::empty({C}, x.options());
    auto rstd = torch::empty({C}, x.options());
    constexpr int BLOCK = 256;
    auto stream = at::cuda::getCurrentCUDAStream();
    batch_norm_stats_kernel<BLOCK><<<C, BLOCK, 0, stream>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), B, C, HW, 1e-5f);
    AT_CUDA_CHECK(cudaGetLastError());
    int64_t N = x.numel();
    int grid = (int)((N + BLOCK - 1) / BLOCK);
    batch_norm_apply_kernel<<<grid, BLOCK, 0, stream>>>(
        x.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), y.data_ptr<float>(), N, C, HW);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "BatchNorm2d training forward, no affine"); }
