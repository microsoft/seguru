// MSE loss: scalar = mean((a - b)^2)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Single-block reduction over entire array (N up to ~1M is plenty for 1 block).
template <int BLOCK>
__global__ void mse_loss_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ out, int64_t N) {
    constexpr int NW = BLOCK / 32;
    float acc = 0.f;
    for (int64_t i = threadIdx.x; i < N; i += BLOCK) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    acc = warp_sum(acc);
    __shared__ float smem[NW];
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = acc;
    __syncthreads();
    if (warpid == 0) {
        float v = (lane < NW) ? smem[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) out[0] = v / (float)N;
    }
}

torch::Tensor run(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.scalar_type() == torch::kFloat32);
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kFloat32);
    TORCH_CHECK(a.numel() == b.numel());
    auto out = torch::empty({}, a.options());
    constexpr int BLOCK = 1024;
    mse_loss_kernel<BLOCK><<<1, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "mse_loss (scalar)"); }
