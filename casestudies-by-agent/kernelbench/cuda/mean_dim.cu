// Row-wise mean over last dim: [B, D] -> [B]
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void mean_dim_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t D) {
    constexpr int NW = BLOCK / 32;
    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;

    float acc = 0.f;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    int64_t D4 = D >> 2;
    for (int64_t i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = __ldg(xr4 + i);
        acc += v.x + v.y + v.z + v.w;
    }
    for (int64_t i = (D4 << 2) + threadIdx.x; i < D; i += BLOCK) {
        acc += __ldg(xr + i);
    }
    acc = warp_sum(acc);
    __shared__ float smem[NW];
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = acc;
    __syncthreads();
    if (warpid == 0) {
        float v = (lane < NW) ? smem[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) y[row] = v / (float)D;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    int64_t B = x.size(0), D = x.size(1);
    auto y = torch::empty({B}, x.options());
    constexpr int BLOCK = 256;
    mean_dim_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "mean over last dim"); }
