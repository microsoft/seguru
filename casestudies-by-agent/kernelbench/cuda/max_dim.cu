// Row-wise max over last dim: [B, D] -> [B]
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

template <int BLOCK>
__global__ void max_dim_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t D) {
    constexpr int NW = BLOCK / 32;
    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;

    float acc = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < D; i += BLOCK) {
        float v = __ldg(xr + i);
        acc = fmaxf(acc, v);
    }
    acc = warp_max(acc);
    __shared__ float smem[NW];
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    if (lane == 0) smem[warpid] = acc;
    __syncthreads();
    if (warpid == 0) {
        float v = (lane < NW) ? smem[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) y[row] = v;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    int64_t B = x.size(0), D = x.size(1);
    auto y = torch::empty({B}, x.options());
    constexpr int BLOCK = 256;
    max_dim_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "max over last dim"); }
