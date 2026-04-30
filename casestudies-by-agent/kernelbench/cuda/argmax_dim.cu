// Row-wise argmax over last dim: [B, D] -> [B] (int64)
// Two-pass within a single kernel: (1) block_max, (2) lowest index with value==max.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>
#include <limits.h>

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ int warp_min_i(int v) {
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

template <int BLOCK>
__global__ void argmax_dim_kernel(const float* __restrict__ x, long long* __restrict__ y, int D) {
    constexpr int NW = BLOCK / 32;
    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;

    float acc = -FLT_MAX;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        acc = fmaxf(acc, v);
    }
    acc = warp_max(acc);
    __shared__ float smax[NW];
    __shared__ int   smin[NW];
    int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    if (lane == 0) smax[warpid] = acc;
    __syncthreads();
    float bm;
    if (warpid == 0) {
        float v = (lane < NW) ? smax[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) smax[0] = v;
    }
    __syncthreads();
    bm = smax[0];

    int local_idx = INT_MAX;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        if (v == bm && i < local_idx) local_idx = i;
    }
    int wi = warp_min_i(local_idx);
    if (lane == 0) smin[warpid] = wi;
    __syncthreads();
    if (warpid == 0) {
        int v = (lane < NW) ? smin[lane] : INT_MAX;
        v = warp_min_i(v);
        if (lane == 0) y[row] = (long long)v;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    int64_t B = x.size(0), D = x.size(1);
    auto y = torch::empty({B}, x.options().dtype(torch::kInt64));
    constexpr int BLOCK = 256;
    argmax_dim_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), (long long*)y.data_ptr<int64_t>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "argmax over last dim (i64)"); }
