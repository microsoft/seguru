// Row-wise L1 normalize: y = x / (sum(|x|, dim=-1, keepdim=True) + 1e-12)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void l1_norm_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
    constexpr int NW = BLOCK / 32;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;

    __shared__ float smem[NW];

    float acc = 0.f;
    const int D4 = D >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = __ldg(xr4 + i);
        acc += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }
    for (int i = (D4 << 2) + threadIdx.x; i < D; i += BLOCK) {
        acc += fabsf(__ldg(xr + i));
    }
    float ws = warp_sum(acc);
    if (lane == 0) smem[warpid] = ws;
    __syncthreads();
    float bs;
    if (warpid == 0) {
        float v = (lane < NW) ? smem[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    bs = smem[0];
    const float inv = 1.f / (bs + 1e-12f);

    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = __ldg(xr4 + i);
        v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
        yr4[i] = v;
    }
    for (int i = (D4 << 2) + threadIdx.x; i < D; i += BLOCK) {
        yr[i] = xr[i] * inv;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    const int64_t B = x.size(0), D = x.size(1);
    auto y = torch::empty_like(x);
    constexpr int BLOCK = 256;
    l1_norm_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "l1_norm"); }
