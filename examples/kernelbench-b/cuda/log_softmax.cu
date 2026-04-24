// Row-wise log_softmax: y = (x - max) - log(sum(exp(x - max)))
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void log_softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
    static_assert(BLOCK % 32 == 0);
    constexpr int NW = BLOCK / 32;
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;

    __shared__ float smem_max[NW];
    __shared__ float smem_sum[NW];

    float lm = -FLT_MAX, ls = 0.f;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        float om = lm;
        lm = fmaxf(lm, v);
        ls *= __expf(om - lm);
        ls += __expf(v - lm);
    }
    float wm = warp_max(lm);
    if (lane == 0) smem_max[warpid] = wm;
    __syncthreads();
    float bm;
    if (warpid == 0) {
        float v = (lane < NW) ? smem_max[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) smem_max[0] = v;
    }
    __syncthreads();
    bm = smem_max[0];

    ls *= __expf(lm - bm);
    float ws = warp_sum(ls);
    if (lane == 0) smem_sum[warpid] = ws;
    __syncthreads();
    float bs;
    if (warpid == 0) {
        float v = (lane < NW) ? smem_sum[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) smem_sum[0] = v;
    }
    __syncthreads();
    bs = smem_sum[0];
    float log_sum = logf(bs);

    for (int i = threadIdx.x; i < D; i += BLOCK)
        yr[i] = (xr[i] - bm) - log_sum;
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    const int64_t B = x.size(0), D = x.size(1);
    auto y = torch::empty_like(x);
    constexpr int BLOCK = 256;
    log_softmax_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "log_softmax"); }
