// L2 normalization: for each row of [B, D]:
//     s = sqrt(sum(row^2) + 1e-12)
//     y = row / s
//
// One block per row.  Threads cooperatively reduce sum-of-squares via
// float4 vectorized loads + warp-shuffle, then apply the scale in a
// second pass.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <int BLOCK>
__global__ void l2_norm_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int D) {
    int row = blockIdx.x;
    const float* __restrict__ xr = x + (int64_t)row * D;
    float* yr = y + (int64_t)row * D;

    // ---- Pass 1: accumulate sum of squares ----
    float acc = 0.0f;

    int D4 = D / 4;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = xr4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // Scalar tail (handles D % 4 != 0).
    for (int i = D4 * 4 + threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        acc += v * v;
    }

    // Warp-level reduce.
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);

    __shared__ float warp_sum[BLOCK / 32];
    if ((threadIdx.x & 31) == 0)
        warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();

    // Cross-warp reduce: have all 32 lanes of warp 0 participate.
    if (threadIdx.x < 32) {
        acc = (threadIdx.x < BLOCK / 32) ? warp_sum[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, o);
        if (threadIdx.x == 0)
            warp_sum[0] = acc;
    }
    __syncthreads();

    float scale = rsqrtf(warp_sum[0] + 1e-12f);

    // ---- Pass 2: apply scale ----
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = threadIdx.x; i < D4; i += BLOCK) {
        float4 v = xr4[i];
        float4 r;
        r.x = v.x * scale;
        r.y = v.y * scale;
        r.z = v.z * scale;
        r.w = v.w * scale;
        yr4[i] = r;
    }
    for (int i = D4 * 4 + threadIdx.x; i < D; i += BLOCK) {
        yr[i] = xr[i] * scale;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),  "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2,       "x must be 2-D [B, D]");

    int64_t B = x.size(0);
    int64_t D = x.size(1);

    auto y = torch::empty_like(x);

    constexpr int BLOCK = 256;
    // One block per row; the kernel handles any D via stride loops.
    l2_norm_kernel<BLOCK><<<(int)B, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "L2 normalization (per-row)");
}
