// Layer normalization (no affine transform) for a 2-D tensor [B, D].
//
// Formula (per row):
//   mean = sum(row) / D
//   var  = sum((row - mean)^2) / D
//   y    = (row - mean) / sqrt(var + eps)
//
// Design:
//   - One block per row (blockIdx.x == row).
//   - BLOCK = 256 threads per block (8 warps).
//   - Pass 1: each thread accumulates sum and sum_sq using float4 loads;
//     warp-shuffle reduces within each warp; thread 0 does the final
//     cross-warp reduce and writes mean/rstd to shared memory.
//   - Pass 2: all threads read mean/rstd and write normalized output
//     using float4 stores.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <int BLOCK>
__global__ void layer_norm_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int D,
                                   float eps) {
    constexpr int WARPS = BLOCK / 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;

    // ------------------------------------------------------------------
    // Pass 1: accumulate sum and sum_sq (float4 main loop + scalar tail).
    // ------------------------------------------------------------------
    float local_sum = 0.0f, local_sumsq = 0.0f;

    int D4 = D / 4;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int i = tid; i < D4; i += BLOCK) {
        float4 v = __ldg(&xr4[i]);
        local_sum   += v.x + v.y + v.z + v.w;
        local_sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Scalar tail for D not divisible by 4.
    for (int i = D4 * 4 + tid; i < D; i += BLOCK) {
        float v = xr[i];
        local_sum   += v;
        local_sumsq += v * v;
    }

    // Warp-level reduce via shuffle.
    for (int o = 16; o > 0; o >>= 1) {
        local_sum   += __shfl_down_sync(0xffffffff, local_sum,   o);
        local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, o);
    }

    // Lane 0 of each warp writes to shared memory.
    __shared__ float s_sum[WARPS], s_sumsq[WARPS];
    if (lane == 0) {
        s_sum[wid]   = local_sum;
        s_sumsq[wid] = local_sumsq;
    }
    __syncthreads();

    // Thread 0 does the final cross-warp reduce and broadcasts mean/rstd.
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float total_sum = 0.0f, total_sumsq = 0.0f;
        for (int i = 0; i < WARPS; ++i) {
            total_sum   += s_sum[i];
            total_sumsq += s_sumsq[i];
        }
        float inv_D = 1.0f / (float)D;
        float mean  = total_sum * inv_D;
        float var   = total_sumsq * inv_D - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // ------------------------------------------------------------------
    // Pass 2: write normalized output using float4 stores.
    // ------------------------------------------------------------------
    float4* yr4 = reinterpret_cast<float4*>(yr);
    for (int i = tid; i < D4; i += BLOCK) {
        float4 v = __ldg(&xr4[i]);
        float4 out;
        out.x = (v.x - mean) * rstd;
        out.y = (v.y - mean) * rstd;
        out.z = (v.z - mean) * rstd;
        out.w = (v.w - mean) * rstd;
        yr4[i] = out;
    }
    // Scalar tail.
    for (int i = D4 * 4 + tid; i < D; i += BLOCK) {
        yr[i] = (xr[i] - mean) * rstd;
    }
}

torch::Tensor run(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),  "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2,       "x must be 2-D [B, D]");

    int64_t B = x.size(0);
    int64_t D = x.size(1);

    auto y = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();

    layer_norm_kernel<256><<<(int)B, 256, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)D,
        (float)eps
    );
    AT_CUDA_CHECK(cudaGetLastError());

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "LayerNorm forward (no affine)");
}
