// Row-wise softmax over the last dimension: y = exp(x - max(x)) / sum(exp(x - max(x)))
//
// Contract: torch::Tensor run(torch::Tensor x)
//   x must be fp32, CUDA, contiguous, shape [B, D].
//   Returns y with the same shape.
//
// Strategy: one block per row (gridDim.x = B), blockDim.x = BLOCK = 256.
// Each thread processes D/BLOCK elements with the online softmax algorithm
// (single pass: maintains local_max and local_sum together), then two block-wide
// reductions (max, then sum) using __shfl_xor_sync within warps and __shared__
// across warps, then a second pass to write normalised outputs.
//
// References: cuda-raw-kernel-skill.md (Reduction skeleton, row-wise) and
//             llm-rs/softmax_forward_kernel5.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Warp reduce — XOR butterfly, all 32 lanes get the result.
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int BLOCK>
__global__ void softmax_kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                int D) {
    static_assert(BLOCK % 32 == 0, "BLOCK must be a multiple of warp size");
    constexpr int NUM_WARPS = BLOCK / 32;

    const int row    = blockIdx.x;
    const int lane   = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;

    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;

    // Shared memory: one slot per warp for cross-warp reductions.
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];

    // -----------------------------------------------------------------------
    // Pass 1 – online max+sum over each thread's strided slice of the row.
    // After this: local_sum = sum_{i in thread's slice} exp(x[i] - local_max)
    // -----------------------------------------------------------------------
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float v = xr[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, v);
        local_sum *= __expf(old_max - local_max);
        local_sum += __expf(v - local_max);
    }

    // -----------------------------------------------------------------------
    // Block-wide max reduction.
    // -----------------------------------------------------------------------
    // (a) Within-warp XOR butterfly – all lanes of each warp get warp max.
    float warp_max = warp_reduce_max(local_max);

    // (b) Lane 0 writes its warp max; others wait.
    if (lane == 0) smem_max[warpid] = warp_max;
    __syncthreads();

    // (c) Cross-warp reduce on warp 0 only; thread 0 stores block max.
    float block_max;
    if (warpid == 0) {
        float v = (lane < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) smem_max[0] = v;
    }
    __syncthreads();
    block_max = smem_max[0]; // broadcast to all threads via smem

    // -----------------------------------------------------------------------
    // Rescale each thread's partial sum from local_max to block_max.
    // -----------------------------------------------------------------------
    local_sum *= __expf(local_max - block_max);

    // -----------------------------------------------------------------------
    // Block-wide sum reduction.
    // -----------------------------------------------------------------------
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem_sum[warpid] = warp_sum;
    __syncthreads();

    float block_sum;
    if (warpid == 0) {
        float v = (lane < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem_sum[0] = v;
    }
    __syncthreads();
    block_sum = smem_sum[0];

    const float inv_sum = 1.0f / block_sum;

    // -----------------------------------------------------------------------
    // Pass 2 – write normalised outputs.  Recompute exp(x[i] - block_max)
    // rather than storing intermediate values (saves shared memory / registers).
    // -----------------------------------------------------------------------
    for (int i = threadIdx.x; i < D; i += BLOCK)
        yr[i] = __expf(xr[i] - block_max) * inv_sum;
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [B, D]");

    x = x.contiguous();
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);

    auto y = torch::empty_like(x);

    constexpr int BLOCK = 256;
    // One block per row.
    const int64_t grid = B;
    TORCH_CHECK(grid <= 65535LL * 65535, "B too large for grid");

    softmax_kernel<BLOCK><<<(int)grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)D);

    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Numerically-stable row-wise softmax (fp32, [B,D])");
}
