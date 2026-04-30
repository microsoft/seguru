// Row-wise inclusive cumulative sum over last dim: [B, D] -> [B, D]
// One block per row, block-wide scan in shared memory (Kogge-Stone).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Block scan: BLOCK threads each own ITEMS elements, D = BLOCK*ITEMS.
// Assumes D is a multiple of BLOCK.
template <int BLOCK>
__global__ void cumsum_kernel(const float* __restrict__ x, float* __restrict__ y, int D) {
    extern __shared__ float smem[]; // size BLOCK
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xr = x + (int64_t)row * D;
    float*       yr = y + (int64_t)row * D;
    const int items = D / BLOCK;

    // Phase 1: thread-local inclusive prefix of `items` consecutive elements,
    // stored into y[row] in place for later patch-up.
    float run = 0.f;
    #pragma unroll 1
    for (int i = 0; i < items; i++) {
        run += xr[tid * items + i];
        yr[tid * items + i] = run;
    }
    // `run` now holds the thread's total.
    smem[tid] = run;
    __syncthreads();

    // Phase 2: block-wide Kogge-Stone inclusive scan over per-thread totals.
    for (int off = 1; off < BLOCK; off <<= 1) {
        float v = (tid >= off) ? smem[tid - off] : 0.f;
        __syncthreads();
        smem[tid] += v;
        __syncthreads();
    }
    // Exclusive offset for this thread = smem[tid] - run.
    float offset = smem[tid] - run;

    // Phase 3: add offset to this thread's `items` output values.
    for (int i = 0; i < items; i++) {
        yr[tid * items + i] += offset;
    }
}

torch::Tensor run(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 2);
    const int64_t B = x.size(0), D = x.size(1);
    constexpr int BLOCK = 256;
    TORCH_CHECK(D % BLOCK == 0, "D must be divisible by BLOCK");
    auto y = torch::empty_like(x);
    cumsum_kernel<BLOCK><<<(int)B, BLOCK, BLOCK * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), (int)D);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "cumsum over last dim"); }
