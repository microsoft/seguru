#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Fused: y[m] = Σ_n sigmoid( Σ_k x[m,k] * W[n,k] + b[n] )
//
// Shapes: x [M, K], W [N, K], b [N], out [M, 1].
// For Model 56 we see M=128, K=N=32768 — huge inner & hidden, tiny batch.
// The [M, N] intermediate (128 * 32768 * 4 = 16 MB) is never materialized:
// each thread accumulates partial sigmoid-sums in registers, block-reduces
// at the end, and writes one f32 per row to y.
//
// Layout:
//   grid  = (M,)                                 — one block per row
//   block = (BDIM,)                              — 256 threads
//   Each thread owns TN=4 output columns per "n-chunk" (nc step = BDIM*TN=1024).
//   For each n-chunk we keep 4 register accumulators and sweep the full K
//   dimension with a BK=256 shared-memory tile of x[m, :].
//   After K is done we apply bias + sigmoid and fold into a scalar `partial`.
//   After all n-chunks we block-reduce `partial` via tree-reduce in smem.
//
// Notes on perf:
//   * Compute dominates bandwidth (≈137 GFMA per row on A100 → ~70 ms/row
//     at fp32 peak). Block sizing keeps occupancy reasonable (smem small).
//   * W reads are stride-K across threads within the inner j-loop (NOT
//     warp-coalesced by n), but each thread streams contiguously along K,
//     so sector-per-thread throughput is still OK. A classical 2-D smem tile
//     of W would coalesce, but the fused reduction keeps the kernel simple.
//   * `--use_fast_math` turns __expf into a hardware approx; row-sum drift
//     over N=32768 accumulations is the reason compare.py sets atol=0.2.

#define BDIM 256
#define BK   256
#define TN   4
#define NC   (BDIM * TN)   // 1024 n's processed per outer iteration

__global__ void matmul_sigmoid_sum_kernel(
    const float* __restrict__ X,   // [M, K]
    const float* __restrict__ W,   // [N, K]  (nn.Linear layout)
    const float* __restrict__ B,   // [N]
    float* __restrict__ Y,         // [M, 1]
    int M, int N, int K)
{
    const int m   = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = X + (size_t)m * K;

    __shared__ float xs[BK];
    __shared__ float red[BDIM];

    float partial = 0.0f;

    for (int nc = 0; nc < N; nc += NC) {
        float acc[TN];
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[j] = 0.0f;

        for (int k_tile = 0; k_tile < K; k_tile += BK) {
            // Cooperative load: BK == BDIM so each thread loads one float.
            xs[tid] = x_row[k_tile + tid];
            __syncthreads();

            #pragma unroll 8
            for (int k = 0; k < BK; ++k) {
                const float xk = xs[k];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int n_idx = nc + j * BDIM + tid;
                    acc[j] += xk * W[(size_t)n_idx * K + k_tile + k];
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n_idx = nc + j * BDIM + tid;
            const float v = acc[j] + B[n_idx];
            partial += 1.0f / (1.0f + __expf(-v));
        }
    }

    // Block tree-reduce `partial` → thread 0 writes y[m].
    red[tid] = partial;
    __syncthreads();
    #pragma unroll
    for (int s = BDIM / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) Y[m] = red[0];
}

torch::Tensor run(torch::Tensor x, torch::Tensor W, torch::Tensor b)
{
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && b.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2 && b.dim() == 1, "bad shapes");

    const int M = (int)x.size(0);
    const int K = (int)x.size(1);
    const int N = (int)W.size(0);
    TORCH_CHECK(W.size(1) == K, "W.size(1) must equal x.size(1)");
    TORCH_CHECK(b.size(0) == N, "b.size(0) must equal W.size(0)");
    TORCH_CHECK((K % BK) == 0, "K must be a multiple of 256");
    TORCH_CHECK((N % NC) == 0, "N must be a multiple of 1024");

    auto y = torch::empty({M, 1}, x.options());

    dim3 grid(M, 1, 1);
    dim3 block(BDIM, 1, 1);

    matmul_sigmoid_sum_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N, K);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused matmul + sigmoid + row-sum (float32)");
}
