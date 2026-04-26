#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Classic tiled SGEMM with fused epilogue:
//     z[m,n] = sum_k x[m,k] * W[n,k] + b[n]
//     y[m,n] = z[m,n] * sigmoid(z[m,n]) * 2.0
//
// PyTorch nn.Linear convention: x: [M,K], W: [N,K], b: [N]  =>  y = x @ W.T + b
//
// Tile config (tuned for A100 sm_80, FP32):
//   BM=128, BN=128, BK=8, TM=8, TN=8, block = 16x16 = 256 threads
//   Each thread computes an 8x8 output sub-tile (64 outputs).
//   Shared memory per block: (BM+BN)*BK*4 = 256*8*4 = 8192 B (trivial).
//
// Loads are vectorized via float4 from gmem -> smem; smem is stored in
// [k][m] / [k][n] layout so the inner product loads 8 contiguous lanes
// per thread via scalar loads that the compiler lifts into registers.

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matmul_swish_scaling_kernel(
    const float* __restrict__ X,  // [M, K]
    const float* __restrict__ W,  // [N, K]
    const float* __restrict__ B,  // [N]
    float* __restrict__ Y,        // [M, N]
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    const int tid        = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
    const int thread_row = threadIdx.y;                              // 0..15
    const int thread_col = threadIdx.x;                              // 0..15

    __shared__ float As[BK][BM];   // transposed: As[k][m]
    __shared__ float Bs[BK][BN];   // Bs[k][n]

    // Each thread loads exactly one float4 for A and one float4 for B per outer step.
    //   A tile: BM rows * BK cols = 128*8 = 1024 floats = 256 float4s.
    //   tid -> (a_row = tid/2, a_col = (tid%2)*4)
    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 2;
    const int b_row = tid >> 1;   // same layout; reused for W rows (n dim)
    const int b_col = (tid & 1) << 2;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
    }

    float a_reg[TM];
    float b_reg[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile: X[bm + a_row, k0 + a_col .. +3] -> As[a_col+i][a_row]
        {
            const float4 va = *reinterpret_cast<const float4*>(
                &X[(bm + a_row) * K + (k0 + a_col)]);
            As[a_col + 0][a_row] = va.x;
            As[a_col + 1][a_row] = va.y;
            As[a_col + 2][a_row] = va.z;
            As[a_col + 3][a_row] = va.w;
        }
        // ---- Load B tile from W: W[bn + b_row, k0 + b_col .. +3] -> Bs[b_col+i][b_row]
        {
            const float4 vb = *reinterpret_cast<const float4*>(
                &W[(bn + b_row) * K + (k0 + b_col)]);
            Bs[b_col + 0][b_row] = vb.x;
            Bs[b_col + 1][b_row] = vb.y;
            Bs[b_col + 2][b_row] = vb.z;
            Bs[b_col + 3][b_row] = vb.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load TM=8 A lanes as 2 float4s from contiguous As[k][thread_row*TM..].
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k][thread_row * TM + 0]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[k][thread_row * TM + 4]);
            a_reg[0] = a0.x; a_reg[1] = a0.y; a_reg[2] = a0.z; a_reg[3] = a0.w;
            a_reg[4] = a1.x; a_reg[5] = a1.y; a_reg[6] = a1.z; a_reg[7] = a1.w;

            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][thread_col * TN + 0]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][thread_col * TN + 4]);
            b_reg[0] = b0.x; b_reg[1] = b0.y; b_reg[2] = b0.z; b_reg[3] = b0.w;
            b_reg[4] = b1.x; b_reg[5] = b1.y; b_reg[6] = b1.z; b_reg[7] = b1.w;

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: fused bias + swish + scaling, then store.
    float bias_reg[TN];
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        bias_reg[j] = B[bn + thread_col * TN + j];
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = bm + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = bn + thread_col * TN + j;
            float z = acc[i][j] + bias_reg[j];
            float y = z / (1.0f + expf(-z));
            y *= 2.0f;
            Y[row * N + col] = y;
        }
    }
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
    TORCH_CHECK((M % BM) == 0 && (N % BN) == 0 && (K % BK) == 0,
                "M, N, K must be multiples of 128, 128, 8 respectively");

    auto y = torch::empty({M, N}, x.options());

    dim3 block(16, 16, 1);
    dim3 grid(N / BN, M / BM, 1);

    matmul_swish_scaling_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N, K);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused Matmul + Swish + Scaling (float32)");
}
