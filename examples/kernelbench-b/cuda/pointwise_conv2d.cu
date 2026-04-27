#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void pointwise_conv2d_tiled_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WGT,
    float* __restrict__ Y,
    int Bsz, int Cin, int H, int Wi, int Cout)
{
    __shared__ float Xs[TILE_M][TILE_K + 1];
    __shared__ float Ws[TILE_K][TILE_N + 1];

    const int tm = (int)threadIdx.x;
    const int tn = (int)threadIdx.y;
    const int hw = (int)blockIdx.x * TILE_M + tm;
    const int co = (int)blockIdx.y * TILE_N + tn;
    const int b = (int)blockIdx.z;
    const int M = H * Wi;
    float acc = 0.0f;

    for (int k0 = 0; k0 < Cin; k0 += TILE_K) {
        const int x_ci = k0 + tn;
        const int w_ci = k0 + tm;

        Xs[tm][tn] = (hw < M && x_ci < Cin)
            ? X[((size_t)b * Cin + x_ci) * M + hw]
            : 0.0f;
        Ws[tm][tn] = (w_ci < Cin && co < Cout)
            ? WGT[(size_t)co * Cin + w_ci]
            : 0.0f;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += Xs[tm][kk] * Ws[kk][tn];
        }
        __syncthreads();
    }

    if (hw < M && co < Cout) {
        Y[((size_t)b * Cout + co) * M + hw] = acc;
    }
}

torch::Tensor run(torch::Tensor x, torch::Tensor w)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "pointwise_conv2d expects x [B,Cin,H,W], w [Cout,Cin,1,1]");
    TORCH_CHECK(w.size(2) == 1 && w.size(3) == 1, "pointwise_conv2d requires 1x1 weights");

    const int Bsz = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int H = (int)x.size(2);
    const int Wi = (int)x.size(3);
    const int Cout = (int)w.size(0);
    TORCH_CHECK(w.size(1) == Cin, "w.size(1) must equal x.size(1)");

    auto y = torch::empty({Bsz, Cout, H, Wi}, x.options());
    const int M = H * Wi;
    const dim3 block(TILE_M, TILE_N, 1);
    const dim3 grid(
        (unsigned int)((M + TILE_M - 1) / TILE_M),
        (unsigned int)((Cout + TILE_N - 1) / TILE_N),
        (unsigned int)Bsz);

    pointwise_conv2d_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        Bsz, Cin, H, Wi, Cout);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Pointwise 1x1 Conv2d, stride=1, padding=0");
}
