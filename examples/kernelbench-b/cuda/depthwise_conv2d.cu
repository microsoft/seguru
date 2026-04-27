#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define BLK 256

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WGT,
    float* __restrict__ Y,
    int Bsz, int C, int H, int Wi, int Kh, int Kw, int Ho, int Wo)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)Bsz * C * Ho * Wo;
    if (idx >= total) return;

    const int wo = (int)(idx % Wo);
    size_t t = idx / Wo;
    const int ho = (int)(t % Ho);
    t /= Ho;
    const int c = (int)(t % C);
    const int b = (int)(t / C);

    const float* x_bc = X + ((size_t)b * C + c) * H * Wi;
    const float* w_c = WGT + (size_t)c * Kh * Kw;
    float acc = 0.0f;
    for (int kh = 0; kh < Kh; ++kh) {
        const float* x_row = x_bc + (size_t)(ho + kh) * Wi + wo;
        const float* w_row = w_c + kh * Kw;
        for (int kw = 0; kw < Kw; ++kw) {
            acc += x_row[kw] * w_row[kw];
        }
    }
    Y[idx] = acc;
}

torch::Tensor run(torch::Tensor x, torch::Tensor w)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4, "depthwise_conv2d expects x [B,C,H,W], w [C,1,Kh,Kw]");
    const int Bsz = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int Wi = (int)x.size(3);
    const int Kh = (int)w.size(2);
    const int Kw = (int)w.size(3);
    TORCH_CHECK(w.size(0) == C && w.size(1) == 1, "w must have shape [C,1,Kh,Kw]");
    TORCH_CHECK(H >= Kh && Wi >= Kw, "kernel larger than input");

    const int Ho = H - Kh + 1;
    const int Wo = Wi - Kw + 1;
    auto y = torch::empty({Bsz, C, Ho, Wo}, x.options());
    const size_t total = (size_t)Bsz * C * Ho * Wo;
    const dim3 block(BLK, 1, 1);
    const dim3 grid((unsigned int)((total + BLK - 1) / BLK), 1, 1);

    depthwise_conv2d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        Bsz, C, H, Wi, Kh, Kw, Ho, Wo);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Depthwise Conv2d, groups=C, stride=1, padding=0");
}
