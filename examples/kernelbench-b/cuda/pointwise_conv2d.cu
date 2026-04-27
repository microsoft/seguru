#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define BLK 256

__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WGT,
    float* __restrict__ Y,
    int Bsz, int Cin, int H, int Wi, int Cout)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)Bsz * Cout * H * Wi;
    if (idx >= total) return;

    const int wi = (int)(idx % Wi);
    size_t t = idx / Wi;
    const int h = (int)(t % H);
    t /= H;
    const int co = (int)(t % Cout);
    const int b = (int)(t / Cout);

    float acc = 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        const size_t x_idx = (((size_t)b * Cin + ci) * H + h) * Wi + wi;
        const size_t w_idx = (size_t)co * Cin + ci;
        acc += X[x_idx] * WGT[w_idx];
    }
    Y[idx] = acc;
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
    const size_t total = (size_t)Bsz * Cout * H * Wi;
    const dim3 block(BLK, 1, 1);
    const dim3 grid((unsigned int)((total + BLK - 1) / BLK), 1, 1);

    pointwise_conv2d_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        Bsz, Cin, H, Wi, Cout);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Pointwise 1x1 Conv2d, stride=1, padding=0");
}
