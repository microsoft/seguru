#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define BLK 256
#define KSZ 3

__global__ void conv2d_scale_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WGT,
    const float* __restrict__ BIAS,
    float* __restrict__ TMP,
    int Bsz, int Cin, int H, int Wi, int Cout, int Ho, int Wo)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)Bsz * Cout * Ho * Wo;
    if (idx >= total) return;

    const int wo = (int)(idx % Wo);
    size_t t = idx / Wo;
    const int ho = (int)(t % Ho);
    t /= Ho;
    const int co = (int)(t % Cout);
    const int bi = (int)(t / Cout);

    float acc = BIAS[co];
    const float* x_batch = X + ((size_t)bi * Cin * H * Wi);
    const float* w_chan = WGT + ((size_t)co * Cin * KSZ * KSZ);

    for (int ci = 0; ci < Cin; ++ci) {
        const float* x_ci = x_batch + (size_t)ci * H * Wi + (size_t)ho * Wi + wo;
        const float* w_ci = w_chan + ci * KSZ * KSZ;
        #pragma unroll
        for (int kh = 0; kh < KSZ; ++kh) {
            const float* x_row = x_ci + kh * Wi;
            const float* w_row = w_ci + kh * KSZ;
            acc += x_row[0] * w_row[0];
            acc += x_row[1] * w_row[1];
            acc += x_row[2] * w_row[2];
        }
    }

    TMP[idx] = acc * 2.0f;
}

__global__ void channel_min_kernel(
    const float* __restrict__ TMP,
    float* __restrict__ Y,
    int Bsz, int Cout, int Ho, int Wo)
{
    const size_t out_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)Bsz * Ho * Wo;
    if (out_idx >= total) return;

    const int wo = (int)(out_idx % Wo);
    size_t t = out_idx / Wo;
    const int ho = (int)(t % Ho);
    const int bi = (int)(t / Ho);

    float m = TMP[((size_t)bi * Cout * Ho + ho) * Wo + wo];
    for (int co = 1; co < Cout; ++co) {
        const float v = TMP[((size_t)(bi * Cout + co) * Ho + ho) * Wo + wo];
        m = fminf(m, v);
    }
    Y[out_idx] = m;
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
    TORCH_CHECK(x.dim() == 4 && W.dim() == 4 && b.dim() == 1, "bad shapes");
    TORCH_CHECK(W.size(2) == KSZ && W.size(3) == KSZ, "kernel_size must be 3");

    const int Bsz = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int H = (int)x.size(2);
    const int Wi = (int)x.size(3);
    const int Cout = (int)W.size(0);
    TORCH_CHECK(W.size(1) == Cin, "W.size(1) must equal x.size(1)");
    TORCH_CHECK(b.size(0) == Cout, "b.size(0) must equal W.size(0)");

    const int Ho = H - KSZ + 1;
    const int Wo = Wi - KSZ + 1;
    auto tmp = torch::empty({Bsz, Cout, Ho, Wo}, x.options());
    auto y = torch::empty({Bsz, 1, Ho, Wo}, x.options());

    const size_t conv_total = (size_t)Bsz * Cout * Ho * Wo;
    const size_t out_total = (size_t)Bsz * Ho * Wo;
    const dim3 block(BLK, 1, 1);
    const dim3 conv_grid((unsigned int)((conv_total + BLK - 1) / BLK), 1, 1);
    const dim3 min_grid((unsigned int)((out_total + BLK - 1) / BLK), 1, 1);

    conv2d_scale_kernel<<<conv_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), tmp.data_ptr<float>(),
        Bsz, Cin, H, Wi, Cout, Ho, Wo);
    AT_CUDA_CHECK(cudaGetLastError());

    channel_min_kernel<<<min_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tmp.data_ptr<float>(), y.data_ptr<float>(), Bsz, Cout, Ho, Wo);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Conv2d + scaling + channel min (float32, k=3, no pad/stride)");
}
