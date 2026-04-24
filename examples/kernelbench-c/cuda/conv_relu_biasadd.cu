#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// KernelBench Level 2 problem 1: Conv2d + ReLU + (extra) BiasAdd.
//
// PyTorch reference:
//     y  = F.conv2d(x, W, conv_bias)            // no padding, stride 1
//     y  = F.relu(y)
//     y  = y + extra_bias                       // extra_bias: [Cout, 1, 1]
//
// Note: the ReLU sits BETWEEN the two biases, so we cannot simply sum them
// into a single effective bias; we apply conv_bias first, then relu, then add
// extra_bias.  Both are broadcast per-output-channel.
//
// Shapes (from compare.py):
//     x       : [B=128, Cin=64, H=128, W=128]
//     W       : [Cout=128, Cin=64, Kh=3, Kw=3]
//     b       : [Cout]
//     bias2   : [Cout, 1, 1]  (treated as [Cout])
//     y       : [B, Cout, Ho=126, Wo=126]
//
// Strategy: direct convolution, one output per thread.
//   * 2-D output tile of TH=16 × TW=16 pixels per block.
//   * blockIdx.z indexes (batch, out_channel).
//   * Inner loop streams Cin·Kh·Kw = 576 FMAs per output from global memory;
//     Kh=Kw=3 is fully unrolled.  No shared memory — the simple pattern is
//     adequate for ~150 GFLOPs of work on A100 and keeps the code compact.

#define TH 16
#define TW 16

__global__ void conv_relu_biasadd_kernel(
    const float* __restrict__ X,       // [B, Cin, H, Wd]
    const float* __restrict__ WGT,     // [Cout, Cin, Kh, Kw]
    const float* __restrict__ B1,      // [Cout]       (conv bias)
    const float* __restrict__ B2,      // [Cout, 1, 1] (extra bias, stored as [Cout])
    float* __restrict__ Y,             // [B, Cout, Ho, Wo]
    int B, int Cin, int H, int Wd, int Cout,
    int Kh, int Kw, int Ho, int Wo)
{
    const int wo = blockIdx.x * TW + threadIdx.x;
    const int ho = blockIdx.y * TH + threadIdx.y;
    const int bc = blockIdx.z;                    // 0 .. B*Cout
    const int bi = bc / Cout;
    const int co = bc - bi * Cout;

    if (wo >= Wo || ho >= Ho) return;

    const int xbs = Cin * H * Wd;                 // stride of one batch in X
    const int wcs = Cin * Kh * Kw;                // stride of one out-channel in W

    float acc = 0.0f;
    const float* x_batch = X + bi * xbs;
    const float* w_chan  = WGT + co * wcs;

    for (int ci = 0; ci < Cin; ++ci) {
        const float* x_p = x_batch + ci * H * Wd + ho * Wd + wo;
        const float* w_p = w_chan  + ci * Kh * Kw;
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                acc += x_p[kh * Wd + kw] * w_p[kh * Kw + kw];
            }
        }
    }

    // Fused epilogue: conv_bias + ReLU + extra_bias.
    float v = acc + B1[co];
    v = v > 0.0f ? v : 0.0f;
    v += B2[co];

    Y[((bi * Cout + co) * Ho + ho) * Wo + wo] = v;
}

torch::Tensor run(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor bias2)
{
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda() && bias2.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && b.is_contiguous() &&
                    bias2.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32 &&
                    bias2.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(x.dim() == 4 && W.dim() == 4 && b.dim() == 1, "bad shapes");
    // bias2 may be [Cout, 1, 1] or [Cout]; reshape view to [Cout].
    auto bias2_flat = bias2.contiguous().view(-1);

    const int B   = (int)x.size(0);
    const int Cin = (int)x.size(1);
    const int H   = (int)x.size(2);
    const int Wd  = (int)x.size(3);
    const int Cout = (int)W.size(0);
    const int Kh   = (int)W.size(2);
    const int Kw   = (int)W.size(3);
    const int Ho = H - Kh + 1;
    const int Wo = Wd - Kw + 1;
    TORCH_CHECK(W.size(1) == Cin, "W.size(1) must equal Cin");
    TORCH_CHECK(b.size(0) == Cout, "b.size(0) must equal Cout");
    TORCH_CHECK(bias2_flat.size(0) == Cout, "bias2 must have Cout elements");
    TORCH_CHECK(Kh == 3 && Kw == 3, "this kernel is specialized for 3x3");

    auto y = torch::empty({B, Cout, Ho, Wo}, x.options());

    dim3 block(TW, TH, 1);
    dim3 grid((Wo + TW - 1) / TW, (Ho + TH - 1) / TH, B * Cout);

    conv_relu_biasadd_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        bias2_flat.data_ptr<float>(),
        y.data_ptr<float>(),
        B, Cin, H, Wd, Cout, Kh, Kw, Ho, Wo);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused Conv2d + ReLU + BiasAdd (float32)");
}
