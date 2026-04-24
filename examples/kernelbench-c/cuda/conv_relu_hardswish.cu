#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Fused Conv2d + ReLU + HardSwish (KernelBench Level-2 problem 57).
//
// PyTorch reference:
//     y = conv2d(x, W, b)         // [B, Cout, Ho, Wo], Ho=H-2, Wo=W-2, k=3
//     y = relu(y)
//     y = y * clamp((y + 3) / 6, 0, 1)
//
// Strategy: direct convolution, one output element per thread. For the pilot
// shape (B=128, Cin=8, H=W=128, Cout=64, k=3) each thread does 8*3*3 = 72
// FMADDs. We flatten the output to rows=(B*Cout*Ho), cols=Wo and launch a 2D
// grid so each thread writes exactly one y[b, co, ho, wo]. ReLU and HardSwish
// are fused into the final store — no intermediate tensor is allocated.
//
// Block 16x16 (256 threads); bounds-guarded because Wo=126 is not 16-aligned.

#define BLK_X 16
#define BLK_Y 16
#define KSZ   3

__global__ void conv_relu_hardswish_kernel(
    const float* __restrict__ X,   // [B, Cin, H,  W ]
    const float* __restrict__ WGT, // [Cout, Cin, 3, 3]
    const float* __restrict__ B,   // [Cout]
    float* __restrict__ Y,         // [B, Cout, Ho, Wo]
    int Bsz, int Cin, int H, int Wi,
    int Cout, int Ho, int Wo)
{
    const int wo  = blockIdx.x * BLK_X + threadIdx.x;
    const int row = blockIdx.y * BLK_Y + threadIdx.y;   // = bco*Ho + ho
    const int total_rows = Bsz * Cout * Ho;
    if (wo >= Wo || row >= total_rows) return;

    const int bco = row / Ho;
    const int ho  = row - bco * Ho;
    const int bi  = bco / Cout;
    const int co  = bco - bi * Cout;

    float acc = B[co];

    // Inner loop: Σ_{ci, kh, kw} W[co, ci, kh, kw] * x[bi, ci, ho+kh, wo+kw]
    const float* x_bi = X + ((bi * Cin) * H) * Wi;
    const float* w_co = WGT + (co * Cin) * (KSZ * KSZ);

    #pragma unroll 1
    for (int ci = 0; ci < Cin; ++ci) {
        const float* xc = x_bi + ci * H * Wi;
        const float* wc = w_co + ci * (KSZ * KSZ);
        #pragma unroll
        for (int kh = 0; kh < KSZ; ++kh) {
            const float* xr = xc + (ho + kh) * Wi + wo;
            const float* wr = wc + kh * KSZ;
            // kw = 0,1,2 — unrolled
            acc += xr[0] * wr[0];
            acc += xr[1] * wr[1];
            acc += xr[2] * wr[2];
        }
    }

    // Fused epilogue: ReLU then HardSwish.
    float r = acc > 0.f ? acc : 0.f;
    float hs = (r + 3.f) * (1.f / 6.f);
    hs = fminf(fmaxf(hs, 0.f), 1.f);
    Y[((bi * Cout + co) * Ho + ho) * Wo + wo] = r * hs;
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

    const int Bsz  = (int)x.size(0);
    const int Cin  = (int)x.size(1);
    const int H    = (int)x.size(2);
    const int Wi   = (int)x.size(3);
    const int Cout = (int)W.size(0);
    TORCH_CHECK(W.size(1) == Cin, "W.size(1) must equal x.size(1)");
    TORCH_CHECK(b.size(0) == Cout, "b.size(0) must equal W.size(0)");

    const int Ho = H - (KSZ - 1);
    const int Wo = Wi - (KSZ - 1);

    auto y = torch::empty({Bsz, Cout, Ho, Wo}, x.options());

    dim3 block(BLK_X, BLK_Y, 1);
    dim3 grid((Wo + BLK_X - 1) / BLK_X,
              (Bsz * Cout * Ho + BLK_Y - 1) / BLK_Y,
              1);

    conv_relu_hardswish_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        Bsz, Cin, H, Wi, Cout, Ho, Wo);
    AT_CUDA_CHECK(cudaGetLastError());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused Conv2d + ReLU + HardSwish (float32, k=3, no pad/stride)");
}
