#include <torch/extension.h>
#include <cuda_runtime.h>

// Pass 1: for every spatial position (b, h, w) compute
//     inv_rms[b, h, w] = rsqrt( mean_c( x[b, c, h, w]^2 ) + eps )
//
// Layout observation: for fixed (b, h, w) the C elements to reduce are at
// stride HW. For fixed c, consecutive w elements are contiguous. So if each
// thread owns one pixel and loops over c, adjacent threads in a warp hit
// adjacent w's at the same c => fully coalesced global loads.
//
// We go one step further: each thread owns FOUR consecutive w-pixels and
// issues float4 loads, keeping four independent accumulators. Requires
// HW % 4 == 0 (trivially true for 512x512).
__global__ void rms_reduce_kernel(const float* __restrict__ x,
                                  float* __restrict__ inv_rms,
                                  int C, int HW, float eps, float inv_C) {
    int64_t quad = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t B_HW_q = (int64_t)gridDim.x * blockDim.x; // may overshoot; guarded by caller via exact grid
    (void)B_HW_q;

    // Decode quad -> (b, hw_base)
    int64_t base = quad * 4;                 // element offset within the (B, HW) plane
    int b  = (int)(base / HW);
    int hw = (int)(base - (int64_t)b * HW);

    const float* xb = x + ((int64_t)b * C * HW) + hw;

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll 8
    for (int c = 0; c < C; ++c) {
        float4 v = *reinterpret_cast<const float4*>(xb + (int64_t)c * HW);
        s0 += v.x * v.x;
        s1 += v.y * v.y;
        s2 += v.z * v.z;
        s3 += v.w * v.w;
    }

    float4 out;
    out.x = rsqrtf(s0 * inv_C + eps);
    out.y = rsqrtf(s1 * inv_C + eps);
    out.z = rsqrtf(s2 * inv_C + eps);
    out.w = rsqrtf(s3 * inv_C + eps);
    *reinterpret_cast<float4*>(inv_rms + (int64_t)b * HW + hw) = out;
}

// Pass 2: y = x * broadcast(inv_rms). Vectorized float4 across the contiguous
// (w) axis; all four lanes share the same (b, h) but different w, so we also
// load 4 adjacent inv_rms values with a single float4.
__global__ void rms_apply_kernel(const float* __restrict__ x,
                                 const float* __restrict__ inv_rms,
                                 float* __restrict__ y,
                                 int C, int HW) {
    int64_t quad = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base = quad * 4; // element offset into full (B, C, HW) tensor

    // Decode: base = b*(C*HW) + c*HW + w  ; we need (b, w)
    int64_t CHW = (int64_t)C * HW;
    int b  = (int)(base / CHW);
    int64_t rem = base - (int64_t)b * CHW;
    int w  = (int)(rem % HW); // position inside the HW plane
    // c not needed

    float4 xv = *reinterpret_cast<const float4*>(x + base);
    float4 iv = *reinterpret_cast<const float4*>(inv_rms + (int64_t)b * HW + w);
    float4 yv;
    yv.x = xv.x * iv.x;
    yv.y = xv.y * iv.y;
    yv.z = xv.z * iv.z;
    yv.w = xv.w * iv.w;
    *reinterpret_cast<float4*>(y + base) = yv;
}

torch::Tensor run(torch::Tensor x, float eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 4);

    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t HW = H * W;
    TORCH_CHECK((HW % 4) == 0, "H*W must be divisible by 4 for float4 path");

    auto y = torch::empty_like(x);
    auto inv_rms = torch::empty({B, 1, H, W}, x.options());

    const float inv_C = 1.0f / (float)C;

    // Pass 1 launch: one thread per 4 pixels.
    {
        int64_t total_quads = B * HW / 4;
        int block = 256;
        int64_t grid64 = (total_quads + block - 1) / block;
        TORCH_CHECK(grid64 < (1LL << 31), "grid too large");
        int grid = (int)grid64;
        rms_reduce_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            (int)C, (int)HW, eps, inv_C);
    }

    // Pass 2 launch: one thread per 4 elements of the full tensor.
    {
        int64_t total_quads = B * C * HW / 4;
        int block = 256;
        int64_t grid64 = (total_quads + block - 1) / block;
        TORCH_CHECK(grid64 < (1LL << 31), "grid too large");
        int grid = (int)grid64;
        rms_apply_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            y.data_ptr<float>(),
            (int)C, (int)HW);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "RMSNorm forward");
}
