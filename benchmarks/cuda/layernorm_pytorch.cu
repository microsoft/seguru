// Standalone distillation of PyTorch's vectorized_layer_norm_kernel
// (aten/src/ATen/native/cuda/layer_norm_kernel.cu).
// One block per row; warp reduction → block reduction via shared memory.
// Vectorized loads (float4) when N % 4 == 0. For fp32 only.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

constexpr int VEC = 4;
constexpr int WARP_SIZE = 32;

struct WelfordLN {
    float mean;
    float sigma2;
    float count;
};

__device__ WelfordLN welfordCombine(WelfordLN a, WelfordLN b) {
    float count = a.count + b.count;
    if (count <= 0.f) return {0.f, 0.f, 0.f};
    float coef = 1.f / count;
    float nA = a.count * coef;
    float nB = b.count * coef;
    float delta = b.mean - a.mean;
    return {
        nA * a.mean + nB * b.mean,
        a.sigma2 + b.sigma2 + delta * delta * a.count * nB,
        count
    };
}

__device__ WelfordLN welfordOnline(float val, WelfordLN curr) {
    float new_count = curr.count + 1.f;
    float delta = val - curr.mean;
    float new_mean = curr.mean + delta * (1.f / new_count);
    return {new_mean, curr.sigma2 + delta * (val - new_mean), new_count};
}

__global__ void layernorm_pytorch_vec(
    int N,
    float eps,
    const float* __restrict__ X,
    const float* gamma,
    const float* beta,
    float* Y)
{
    extern __shared__ float s_data[];
    int i1 = blockIdx.x;
    const float* block_row = X + i1 * N;

    const int n_vec = N / VEC;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;

    const float4* X_vec = reinterpret_cast<const float4*>(block_row);

    WelfordLN wd = {0.f, 0.f, 0.f};
    for (int i = thrx; i < n_vec; i += numx) {
        float4 d = X_vec[i];
        wd = welfordOnline(d.x, wd);
        wd = welfordOnline(d.y, wd);
        wd = welfordOnline(d.z, wd);
        wd = welfordOnline(d.w, wd);
    }
    // intra-warp reduction
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        WelfordLN b = {
            __shfl_down_sync(0xffffffff, wd.mean, off),
            __shfl_down_sync(0xffffffff, wd.sigma2, off),
            __shfl_down_sync(0xffffffff, wd.count, off)
        };
        wd = welfordCombine(wd, b);
    }
    // inter-warp reduction via shared memory
    if (blockDim.y > 1) {
        float* meansigma = s_data;
        float* cbuf = s_data + blockDim.y * 2;
        for (int off = blockDim.y / 2; off > 0; off /= 2) {
            if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2*off) {
                int wy = threadIdx.y - off;
                meansigma[2*wy] = wd.mean;
                meansigma[2*wy+1] = wd.sigma2;
                cbuf[wy] = wd.count;
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y < off) {
                WelfordLN b = {meansigma[2*threadIdx.y], meansigma[2*threadIdx.y+1], cbuf[threadIdx.y]};
                wd = welfordCombine(wd, b);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            meansigma[0] = wd.mean;
            meansigma[1] = wd.sigma2 / float(N);
        }
        __syncthreads();
        wd.mean = meansigma[0];
        wd.sigma2 = meansigma[1];
    } else {
        wd.mean = __shfl_sync(0xffffffff, wd.mean, 0);
        wd.sigma2 = __shfl_sync(0xffffffff, wd.sigma2, 0) / float(N);
    }

    float rstd = rsqrtf(wd.sigma2 + eps);

    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec  = reinterpret_cast<const float4*>(beta);
    float4* Y_vec = reinterpret_cast<float4*>(Y + i1 * N);

    for (int i = thrx; i < n_vec; i += numx) {
        float4 d = X_vec[i];
        float4 g = gamma_vec[i];
        float4 b = beta_vec[i];
        float4 o;
        o.x = g.x * (rstd * (d.x - wd.mean)) + b.x;
        o.y = g.y * (rstd * (d.y - wd.mean)) + b.y;
        o.z = g.z * (rstd * (d.z - wd.mean)) + b.z;
        o.w = g.w * (rstd * (d.w - wd.mean)) + b.w;
        Y_vec[i] = o;
    }
}

int main() {
    const int M = 8192;
    const int N = 1024;
    const size_t bytes = size_t(M) * N * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_g = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(bytes);
    for (size_t i = 0; i < size_t(M)*N; i++) h_x[i] = (float)(rand() % 1000) / 1000.f - 0.5f;
    for (int i = 0; i < N; i++) { h_g[i] = 1.f; h_b[i] = 0.f; }

    float *d_x, *d_g, *d_b, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_g, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Block: 32 × 8 (256 threads, 8 warps).
    dim3 block(32, 8);
    dim3 grid(M);
    size_t smem = block.y * 3 * sizeof(float);

    // Warmup
    layernorm_pytorch_vec<<<grid, block, smem>>>(N, 1e-5f, d_x, d_g, d_b, d_y);
    cudaDeviceSynchronize();

    int iters = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        layernorm_pytorch_vec<<<grid, block, smem>>>(N, 1e-5f, d_x, d_g, d_b, d_y);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    printf("layernorm pytorch_vec CUDA:  %8.2f us/iter  (M=%d N=%d)\n", us, M, N);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_g); cudaFree(d_b);
    free(h_x); free(h_g); free(h_b); free(h_y);
    return 0;
}
