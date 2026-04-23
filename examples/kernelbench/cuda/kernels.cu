#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "kernels.h"

// ============================================================
// Timing helpers
// ============================================================

static int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static float median(float* arr, int n) {
    qsort(arr, n, sizeof(float), cmp_float);
    if (n % 2 == 1) return arr[n / 2];
    return 0.5f * (arr[n / 2 - 1] + arr[n / 2]);
}

// ============================================================
// Elementwise kernels
// ============================================================

__global__ void k_relu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : 0.f; }
}

__global__ void k_leaky_relu(const float* in, float* out, int n, float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : alpha * x; }
}

__global__ void k_sigmoid(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = 1.f / (1.f + expf(-x)); }
}

__global__ void k_tanh(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { out[tid] = tanhf(in[tid]); }
}

__global__ void k_swish(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x / (1.f + expf(-x)); }
}

__global__ void k_selu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        const float alpha = 1.6732632423543772f;
        const float scale = 1.0507009873554805f;
        float pos = x > 0.f ? x : 0.f;
        float neg = x < 0.f ? alpha * (expf(x) - 1.f) : 0.f;
        out[tid] = scale * (pos + neg);
    }
}

__global__ void k_hard_sigmoid(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float val = (in[tid] + 3.f) / 6.f;
        out[tid] = val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
    }
}

__global__ void k_softplus(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { out[tid] = logf(1.f + expf(in[tid])); }
}

__global__ void k_softsign(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x / (1.f + fabsf(x)); }
}

__global__ void k_elu(const float* in, float* out, int n, float alpha) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { float x = in[tid]; out[tid] = x > 0.f ? x : alpha * (expf(x) - 1.f); }
}

__global__ void k_hard_tanh(const float* in, float* out, int n, float min_val, float max_val) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = x < min_val ? min_val : (x > max_val ? max_val : x);
    }
}

// ============================================================
// GELU kernels
// ============================================================

__global__ void k_gelu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        out[tid] = 0.5f * x * (1.f + tanhf(inner));
    }
}

__global__ void k_mingpt_gelu(const float* in, float* out, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        out[tid] = 0.5f * x * (1.f + tanhf(inner));
    }
}

// ============================================================
// Elementwise benchmark macro (1 input, no extra params)
// ============================================================

#define BENCH_ELEM_1IN(name, kernel)                                           \
extern "C" float bench_##name(const float* h_in, float* h_out, int n,         \
                              int grid, int block, int warmup, int iters) {    \
    float *d_in, *d_out;                                                       \
    size_t sz = (size_t)n * sizeof(float);                                     \
    cudaMalloc(&d_in, sz);                                                     \
    cudaMalloc(&d_out, sz);                                                    \
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);                       \
    for (int i = 0; i < warmup; i++) kernel<<<grid, block>>>(d_in, d_out, n); \
    cudaDeviceSynchronize();                                                   \
    float* times = (float*)malloc(iters * sizeof(float));                      \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start); cudaEventCreate(&stop);                           \
    for (int i = 0; i < iters; i++) {                                          \
        cudaEventRecord(start);                                                \
        kernel<<<grid, block>>>(d_in, d_out, n);                               \
        cudaEventRecord(stop);                                                 \
        cudaEventSynchronize(stop);                                            \
        float ms; cudaEventElapsedTime(&ms, start, stop);                      \
        times[i] = ms * 1000.f;                                               \
    }                                                                          \
    float med = median(times, iters);                                          \
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);                     \
    cudaEventDestroy(start); cudaEventDestroy(stop);                           \
    free(times); cudaFree(d_in); cudaFree(d_out);                             \
    return med;                                                                \
}

BENCH_ELEM_1IN(relu_forward, k_relu)
BENCH_ELEM_1IN(sigmoid_forward, k_sigmoid)
BENCH_ELEM_1IN(tanh_forward, k_tanh)
BENCH_ELEM_1IN(swish_forward, k_swish)
BENCH_ELEM_1IN(selu_forward, k_selu)
BENCH_ELEM_1IN(hard_sigmoid_forward, k_hard_sigmoid)
BENCH_ELEM_1IN(softplus_forward, k_softplus)
BENCH_ELEM_1IN(softsign_forward, k_softsign)
BENCH_ELEM_1IN(gelu_forward, k_gelu)
BENCH_ELEM_1IN(mingpt_new_gelu_forward, k_mingpt_gelu)

// Leaky ReLU (extra param: alpha)
extern "C" float bench_leaky_relu_forward(const float* h_in, float* h_out, int n,
                                          float alpha, int grid, int block,
                                          int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_leaky_relu<<<grid, block>>>(d_in, d_out, n, alpha);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_leaky_relu<<<grid, block>>>(d_in, d_out, n, alpha);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// ELU (extra param: alpha)
extern "C" float bench_elu_forward(const float* h_in, float* h_out, int n,
                                   float alpha, int grid, int block,
                                   int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_elu<<<grid, block>>>(d_in, d_out, n, alpha);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_elu<<<grid, block>>>(d_in, d_out, n, alpha);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// HardTanh (extra params: min_val, max_val)
extern "C" float bench_hard_tanh_forward(const float* h_in, float* h_out, int n,
                                         float min_val, float max_val,
                                         int grid, int block,
                                         int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_hard_tanh<<<grid, block>>>(d_in, d_out, n, min_val, max_val);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_hard_tanh<<<grid, block>>>(d_in, d_out, n, min_val, max_val);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// ============================================================
// Matmul 2D kernels
// ============================================================

// C(M×N) = A(M×K) * B(K×N)
__global__ void k_matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[row * k + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

// C = Aᵀ * B, A stored as K×M
__global__ void k_matmul_ta(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[i * m + row] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

// C = A * Bᵀ, B stored as N×K
__global__ void k_matmul_tb(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[row * k + i] * b[col * k + i];
        c[row * n + col] = sum;
    }
}

// C = Aᵀ * Bᵀ, A stored as K×M, B stored as N×K
__global__ void k_matmul_tab(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.f;
        for (int i = 0; i < k; i++) sum += a[i * m + row] * b[col * k + i];
        c[row * n + col] = sum;
    }
}

// Matmul 2D benchmark macro
#define BENCH_MATMUL2D(name, kernel, a_rows, a_cols)                                                     \
extern "C" float bench_##name(const float* h_a, const float* h_b, float* h_c,                           \
                              int m, int n, int k,                                                       \
                              int grid_x, int grid_y, int block_x, int block_y,                          \
                              int warmup, int iters) {                                                   \
    float *d_a, *d_b, *d_c;                                                                              \
    size_t sz_a = (size_t)(a_rows) * (a_cols) * sizeof(float);                                           \
    size_t sz_b = (size_t)(a_rows == m ? k : (a_cols == k ? k : n)) *                                    \
                  (a_rows == m ? n : (a_cols == k ? n : k)) * sizeof(float);                             \
    size_t sz_c = (size_t)m * n * sizeof(float);                                                         \
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);                             \
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);                                                 \
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);                                                 \
    dim3 grid(grid_x, grid_y), blk(block_x, block_y);                                                   \
    for (int i = 0; i < warmup; i++) kernel<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);                     \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);                                                  \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);                                                 \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);                                           \
    return med;                                                                                          \
}

// For matmul variants we compute buffer sizes directly in the wrapper
// to avoid complex macro expressions for B size.

extern "C" float bench_matmul_forward(const float* h_a, const float* h_b, float* h_c,
                                      int m, int n, int k,
                                      int grid_x, int grid_y, int block_x, int block_y,
                                      int warmup, int iters) {
    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)m * k * sizeof(float);
    size_t sz_b = (size_t)k * n * sizeof(float);
    size_t sz_c = (size_t)m * n * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    dim3 grid(grid_x, grid_y), blk(block_x, block_y);
    for (int i = 0; i < warmup; i++) k_matmul<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_matmul<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return med;
}

extern "C" float bench_matmul_transposed_a(const float* h_a, const float* h_b, float* h_c,
                                           int m, int n, int k,
                                           int grid_x, int grid_y, int block_x, int block_y,
                                           int warmup, int iters) {
    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)k * m * sizeof(float); // A stored as K×M
    size_t sz_b = (size_t)k * n * sizeof(float);
    size_t sz_c = (size_t)m * n * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    dim3 grid(grid_x, grid_y), blk(block_x, block_y);
    for (int i = 0; i < warmup; i++) k_matmul_ta<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_matmul_ta<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return med;
}

extern "C" float bench_matmul_transposed_b(const float* h_a, const float* h_b, float* h_c,
                                           int m, int n, int k,
                                           int grid_x, int grid_y, int block_x, int block_y,
                                           int warmup, int iters) {
    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)m * k * sizeof(float);
    size_t sz_b = (size_t)n * k * sizeof(float); // B stored as N×K
    size_t sz_c = (size_t)m * n * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    dim3 grid(grid_x, grid_y), blk(block_x, block_y);
    for (int i = 0; i < warmup; i++) k_matmul_tb<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_matmul_tb<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return med;
}

extern "C" float bench_matmul_transposed_both(const float* h_a, const float* h_b, float* h_c,
                                              int m, int n, int k,
                                              int grid_x, int grid_y, int block_x, int block_y,
                                              int warmup, int iters) {
    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)k * m * sizeof(float); // A stored as K×M
    size_t sz_b = (size_t)n * k * sizeof(float); // B stored as N×K
    size_t sz_c = (size_t)m * n * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    dim3 grid(grid_x, grid_y), blk(block_x, block_y);
    for (int i = 0; i < warmup; i++) k_matmul_tab<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_matmul_tab<<<grid, blk>>>(d_a, d_b, d_c, m, n, k);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return med;
}

// ============================================================
// Batched matmul kernels
// ============================================================

__global__ void k_matmul_batched(const float* a, const float* b, float* c,
                                 int batch, int m, int n, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int mn = m * n;
    int total = batch * mn;
    if (tid < total) {
        int b_idx = tid / mn;
        int rem = tid % mn;
        int row = rem / n;
        int col = rem % n;
        float sum = 0.f;
        int a_off = b_idx * m * k;
        int b_off = b_idx * k * n;
        for (int i = 0; i < k; i++) sum += a[a_off + row * k + i] * b[b_off + i * n + col];
        c[b_idx * mn + row * n + col] = sum;
    }
}

__global__ void k_tensor3d_matmul(const float* a, const float* b, float* c,
                                  int batch, int m, int n, int k) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int mn = m * n;
    int total = batch * mn;
    if (tid < total) {
        int b_idx = tid / mn;
        int rem = tid % mn;
        int row = rem / n;
        int col = rem % n;
        float sum = 0.f;
        int a_off = b_idx * m * k;
        int b_off = b_idx * k * n;
        for (int i = 0; i < k; i++) sum += a[a_off + row * k + i] * b[b_off + i * n + col];
        c[b_idx * mn + row * n + col] = sum;
    }
}

#define BENCH_BATCHED_MATMUL(name, kernel)                                                               \
extern "C" float bench_##name(const float* h_a, const float* h_b, float* h_c,                           \
                              int batch, int m, int n, int k,                                            \
                              int grid, int block, int warmup, int iters) {                              \
    float *d_a, *d_b, *d_c;                                                                              \
    size_t sz_a = (size_t)batch * m * k * sizeof(float);                                                 \
    size_t sz_b = (size_t)batch * k * n * sizeof(float);                                                 \
    size_t sz_c = (size_t)batch * m * n * sizeof(float);                                                 \
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);                             \
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);                                                 \
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k);            \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k);                                         \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);                                                 \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);                                           \
    return med;                                                                                          \
}

BENCH_BATCHED_MATMUL(matmul_batched, k_matmul_batched)
BENCH_BATCHED_MATMUL(tensor3d_matmul, k_tensor3d_matmul)

// ============================================================
// Matvec kernels
// ============================================================

// y = A(M×N) * x(N), one thread per row
__global__ void k_matvec(const float* a, const float* x, float* y, int m, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        float sum = 0.f;
        int row_start = tid * n;
        for (int i = 0; i < n; i++) sum += a[row_start + i] * x[i];
        y[tid] = sum;
    }
}

extern "C" float bench_matvec_forward(const float* h_a, const float* h_x, float* h_y,
                                      int m, int n, int grid, int block,
                                      int warmup, int iters) {
    float *d_a, *d_x, *d_y;
    size_t sz_a = (size_t)m * n * sizeof(float);
    size_t sz_x = (size_t)n * sizeof(float);
    size_t sz_y = (size_t)m * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_x, sz_x); cudaMalloc(&d_y, sz_y);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sz_x, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_matvec<<<grid, block>>>(d_a, d_x, d_y, m, n);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_matvec<<<grid, block>>>(d_a, d_x, d_y, m, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_y, d_y, sz_y, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_x); cudaFree(d_y);
    return med;
}

// Scalar multiply
__global__ void k_scalar_multiply(const float* in, float* out, float s, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) out[tid] = in[tid] * s;
}

extern "C" float bench_scalar_multiply(const float* h_in, float* h_out,
                                       float s, int n, int grid, int block,
                                       int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_scalar_multiply<<<grid, block>>>(d_in, d_out, s, n);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_scalar_multiply<<<grid, block>>>(d_in, d_out, s, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// tensor3d_matvec — same as batched matmul (B is batch×K×N with N=1 for matvec, but the
// benchmark caller controls dimensions, so we reuse the batched matmul kernel pattern)
extern "C" float bench_tensor3d_matvec(const float* h_a, const float* h_b, float* h_c,
                                       int batch, int m, int n, int k,
                                       int grid, int block, int warmup, int iters) {
    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)batch * m * k * sizeof(float);
    size_t sz_b = (size_t)batch * k * n * sizeof(float);
    size_t sz_c = (size_t)batch * m * n * sizeof(float);
    cudaMalloc(&d_a, sz_a); cudaMalloc(&d_b, sz_b); cudaMalloc(&d_c, sz_c);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    for (int i = 0; i < warmup; i++) k_tensor3d_matmul<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_tensor3d_matmul<<<grid, block>>>(d_a, d_b, d_c, batch, m, n, k);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_c, d_c, sz_c, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return med;
}

// ============================================================
// Reduction kernels (shared memory tree reduction)
// ============================================================

__global__ void k_sum_reduce(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += in[row_start + i];
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = smem[0];
}

__global__ void k_mean_reduce(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += in[row_start + i];
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = smem[0] / (float)dim;
}

__global__ void k_max_reduce(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = -3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; if (v > val) val = v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = smem[0];
}

__global__ void k_min_reduce(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; if (v < val) val = v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] < smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[bid] = smem[0];
}

#define BENCH_REDUCE(name, kernel)                                                                       \
extern "C" float bench_##name(const float* h_in, float* h_out,                                          \
                              int batch, int dim, int block, int warmup, int iters) {                    \
    float *d_in, *d_out;                                                                                 \
    size_t sz_in = (size_t)batch * dim * sizeof(float);                                                  \
    size_t sz_out = (size_t)batch * sizeof(float);                                                       \
    cudaMalloc(&d_in, sz_in); cudaMalloc(&d_out, sz_out);                                               \
    cudaMemcpy(d_in, h_in, sz_in, cudaMemcpyHostToDevice);                                              \
    size_t smem = block * sizeof(float);                                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<batch, block, smem>>>(d_in, d_out, dim);                  \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<batch, block, smem>>>(d_in, d_out, dim);                                                \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sz_out, cudaMemcpyDeviceToHost);                                           \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_in); cudaFree(d_out);                                                       \
    return med;                                                                                          \
}

BENCH_REDUCE(sum_reduce, k_sum_reduce)
BENCH_REDUCE(mean_reduce, k_mean_reduce)
BENCH_REDUCE(max_reduce, k_max_reduce)
BENCH_REDUCE(min_reduce, k_min_reduce)

// ============================================================
// Argreduce kernels
// ============================================================

__global__ void k_argmax_reduce(const float* in, unsigned int* out, int dim) {
    extern __shared__ char shared_raw[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    float* sval = (float*)shared_raw;
    unsigned int* sidx = (unsigned int*)(shared_raw + bdim * sizeof(float));
    int row_start = bid * dim;
    float best = -3.4028235e38f;
    unsigned int best_idx = 0;
    for (int i = tid; i < dim; i += bdim) {
        float v = in[row_start + i];
        if (v > best) { best = v; best_idx = (unsigned int)i; }
    }
    sval[tid] = best; sidx[tid] = best_idx;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[bid] = sidx[0];
}

__global__ void k_argmin_reduce(const float* in, unsigned int* out, int dim) {
    extern __shared__ char shared_raw[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    float* sval = (float*)shared_raw;
    unsigned int* sidx = (unsigned int*)(shared_raw + bdim * sizeof(float));
    int row_start = bid * dim;
    float best = 3.4028235e38f;
    unsigned int best_idx = 0;
    for (int i = tid; i < dim; i += bdim) {
        float v = in[row_start + i];
        if (v < best) { best = v; best_idx = (unsigned int)i; }
    }
    sval[tid] = best; sidx[tid] = best_idx;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] < sval[tid]) { sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[bid] = sidx[0];
}

#define BENCH_ARGREDUCE(name, kernel)                                                                    \
extern "C" float bench_##name(const float* h_in, unsigned int* h_out,                                   \
                              int batch, int dim, int block, int warmup, int iters) {                    \
    float *d_in; unsigned int *d_out;                                                                    \
    size_t sz_in = (size_t)batch * dim * sizeof(float);                                                  \
    size_t sz_out = (size_t)batch * sizeof(unsigned int);                                                \
    cudaMalloc(&d_in, sz_in); cudaMalloc(&d_out, sz_out);                                               \
    cudaMemcpy(d_in, h_in, sz_in, cudaMemcpyHostToDevice);                                              \
    size_t smem = block * (sizeof(float) + sizeof(unsigned int));                                        \
    for (int i = 0; i < warmup; i++) kernel<<<batch, block, smem>>>(d_in, d_out, dim);                  \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<batch, block, smem>>>(d_in, d_out, dim);                                                \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sz_out, cudaMemcpyDeviceToHost);                                           \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_in); cudaFree(d_out);                                                       \
    return med;                                                                                          \
}

BENCH_ARGREDUCE(argmax_reduce, k_argmax_reduce)
BENCH_ARGREDUCE(argmin_reduce, k_argmin_reduce)

// ============================================================
// Softmax kernels (3-pass: max → sum_exp → normalize)
// ============================================================

__global__ void k_softmax(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;

    // Pass 1: max
    float val = -3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; if (v > val) val = v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: sum exp
    val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += expf(in[row_start + i] - row_max);
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    // Pass 3: normalize
    for (int i = tid; i < dim; i += bdim)
        out[bid * dim + i] = expf(in[row_start + i] - row_max) / row_sum;
}

__global__ void k_log_softmax(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;

    // Pass 1: max
    float val = -3.4028235e38f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; if (v > val) val = v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: sum exp
    val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += expf(in[row_start + i] - row_max);
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float log_sum = logf(smem[0]);
    __syncthreads();

    // Pass 3: log-softmax
    for (int i = tid; i < dim; i += bdim)
        out[bid * dim + i] = (in[row_start + i] - row_max) - log_sum;
}

#define BENCH_SOFTMAX(name, kernel)                                                                      \
extern "C" float bench_##name(const float* h_in, float* h_out,                                          \
                              int batch, int dim, int block, int warmup, int iters) {                    \
    float *d_in, *d_out;                                                                                 \
    size_t sz = (size_t)batch * dim * sizeof(float);                                                     \
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);                                                       \
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);                                                 \
    size_t smem = block * sizeof(float);                                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<batch, block, smem>>>(d_in, d_out, dim);                  \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<batch, block, smem>>>(d_in, d_out, dim);                                                \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);                                               \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_in); cudaFree(d_out);                                                       \
    return med;                                                                                          \
}

BENCH_SOFTMAX(softmax_forward, k_softmax)
BENCH_SOFTMAX(log_softmax_forward, k_log_softmax)

// ============================================================
// Norm kernels
// ============================================================

// RMS norm: x / sqrt(mean(x²) + eps)
__global__ void k_rms_norm(const float* in, float* out, int dim, float eps) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; val += v * v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(smem[0] / (float)dim + eps);
    __syncthreads();
    for (int i = tid; i < dim; i += bdim) out[bid * dim + i] = in[row_start + i] / rms;
}

// Frobenius norm: sqrt(Σx²), single block
__global__ void k_frobenius_norm(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x, bdim = blockDim.x;
    float val = 0.f;
    for (int i = tid; i < n; i += bdim) { float v = in[i]; val += v * v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sqrtf(smem[0]);
}

// L1 norm: x / Σ|x|
__global__ void k_l1_norm(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += fabsf(in[row_start + i]);
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_abs = smem[0];
    __syncthreads();
    for (int i = tid; i < dim; i += bdim) out[bid * dim + i] = in[row_start + i] / sum_abs;
}

// L2 norm: x / sqrt(Σx²)
__global__ void k_l2_norm(const float* in, float* out, int dim) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) { float v = in[row_start + i]; val += v * v; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float l2 = sqrtf(smem[0]);
    __syncthreads();
    for (int i = tid; i < dim; i += bdim) out[bid * dim + i] = in[row_start + i] / l2;
}

// Layer norm: (x - mean) / sqrt(var + eps)
__global__ void k_layer_norm(const float* in, float* out, int dim, float eps) {
    extern __shared__ float smem[];
    int bid = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    int row_start = bid * dim;

    // mean
    float val = 0.f;
    for (int i = tid; i < dim; i += bdim) val += in[row_start + i];
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)dim;
    __syncthreads();

    // variance
    val = 0.f;
    for (int i = tid; i < dim; i += bdim) { float d = in[row_start + i] - mean; val += d * d; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = 1.f / sqrtf(smem[0] / (float)dim + eps);
    __syncthreads();

    for (int i = tid; i < dim; i += bdim)
        out[bid * dim + i] = (in[row_start + i] - mean) * inv_std;
}

// Norm benchmarks — row-wise (batch blocks)
#define BENCH_NORM_ROW(name, kernel)                                                                     \
extern "C" float bench_##name(const float* h_in, float* h_out,                                          \
                              int batch, int dim, int block, int warmup, int iters) {                    \
    float *d_in, *d_out;                                                                                 \
    size_t sz = (size_t)batch * dim * sizeof(float);                                                     \
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);                                                       \
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);                                                 \
    size_t smem = block * sizeof(float);                                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<batch, block, smem>>>(d_in, d_out, dim);                  \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<batch, block, smem>>>(d_in, d_out, dim);                                                \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);                                               \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_in); cudaFree(d_out);                                                       \
    return med;                                                                                          \
}

BENCH_NORM_ROW(l1_norm_forward, k_l1_norm)
BENCH_NORM_ROW(l2_norm_forward, k_l2_norm)

// RMS norm and layer norm have eps parameter
extern "C" float bench_rms_norm_forward(const float* h_in, float* h_out,
                                        int batch, int dim, float eps,
                                        int block, int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)batch * dim * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    size_t smem = block * sizeof(float);
    for (int i = 0; i < warmup; i++) k_rms_norm<<<batch, block, smem>>>(d_in, d_out, dim, eps);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_rms_norm<<<batch, block, smem>>>(d_in, d_out, dim, eps);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

extern "C" float bench_layer_norm_forward(const float* h_in, float* h_out,
                                          int batch, int dim, float eps,
                                          int block, int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz = (size_t)batch * dim * sizeof(float);
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);
    size_t smem = block * sizeof(float);
    for (int i = 0; i < warmup; i++) k_layer_norm<<<batch, block, smem>>>(d_in, d_out, dim, eps);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_layer_norm<<<batch, block, smem>>>(d_in, d_out, dim, eps);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// Frobenius norm: single block, scalar output
extern "C" float bench_frobenius_norm_forward(const float* h_in, float* h_out,
                                              int n, int block, int warmup, int iters) {
    float *d_in, *d_out;
    size_t sz_in = (size_t)n * sizeof(float);
    size_t sz_out = sizeof(float);
    cudaMalloc(&d_in, sz_in); cudaMalloc(&d_out, sz_out);
    cudaMemcpy(d_in, h_in, sz_in, cudaMemcpyHostToDevice);
    size_t smem = block * sizeof(float);
    for (int i = 0; i < warmup; i++) k_frobenius_norm<<<1, block, smem>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_frobenius_norm<<<1, block, smem>>>(d_in, d_out, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sz_out, cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_in); cudaFree(d_out);
    return med;
}

// ============================================================
// Loss kernels (single-block global reduction)
// ============================================================

__global__ void k_mse_loss(const float* pred, const float* tgt, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x, bdim = blockDim.x;
    float val = 0.f;
    for (int i = tid; i < n; i += bdim) { float d = pred[i] - tgt[i]; val += d * d; }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0] / (float)n;
}

__global__ void k_huber_loss(const float* pred, const float* tgt, float* out, int n, float delta) {
    extern __shared__ float smem[];
    int tid = threadIdx.x, bdim = blockDim.x;
    float val = 0.f;
    for (int i = tid; i < n; i += bdim) {
        float d = pred[i] - tgt[i];
        float ad = fabsf(d);
        val += (ad <= delta) ? 0.5f * d * d : delta * (ad - 0.5f * delta);
    }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0] / (float)n;
}

__global__ void k_kl_div_loss(const float* log_pred, const float* tgt, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x, bdim = blockDim.x;
    float val = 0.f;
    for (int i = tid; i < n; i += bdim) {
        float t = tgt[i];
        if (t > 0.f) val += t * (logf(t) - log_pred[i]);
    }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0] / (float)n;
}

__global__ void k_hinge_loss(const float* pred, const float* tgt, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x, bdim = blockDim.x;
    float val = 0.f;
    for (int i = tid; i < n; i += bdim) {
        float v = 1.f - pred[i] * tgt[i];
        if (v > 0.f) val += v;
    }
    smem[tid] = val;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smem[0] / (float)n;
}

#define BENCH_LOSS_2IN(name, kernel)                                                                     \
extern "C" float bench_##name(const float* h_pred, const float* h_tgt, float* h_out,                    \
                              int n, int block, int warmup, int iters) {                                 \
    float *d_pred, *d_tgt, *d_out;                                                                       \
    size_t sz = (size_t)n * sizeof(float);                                                               \
    cudaMalloc(&d_pred, sz); cudaMalloc(&d_tgt, sz); cudaMalloc(&d_out, sizeof(float));                  \
    cudaMemcpy(d_pred, h_pred, sz, cudaMemcpyHostToDevice);                                              \
    cudaMemcpy(d_tgt, h_tgt, sz, cudaMemcpyHostToDevice);                                               \
    size_t smem = block * sizeof(float);                                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<1, block, smem>>>(d_pred, d_tgt, d_out, n);               \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<1, block, smem>>>(d_pred, d_tgt, d_out, n);                                             \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);                                    \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_pred); cudaFree(d_tgt); cudaFree(d_out);                                    \
    return med;                                                                                          \
}

BENCH_LOSS_2IN(mse_loss_forward, k_mse_loss)
BENCH_LOSS_2IN(kl_div_loss_forward, k_kl_div_loss)
BENCH_LOSS_2IN(hinge_loss_forward, k_hinge_loss)

// Huber loss (extra param: delta)
extern "C" float bench_huber_loss_forward(const float* h_pred, const float* h_tgt, float* h_out,
                                          int n, float delta, int block, int warmup, int iters) {
    float *d_pred, *d_tgt, *d_out;
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_pred, sz); cudaMalloc(&d_tgt, sz); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_pred, h_pred, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, h_tgt, sz, cudaMemcpyHostToDevice);
    size_t smem = block * sizeof(float);
    for (int i = 0; i < warmup; i++) k_huber_loss<<<1, block, smem>>>(d_pred, d_tgt, d_out, n, delta);
    cudaDeviceSynchronize();
    float* times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        k_huber_loss<<<1, block, smem>>>(d_pred, d_tgt, d_out, n, delta);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;
    }
    float med = median(times, iters);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(times); cudaFree(d_pred); cudaFree(d_tgt); cudaFree(d_out);
    return med;
}

// ============================================================
// Cumulative kernels (1 thread per row, sequential scan)
// ============================================================

__global__ void k_cumsum(const float* in, float* out, int dim) {
    int bid = blockIdx.x;
    int row_start = bid * dim;
    float acc = 0.f;
    for (int i = 0; i < dim; i++) { acc += in[row_start + i]; out[row_start + i] = acc; }
}

__global__ void k_cumprod(const float* in, float* out, int dim) {
    int bid = blockIdx.x;
    int row_start = bid * dim;
    float acc = 1.f;
    for (int i = 0; i < dim; i++) { acc *= in[row_start + i]; out[row_start + i] = acc; }
}

__global__ void k_cumsum_reverse(const float* in, float* out, int dim) {
    int bid = blockIdx.x;
    int row_start = bid * dim;
    float acc = 0.f;
    for (int i = dim - 1; i >= 0; i--) { acc += in[row_start + i]; out[row_start + i] = acc; }
}

__global__ void k_cumsum_exclusive(const float* in, float* out, int dim) {
    int bid = blockIdx.x;
    int row_start = bid * dim;
    float acc = 0.f;
    for (int i = 0; i < dim; i++) { out[row_start + i] = acc; acc += in[row_start + i]; }
}

#define BENCH_CUMULATIVE(name, kernel)                                                                   \
extern "C" float bench_##name(const float* h_in, float* h_out,                                          \
                              int batch, int dim, int warmup, int iters) {                               \
    float *d_in, *d_out;                                                                                 \
    size_t sz = (size_t)batch * dim * sizeof(float);                                                     \
    cudaMalloc(&d_in, sz); cudaMalloc(&d_out, sz);                                                       \
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);                                                 \
    for (int i = 0; i < warmup; i++) kernel<<<batch, 1>>>(d_in, d_out, dim);                             \
    cudaDeviceSynchronize();                                                                             \
    float* times = (float*)malloc(iters * sizeof(float));                                                \
    cudaEvent_t start, stop;                                                                             \
    cudaEventCreate(&start); cudaEventCreate(&stop);                                                     \
    for (int i = 0; i < iters; i++) {                                                                    \
        cudaEventRecord(start);                                                                          \
        kernel<<<batch, 1>>>(d_in, d_out, dim);                                                          \
        cudaEventRecord(stop); cudaEventSynchronize(stop);                                               \
        float ms; cudaEventElapsedTime(&ms, start, stop); times[i] = ms * 1000.f;                       \
    }                                                                                                    \
    float med = median(times, iters);                                                                    \
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);                                               \
    cudaEventDestroy(start); cudaEventDestroy(stop);                                                     \
    free(times); cudaFree(d_in); cudaFree(d_out);                                                       \
    return med;                                                                                          \
}

BENCH_CUMULATIVE(cumsum_forward, k_cumsum)
BENCH_CUMULATIVE(cumprod_forward, k_cumprod)
BENCH_CUMULATIVE(cumsum_reverse_forward, k_cumsum_reverse)
BENCH_CUMULATIVE(cumsum_exclusive_forward, k_cumsum_exclusive)
