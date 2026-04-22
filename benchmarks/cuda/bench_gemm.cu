#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float *a, const float *b, float *c, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += a[i*N+k] * b[k*N+j];
        c[i*N+j] = sum;
    }
}

int main() {
    const int N = 512;
    size_t sz = N*N*sizeof(float);
    float *h_a = (float*)malloc(sz);
    float *h_b = (float*)malloc(sz);
    float *h_c = (float*)malloc(sz);
    for (int i = 0; i < N*N; i++) { h_a[i] = 1.0f; h_b[i] = 1.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz); cudaMalloc(&d_b, sz); cudaMalloc(&d_c, sz);
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (N+15)/16);

    // Warmup
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("gemm CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
