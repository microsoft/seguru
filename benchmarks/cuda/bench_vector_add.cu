#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20; // 1M elements
    float *h_a = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    float *h_c = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

    int bs = 256;
    int nb = (N + bs - 1) / bs;

    // Warmup
    vector_add<<<nb, bs>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        vector_add<<<nb, bs>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("vector_add CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
