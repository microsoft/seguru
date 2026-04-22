#include <stdio.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float *A, float *B, int NI, int NJ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    float c11=0.2f, c21=0.5f, c31=-0.8f, c12=-0.3f, c22=0.6f, c32=-0.9f, c13=0.4f, c23=0.7f, c33=0.10f;
    if (i > 0 && i < NI-1 && j > 0 && j < NJ-1) {
        B[i*NJ+j] = c11*A[(i-1)*NJ+(j-1)] + c12*A[(i+0)*NJ+(j-1)] + c13*A[(i+1)*NJ+(j-1)]
                   + c21*A[(i-1)*NJ+(j+0)] + c22*A[(i+0)*NJ+(j+0)] + c23*A[(i+1)*NJ+(j+0)]
                   + c31*A[(i-1)*NJ+(j+1)] + c32*A[(i+0)*NJ+(j+1)] + c33*A[(i+1)*NJ+(j+1)];
    }
}

int main() {
    const int NI = 4096, NJ = 4096;
    size_t sz = (size_t)NI * NJ * sizeof(float);
    float *h_a = (float*)malloc(sz);
    float *h_b = (float*)malloc(sz);
    for (int i = 0; i < NI*NJ; i++) { h_a[i] = (float)(i % 1024) / 1024.0f; }

    float *d_a, *d_b;
    cudaMalloc(&d_a, sz); cudaMalloc(&d_b, sz);
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);

    dim3 block(32, 8);
    dim3 grid((NJ+31)/32, (NI+7)/8);

    conv2d_kernel<<<grid, block>>>(d_a, d_b, NI, NJ);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        conv2d_kernel<<<grid, block>>>(d_a, d_b, NI, NJ);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("conv2d CUDA: %.3f us/iter (NI=%d, NJ=%d, %d iters)\n", ms*1000.0f/ITERS, NI, NJ, ITERS);

    cudaFree(d_a); cudaFree(d_b);
    free(h_a); free(h_b);
    return 0;
}
