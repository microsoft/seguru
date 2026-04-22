#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mm2_kernel1(const float *A, const float *B, float *tmp, int NI, int NJ, int NK, float alpha) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j < NJ) {
        float sum = 0.0f;
        for (int k = 0; k < NK; k++)
            sum += alpha * A[i*NK+k] * B[k*NJ+j];
        tmp[i*NJ+j] = sum;
    }
}

__global__ void mm2_kernel2(const float *tmp, const float *C, float *D, int NI, int NJ, int NL, float beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j < NL) {
        float sum = D[i*NL+j] * beta;
        for (int k = 0; k < NJ; k++)
            sum += tmp[i*NJ+k] * C[k*NL+j];
        D[i*NL+j] = sum;
    }
}

int main() {
    const int NI = 1024, NJ = 1024, NK = 1024, NL = 1024;
    float alpha = 1.0f, beta = 1.0f;
    size_t szA = (size_t)NI*NK*sizeof(float);
    size_t szB = (size_t)NK*NJ*sizeof(float);
    size_t szC = (size_t)NJ*NL*sizeof(float);
    size_t szD = (size_t)NI*NL*sizeof(float);
    size_t szT = (size_t)NI*NJ*sizeof(float);

    float *h_a=(float*)malloc(szA), *h_b=(float*)malloc(szB);
    float *h_c=(float*)malloc(szC), *h_d=(float*)malloc(szD);
    for (int i = 0; i < NI*NK; i++) h_a[i] = (float)(i%1024)/1024.0f;
    for (int i = 0; i < NK*NJ; i++) h_b[i] = (float)(i%1024)/1024.0f;
    for (int i = 0; i < NJ*NL; i++) h_c[i] = (float)(i%1024)/1024.0f;
    for (int i = 0; i < NI*NL; i++) h_d[i] = (float)(i%1024)/1024.0f;

    float *d_a, *d_b, *d_c, *d_d, *d_tmp;
    cudaMalloc(&d_a, szA); cudaMalloc(&d_b, szB);
    cudaMalloc(&d_c, szC); cudaMalloc(&d_d, szD);
    cudaMalloc(&d_tmp, szT);
    cudaMemcpy(d_a, h_a, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, szB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, szC, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, szD, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid1((NJ+15)/16, (NI+15)/16);
    dim3 grid2((NL+15)/16, (NI+15)/16);

    mm2_kernel1<<<grid1, block>>>(d_a, d_b, d_tmp, NI, NJ, NK, alpha);
    mm2_kernel2<<<grid2, block>>>(d_tmp, d_c, d_d, NI, NJ, NL, beta);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        mm2_kernel1<<<grid1, block>>>(d_a, d_b, d_tmp, NI, NJ, NK, alpha);
        mm2_kernel2<<<grid2, block>>>(d_tmp, d_c, d_d, NI, NJ, NL, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("twomm CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, NI, ITERS);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_tmp);
    free(h_a); free(h_b); free(h_c); free(h_d);
    return 0;
}
