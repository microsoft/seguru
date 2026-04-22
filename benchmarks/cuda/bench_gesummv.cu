#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gesummv_kernel(const float *A, const float *B, const float *x, float *y, int N, float alpha, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sumA = 0.0f, sumB = 0.0f;
        for (int j = 0; j < N; j++) {
            sumA += A[i*N+j] * x[j];
            sumB += B[i*N+j] * x[j];
        }
        y[i] = alpha * sumA + beta * sumB;
    }
}

int main() {
    const int N = 4096;
    float alpha = 1.0f, beta = 1.0f;
    size_t szM = (size_t)N*N*sizeof(float);
    size_t szV = N*sizeof(float);

    float *h_a=(float*)malloc(szM), *h_b=(float*)malloc(szM);
    float *h_x=(float*)malloc(szV), *h_y=(float*)malloc(szV);
    for (int i=0;i<N*N;i++) { h_a[i]=(float)(i%1024)/1024.0f; h_b[i]=(float)(i%512)/512.0f; }
    for (int i=0;i<N;i++) h_x[i]=(float)(i%1024)/1024.0f;

    float *d_a,*d_b,*d_x,*d_y;
    cudaMalloc(&d_a,szM); cudaMalloc(&d_b,szM);
    cudaMalloc(&d_x,szV); cudaMalloc(&d_y,szV);
    cudaMemcpy(d_a,h_a,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,szV,cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N+255)/256);

    gesummv_kernel<<<grid,block>>>(d_a,d_b,d_x,d_y,N,alpha,beta);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        gesummv_kernel<<<grid,block>>>(d_a,d_b,d_x,d_y,N,alpha,beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("gesummv CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a);cudaFree(d_b);cudaFree(d_x);cudaFree(d_y);
    free(h_a);free(h_b);free(h_x);free(h_y);
    return 0;
}
