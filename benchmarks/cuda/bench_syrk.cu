#include <stdio.h>
#include <cuda_runtime.h>

__global__ void syrk_kernel(const float *A, float *C, int NI, int NJ, float alpha, float beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j <= i) {
        float sum = C[i*NI+j] * beta;
        for (int k = 0; k < NJ; k++)
            sum += alpha * A[i*NJ+k] * A[j*NJ+k];
        C[i*NI+j] = sum;
    }
}

int main() {
    const int NI = 1024, NJ = 1024;
    float alpha = 1.0f, beta = 0.0f;
    size_t szA = (size_t)NI*NJ*sizeof(float);
    size_t szC = (size_t)NI*NI*sizeof(float);

    float *h_a=(float*)malloc(szA);
    float *h_c=(float*)malloc(szC);
    for (int i=0;i<NI*NJ;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NI*NI;i++) h_c[i]=0.0f;

    float *d_a,*d_c;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_c,szC);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,szC,cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((NI+15)/16,(NI+15)/16);

    syrk_kernel<<<grid,block>>>(d_a,d_c,NI,NJ,alpha,beta);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        syrk_kernel<<<grid,block>>>(d_a,d_c,NI,NJ,alpha,beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("syrk CUDA: %.3f us/iter (NI=%d, NJ=%d, %d iters)\n", ms*1000.0f/ITERS, NI, NJ, ITERS);

    cudaFree(d_a);cudaFree(d_c);
    free(h_a);free(h_c);
    return 0;
}
