#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bicg_kernel1(const float *A, const float *r, float *s, int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < NY) {
        float sum = 0.0f;
        for (int i = 0; i < NX; i++)
            sum += r[i] * A[i*NY+j];
        s[j] = sum;
    }
}

__global__ void bicg_kernel2(const float *A, const float *p, float *q, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        float sum = 0.0f;
        for (int j = 0; j < NY; j++)
            sum += A[i*NY+j] * p[j];
        q[i] = sum;
    }
}

int main() {
    const int NX = 4096, NY = 4096;
    size_t szA = (size_t)NX*NY*sizeof(float);
    size_t szR = NX*sizeof(float);
    size_t szS = NY*sizeof(float);
    size_t szP = NY*sizeof(float);
    size_t szQ = NX*sizeof(float);

    float *h_a=(float*)malloc(szA), *h_r=(float*)malloc(szR);
    float *h_p=(float*)malloc(szP);
    for (int i=0;i<NX*NY;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NX;i++) h_r[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NY;i++) h_p[i]=(float)(i%1024)/1024.0f;

    float *d_a,*d_r,*d_s,*d_p,*d_q;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_r,szR);
    cudaMalloc(&d_s,szS); cudaMalloc(&d_p,szP);
    cudaMalloc(&d_q,szQ);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_r,h_r,szR,cudaMemcpyHostToDevice);
    cudaMemcpy(d_p,h_p,szP,cudaMemcpyHostToDevice);

    dim3 block(256,1);
    dim3 grid1((NY+255)/256);
    dim3 grid2((NX+255)/256);

    bicg_kernel1<<<grid1,block>>>(d_a,d_r,d_s,NX,NY);
    bicg_kernel2<<<grid2,block>>>(d_a,d_p,d_q,NX,NY);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        bicg_kernel1<<<grid1,block>>>(d_a,d_r,d_s,NX,NY);
        bicg_kernel2<<<grid2,block>>>(d_a,d_p,d_q,NX,NY);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("bicg CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, NX, ITERS);

    cudaFree(d_a);cudaFree(d_r);cudaFree(d_s);cudaFree(d_p);cudaFree(d_q);
    free(h_a);free(h_r);free(h_p);
    return 0;
}
