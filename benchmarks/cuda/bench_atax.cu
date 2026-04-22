#include <stdio.h>
#include <cuda_runtime.h>

__global__ void atax_kernel1(const float *A, const float *x, float *tmp, int NX, int NY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        float sum = 0.0f;
        for (int j = 0; j < NY; j++)
            sum += A[i*NY+j] * x[j];
        tmp[i] = sum;
    }
}

__global__ void atax_kernel2(const float *A, const float *tmp, float *y, int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < NY) {
        float sum = 0.0f;
        for (int i = 0; i < NX; i++)
            sum += A[i*NY+j] * tmp[i];
        y[j] = sum;
    }
}

int main() {
    const int NX = 4096, NY = 4096;
    size_t szA = (size_t)NX*NY*sizeof(float);
    size_t szX = NY*sizeof(float);
    size_t szY = NY*sizeof(float);
    size_t szT = NX*sizeof(float);

    float *h_a=(float*)malloc(szA), *h_x=(float*)malloc(szX);
    float *h_y=(float*)malloc(szY);
    for (int i=0;i<NX*NY;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NY;i++) h_x[i]=(float)(i%1024)/1024.0f;

    float *d_a,*d_x,*d_y,*d_tmp;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_x,szX);
    cudaMalloc(&d_y,szY); cudaMalloc(&d_tmp,szT);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,szX,cudaMemcpyHostToDevice);
    cudaMemset(d_y,0,szY);

    dim3 block(256,1);
    dim3 grid1((NX+255)/256);
    dim3 grid2((NY+255)/256);

    atax_kernel1<<<grid1,block>>>(d_a,d_x,d_tmp,NX,NY);
    atax_kernel2<<<grid2,block>>>(d_a,d_tmp,d_y,NX,NY);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        atax_kernel1<<<grid1,block>>>(d_a,d_x,d_tmp,NX,NY);
        atax_kernel2<<<grid2,block>>>(d_a,d_tmp,d_y,NX,NY);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("atax CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, NX, ITERS);

    cudaFree(d_a);cudaFree(d_x);cudaFree(d_y);cudaFree(d_tmp);
    free(h_a);free(h_x);free(h_y);
    return 0;
}
