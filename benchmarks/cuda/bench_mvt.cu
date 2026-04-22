#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mvt_kernel1(const float *A, const float *y1, float *x1, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += A[i*N+j] * y1[j];
        x1[i] += sum;
    }
}

__global__ void mvt_kernel2(const float *A, const float *y2, float *x2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += A[j*N+i] * y2[j];
        x2[i] += sum;
    }
}

int main() {
    const int N = 4096;
    size_t szM = (size_t)N*N*sizeof(float);
    size_t szV = N*sizeof(float);

    float *h_a=(float*)malloc(szM);
    float *h_y1=(float*)malloc(szV), *h_y2=(float*)malloc(szV);
    float *h_x1=(float*)malloc(szV), *h_x2=(float*)malloc(szV);
    for (int i=0;i<N*N;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<N;i++) { h_y1[i]=(float)(i%1024)/1024.0f; h_y2[i]=(float)(i%512)/512.0f; h_x1[i]=0.0f; h_x2[i]=0.0f; }

    float *d_a,*d_y1,*d_y2,*d_x1,*d_x2;
    cudaMalloc(&d_a,szM); cudaMalloc(&d_y1,szV); cudaMalloc(&d_y2,szV);
    cudaMalloc(&d_x1,szV); cudaMalloc(&d_x2,szV);
    cudaMemcpy(d_a,h_a,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1,h_y1,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2,h_y2,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1,h_x1,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2,h_x2,szV,cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N+255)/256);

    mvt_kernel1<<<grid,block>>>(d_a,d_y1,d_x1,N);
    mvt_kernel2<<<grid,block>>>(d_a,d_y2,d_x2,N);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        mvt_kernel1<<<grid,block>>>(d_a,d_y1,d_x1,N);
        mvt_kernel2<<<grid,block>>>(d_a,d_y2,d_x2,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("mvt CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a);cudaFree(d_y1);cudaFree(d_y2);cudaFree(d_x1);cudaFree(d_x2);
    free(h_a);free(h_y1);free(h_y2);free(h_x1);free(h_x2);
    return 0;
}
