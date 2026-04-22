#include <stdio.h>
#include <cuda_runtime.h>

__global__ void jacobi2d_kernel1(const float *a, float *b, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        b[i*N+j] = 0.2f * (a[i*N+j] + a[i*N+j-1] + a[i*N+j+1] + a[(i+1)*N+j] + a[(i-1)*N+j]);
    }
}

__global__ void jacobi2d_kernel2(float *a, const float *b, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        a[i*N+j] = b[i*N+j];
    }
}

int main() {
    const int N = 1000, TSTEPS = 20;
    size_t sz = (size_t)N*N*sizeof(float);

    float *h_a=(float*)malloc(sz), *h_b=(float*)malloc(sz);
    for (int i=0;i<N*N;i++) { h_a[i]=(float)(i%1024)/1024.0f; h_b[i]=0.0f; }

    float *d_a,*d_b;
    cudaMalloc(&d_a,sz); cudaMalloc(&d_b,sz);
    cudaMemcpy(d_a,h_a,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,sz,cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((N+15)/16,(N+15)/16);

    // Warmup
    jacobi2d_kernel1<<<grid,block>>>(d_a,d_b,N);
    jacobi2d_kernel2<<<grid,block>>>(d_a,d_b,N);
    cudaDeviceSynchronize();

    // Reset
    cudaMemcpy(d_a,h_a,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,sz,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=1;
    cudaEventRecord(start);
    for (int it=0;it<ITERS;it++) {
        for (int t=0;t<TSTEPS;t++) {
            jacobi2d_kernel1<<<grid,block>>>(d_a,d_b,N);
            jacobi2d_kernel2<<<grid,block>>>(d_a,d_b,N);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("jacobi2d CUDA: %.3f us/iter (N=%d, TSTEPS=%d, %d iters)\n", ms*1000.0f/ITERS, N, TSTEPS, ITERS);

    cudaFree(d_a);cudaFree(d_b);
    free(h_a);free(h_b);
    return 0;
}
