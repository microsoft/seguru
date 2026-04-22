#include <stdio.h>
#include <cuda_runtime.h>

__global__ void lu_kernel1(float *a, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > k && j < N) {
        a[k*N+j] = a[k*N+j] / a[k*N+k];
    }
}

__global__ void lu_kernel2(float *a, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > k && i < N && j > k && j < N) {
        a[i*N+j] = a[i*N+j] - a[i*N+k] * a[k*N+j];
    }
}

int main() {
    const int N = 2048;
    size_t sz = (size_t)N*N*sizeof(float);

    float *h_a=(float*)malloc(sz);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            h_a[i*N+j] = (i==j) ? 2.0f : (float)((i+j)%1024)/2048.0f;

    float *d_a;
    cudaMalloc(&d_a,sz);
    cudaMemcpy(d_a,h_a,sz,cudaMemcpyHostToDevice);

    dim3 block(16,16);

    // Warmup (1 step)
    lu_kernel1<<<(N+15)/16,256>>>(d_a,N,0);
    dim3 gridW((N+15)/16,(N+15)/16);
    lu_kernel2<<<gridW,block>>>(d_a,N,0);
    cudaDeviceSynchronize();

    // Reset
    cudaMemcpy(d_a,h_a,sz,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=1;
    cudaEventRecord(start);
    for (int it=0;it<ITERS;it++) {
        for (int k=0;k<N;k++) {
            dim3 grid1((N+255)/256);
            lu_kernel1<<<grid1,256>>>(d_a,N,k);
            cudaDeviceSynchronize();

            int rem = N - k - 1;
            if (rem > 0) {
                dim3 grid2((rem+15)/16,(rem+15)/16);
                lu_kernel2<<<grid2,block>>>(d_a,N,k);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("lu CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a);
    free(h_a);
    return 0;
}
