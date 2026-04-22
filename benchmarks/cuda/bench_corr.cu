#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define FLOAT_N 3214212.01f
#define EPS 0.005f

__global__ void corr_mean_kernel(const float *data, float *mean, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            sum += data[i*M+j];
        mean[j] = sum / FLOAT_N;
    }
}

__global__ void corr_std_kernel(const float *data, const float *mean, float *stddev, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = data[i*M+j] - mean[j];
            sum += diff * diff;
        }
        stddev[j] = sqrtf(sum / FLOAT_N);
        if (stddev[j] <= EPS)
            stddev[j] = 1.0f;
    }
}

__global__ void corr_reduce_kernel(float *data, const float *mean, const float *stddev, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M) {
        data[i*M+j] = (data[i*M+j] - mean[j]) / (sqrtf((float)FLOAT_N) * stddev[j]);
    }
}

__global__ void corr_corr_kernel(const float *data, float *symmat, int M, int N) {
    int j2 = blockIdx.x * blockDim.x + threadIdx.x;
    int j1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (j1 < M && j2 < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            sum += data[i*M+j1] * data[i*M+j2];
        symmat[j1*M+j2] = sum;
    }
}

int main() {
    const int M = 2048, N = 2048;
    size_t szD = (size_t)N*M*sizeof(float);
    size_t szV = M*sizeof(float);
    size_t szS = (size_t)M*M*sizeof(float);

    float *h_data=(float*)malloc(szD);
    for (int i=0;i<N*M;i++) h_data[i]=(float)(i%1024)/1024.0f;

    float *d_data,*d_mean,*d_stddev,*d_symmat;
    cudaMalloc(&d_data,szD); cudaMalloc(&d_mean,szV);
    cudaMalloc(&d_stddev,szV); cudaMalloc(&d_symmat,szS);
    cudaMemcpy(d_data,h_data,szD,cudaMemcpyHostToDevice);

    dim3 block1(256);
    dim3 grid1((M+255)/256);
    dim3 block2(16,16);
    dim3 grid2((M+15)/16,(N+15)/16);
    dim3 grid3((M+15)/16,(M+15)/16);

    // Warmup
    corr_mean_kernel<<<grid1,block1>>>(d_data,d_mean,M,N);
    corr_std_kernel<<<grid1,block1>>>(d_data,d_mean,d_stddev,M,N);
    corr_reduce_kernel<<<grid2,block2>>>(d_data,d_mean,d_stddev,M,N);
    corr_corr_kernel<<<grid3,block2>>>(d_data,d_symmat,M,N);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int it=0;it<ITERS;it++) {
        cudaMemcpy(d_data,h_data,szD,cudaMemcpyHostToDevice);
        corr_mean_kernel<<<grid1,block1>>>(d_data,d_mean,M,N);
        corr_std_kernel<<<grid1,block1>>>(d_data,d_mean,d_stddev,M,N);
        corr_reduce_kernel<<<grid2,block2>>>(d_data,d_mean,d_stddev,M,N);
        corr_corr_kernel<<<grid3,block2>>>(d_data,d_symmat,M,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("corr CUDA: %.3f us/iter (M=%d, N=%d, %d iters)\n", ms*1000.0f/ITERS, M, N, ITERS);

    cudaFree(d_data);cudaFree(d_mean);cudaFree(d_stddev);cudaFree(d_symmat);
    free(h_data);
    return 0;
}
