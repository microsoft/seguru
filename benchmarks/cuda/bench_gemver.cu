#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gemver_kernel1(float *A, const float *u1, const float *v1, const float *u2, const float *v2, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        A[i*N+j] += u1[i]*v1[j] + u2[i]*v2[j];
    }
}

__global__ void gemver_kernel2(const float *A, const float *y, float *x, int N, float beta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += beta * A[j*N+i] * y[j];
        x[i] = x[i] + sum;
    }
}

__global__ void gemver_kernel3(float *x, const float *z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = x[i] + z[i];
    }
}

__global__ void gemver_kernel4(const float *A, const float *x, float *w, int N, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += alpha * A[i*N+j] * x[j];
        w[i] = sum;
    }
}

int main() {
    const int N = 4096;
    float alpha = 1.0f, beta = 1.0f;
    size_t szM = (size_t)N*N*sizeof(float);
    size_t szV = N*sizeof(float);

    float *h_a=(float*)malloc(szM);
    float *h_u1=(float*)malloc(szV), *h_v1=(float*)malloc(szV);
    float *h_u2=(float*)malloc(szV), *h_v2=(float*)malloc(szV);
    float *h_y=(float*)malloc(szV), *h_z=(float*)malloc(szV);
    for (int i=0;i<N*N;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<N;i++) { h_u1[i]=(float)(i%1024)/1024.0f; h_v1[i]=(float)(i%512)/512.0f; h_u2[i]=(float)(i%256)/256.0f; h_v2[i]=(float)(i%128)/128.0f; h_y[i]=(float)(i%1024)/1024.0f; h_z[i]=(float)(i%1024)/1024.0f; }

    float *d_a,*d_u1,*d_v1,*d_u2,*d_v2,*d_x,*d_y,*d_z,*d_w;
    cudaMalloc(&d_a,szM);
    cudaMalloc(&d_u1,szV); cudaMalloc(&d_v1,szV);
    cudaMalloc(&d_u2,szV); cudaMalloc(&d_v2,szV);
    cudaMalloc(&d_x,szV); cudaMalloc(&d_y,szV);
    cudaMalloc(&d_z,szV); cudaMalloc(&d_w,szV);
    cudaMemcpy(d_a,h_a,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u1,h_u1,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1,h_v1,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2,h_u2,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2,h_v2,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,szV,cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,szV,cudaMemcpyHostToDevice);
    cudaMemset(d_x,0,szV); cudaMemset(d_w,0,szV);

    dim3 block2d(16,16);
    dim3 grid2d((N+15)/16,(N+15)/16);
    dim3 block1d(256);
    dim3 grid1d((N+255)/256);

    // Warmup
    gemver_kernel1<<<grid2d,block2d>>>(d_a,d_u1,d_v1,d_u2,d_v2,N);
    gemver_kernel2<<<grid1d,block1d>>>(d_a,d_y,d_x,N,beta);
    gemver_kernel3<<<grid1d,block1d>>>(d_x,d_z,N);
    gemver_kernel4<<<grid1d,block1d>>>(d_a,d_x,d_w,N,alpha);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        gemver_kernel1<<<grid2d,block2d>>>(d_a,d_u1,d_v1,d_u2,d_v2,N);
        gemver_kernel2<<<grid1d,block1d>>>(d_a,d_y,d_x,N,beta);
        gemver_kernel3<<<grid1d,block1d>>>(d_x,d_z,N);
        gemver_kernel4<<<grid1d,block1d>>>(d_a,d_x,d_w,N,alpha);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("gemver CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_a);cudaFree(d_u1);cudaFree(d_v1);cudaFree(d_u2);cudaFree(d_v2);
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_z);cudaFree(d_w);
    free(h_a);free(h_u1);free(h_v1);free(h_u2);free(h_v2);
    free(h_y);free(h_z);
    return 0;
}
