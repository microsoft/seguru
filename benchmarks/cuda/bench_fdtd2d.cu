#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fdtd_step1(float *ey, const float *hz, const float *fict, int NX, int NY, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX && j < NY) {
        if (i == 0)
            ey[i*NY+j] = fict[t];
        else
            ey[i*NY+j] -= 0.5f * (hz[i*NY+j] - hz[(i-1)*NY+j]);
    }
}

__global__ void fdtd_step2(float *ex, const float *hz, int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX && j < NY && j > 0) {
        ex[i*NY+j] -= 0.5f * (hz[i*NY+j] - hz[i*NY+(j-1)]);
    }
}

__global__ void fdtd_step3(float *hz, const float *ex, const float *ey, int NX, int NY) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NX-1 && j < NY-1) {
        hz[i*NY+j] -= 0.7f * (ex[i*NY+(j+1)] - ex[i*NY+j] + ey[(i+1)*NY+j] - ey[i*NY+j]);
    }
}

int main() {
    const int NX = 2048, NY = 2048, TMAX = 500;
    size_t sz = (size_t)NX*NY*sizeof(float);
    size_t szF = TMAX*sizeof(float);

    float *h_ex=(float*)malloc(sz), *h_ey=(float*)malloc(sz);
    float *h_hz=(float*)malloc(sz), *h_fict=(float*)malloc(szF);
    for (int i=0;i<NX*NY;i++) { h_ex[i]=0.0f; h_ey[i]=0.0f; h_hz[i]=0.0f; }
    for (int i=0;i<TMAX;i++) h_fict[i]=(float)i;

    float *d_ex,*d_ey,*d_hz,*d_fict;
    cudaMalloc(&d_ex,sz); cudaMalloc(&d_ey,sz);
    cudaMalloc(&d_hz,sz); cudaMalloc(&d_fict,szF);
    cudaMemcpy(d_ex,h_ex,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ey,h_ey,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_hz,h_hz,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_fict,h_fict,szF,cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((NY+15)/16,(NX+15)/16);

    // Warmup (1 timestep)
    fdtd_step1<<<grid,block>>>(d_ey,d_hz,d_fict,NX,NY,0);
    fdtd_step2<<<grid,block>>>(d_ex,d_hz,NX,NY);
    fdtd_step3<<<grid,block>>>(d_hz,d_ex,d_ey,NX,NY);
    cudaDeviceSynchronize();

    // Reset
    cudaMemcpy(d_ex,h_ex,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ey,h_ey,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_hz,h_hz,sz,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=1;
    cudaEventRecord(start);
    for (int it=0;it<ITERS;it++) {
        for (int t=0;t<TMAX;t++) {
            fdtd_step1<<<grid,block>>>(d_ey,d_hz,d_fict,NX,NY,t);
            fdtd_step2<<<grid,block>>>(d_ex,d_hz,NX,NY);
            fdtd_step3<<<grid,block>>>(d_hz,d_ex,d_ey,NX,NY);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("fdtd2d CUDA: %.3f us/iter (NX=%d, NY=%d, TMAX=%d, %d iters)\n", ms*1000.0f/ITERS, NX, NY, TMAX, ITERS);

    cudaFree(d_ex);cudaFree(d_ey);cudaFree(d_hz);cudaFree(d_fict);
    free(h_ex);free(h_ey);free(h_hz);free(h_fict);
    return 0;
}
