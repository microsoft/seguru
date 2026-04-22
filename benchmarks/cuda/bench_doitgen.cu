#include <stdio.h>
#include <cuda_runtime.h>

__global__ void doitgen_kernel1(const float *A, const float *C4, float *sum, int NR, int NQ, int NP) {
    int p = threadIdx.x;
    int qr = blockIdx.y * blockDim.y + threadIdx.y;
    int q = qr % NQ;
    int r = qr / NQ;
    if (r < NR && q < NQ && p < NP) {
        float s = 0.0f;
        for (int ss = 0; ss < NP; ss++)
            s += A[r*NQ*NP + q*NP + ss] * C4[ss*NP + p];
        sum[r*NQ*NP + q*NP + p] = s;
    }
}

__global__ void doitgen_kernel2(float *A, const float *sum, int NR, int NQ, int NP) {
    int p = threadIdx.x;
    int qr = blockIdx.y * blockDim.y + threadIdx.y;
    int q = qr % NQ;
    int r = qr / NQ;
    if (r < NR && q < NQ && p < NP) {
        A[r*NQ*NP + q*NP + p] = sum[r*NQ*NP + q*NP + p];
    }
}

int main() {
    const int NR = 128, NQ = 128, NP = 128;
    size_t szA = (size_t)NR*NQ*NP*sizeof(float);
    size_t szC = (size_t)NP*NP*sizeof(float);

    float *h_a=(float*)malloc(szA), *h_c4=(float*)malloc(szC);
    for (int i=0;i<NR*NQ*NP;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NP*NP;i++) h_c4[i]=(float)(i%1024)/1024.0f;

    float *d_a,*d_c4,*d_sum;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_c4,szC); cudaMalloc(&d_sum,szA);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c4,h_c4,szC,cudaMemcpyHostToDevice);

    dim3 block(NP, 8);
    dim3 grid(1, (NR*NQ+7)/8);

    doitgen_kernel1<<<grid,block>>>(d_a,d_c4,d_sum,NR,NQ,NP);
    doitgen_kernel2<<<grid,block>>>(d_a,d_sum,NR,NQ,NP);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        doitgen_kernel1<<<grid,block>>>(d_a,d_c4,d_sum,NR,NQ,NP);
        doitgen_kernel2<<<grid,block>>>(d_a,d_sum,NR,NQ,NP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("doitgen CUDA: %.3f us/iter (NR=%d, NQ=%d, NP=%d, %d iters)\n", ms*1000.0f/ITERS, NR, NQ, NP, ITERS);

    cudaFree(d_a);cudaFree(d_c4);cudaFree(d_sum);
    free(h_a);free(h_c4);
    return 0;
}
