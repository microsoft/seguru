#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mm3_kernel1(const float *A, const float *B, float *E, int NI, int NJ, int NK) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && j < NJ) {
        float sum = 0.0f;
        for (int k = 0; k < NK; k++)
            sum += A[i*NK+k] * B[k*NJ+j];
        E[i*NJ+j] = sum;
    }
}

__global__ void mm3_kernel2(const float *C, const float *D, float *F, int NJ, int NL, int NM) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < NJ && l < NL) {
        float sum = 0.0f;
        for (int k = 0; k < NM; k++)
            sum += C[j*NM+k] * D[k*NL+l];
        F[j*NL+l] = sum;
    }
}

__global__ void mm3_kernel3(const float *E, const float *F, float *G, int NI, int NJ, int NL) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < NI && l < NL) {
        float sum = 0.0f;
        for (int j = 0; j < NJ; j++)
            sum += E[i*NJ+j] * F[j*NL+l];
        G[i*NL+l] = sum;
    }
}

int main() {
    const int NI=512, NJ=512, NK=512, NL=512, NM=512;
    size_t szA=(size_t)NI*NK*sizeof(float), szB=(size_t)NK*NJ*sizeof(float);
    size_t szC=(size_t)NJ*NM*sizeof(float), szD=(size_t)NM*NL*sizeof(float);
    size_t szE=(size_t)NI*NJ*sizeof(float), szF=(size_t)NJ*NL*sizeof(float);
    size_t szG=(size_t)NI*NL*sizeof(float);

    float *h_a=(float*)malloc(szA), *h_b=(float*)malloc(szB);
    float *h_c=(float*)malloc(szC), *h_d=(float*)malloc(szD);
    for (int i=0;i<NI*NK;i++) h_a[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NK*NJ;i++) h_b[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NJ*NM;i++) h_c[i]=(float)(i%1024)/1024.0f;
    for (int i=0;i<NM*NL;i++) h_d[i]=(float)(i%1024)/1024.0f;

    float *d_a,*d_b,*d_c,*d_d,*d_e,*d_f,*d_g;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_b,szB);
    cudaMalloc(&d_c,szC); cudaMalloc(&d_d,szD);
    cudaMalloc(&d_e,szE); cudaMalloc(&d_f,szF);
    cudaMalloc(&d_g,szG);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,szB,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,szC,cudaMemcpyHostToDevice);
    cudaMemcpy(d_d,h_d,szD,cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 gridE((NJ+15)/16,(NI+15)/16);
    dim3 gridF((NL+15)/16,(NJ+15)/16);
    dim3 gridG((NL+15)/16,(NI+15)/16);

    mm3_kernel1<<<gridE,block>>>(d_a,d_b,d_e,NI,NJ,NK);
    mm3_kernel2<<<gridF,block>>>(d_c,d_d,d_f,NJ,NL,NM);
    mm3_kernel3<<<gridG,block>>>(d_e,d_f,d_g,NI,NJ,NL);
    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=100;
    cudaEventRecord(start);
    for (int i=0;i<ITERS;i++) {
        mm3_kernel1<<<gridE,block>>>(d_a,d_b,d_e,NI,NJ,NK);
        mm3_kernel2<<<gridF,block>>>(d_c,d_d,d_f,NJ,NL,NM);
        mm3_kernel3<<<gridG,block>>>(d_e,d_f,d_g,NI,NJ,NL);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("threemm CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, NI, ITERS);

    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);cudaFree(d_d);
    cudaFree(d_e);cudaFree(d_f);cudaFree(d_g);
    free(h_a);free(h_b);free(h_c);free(h_d);
    return 0;
}
