#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void gs_norm(const float *a, float *r, int NI, int NJ, int k) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float nrm = 0.0f;
        for (int i = 0; i < NI; i++)
            nrm += a[i*NJ+k] * a[i*NJ+k];
        r[k*NJ+k] = sqrtf(nrm);
    }
}

__global__ void gs_normalize(float *q, const float *a, const float *r, int NI, int NJ, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NI) {
        q[i*NJ+k] = a[i*NJ+k] / r[k*NJ+k];
    }
}

__global__ void gs_dot(const float *q, const float *a, float *r, int NI, int NJ, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (j < NJ) {
        float sum = 0.0f;
        for (int i = 0; i < NI; i++)
            sum += q[i*NJ+k] * a[i*NJ+j];
        r[k*NJ+j] = sum;
    }
}

__global__ void gs_update(const float *q, const float *r, float *a, int NI, int NJ, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < NJ && i < NI) {
        a[i*NJ+j] -= q[i*NJ+k] * r[k*NJ+j];
    }
}

int main() {
    const int NI = 2048, NJ = 2048;
    size_t szA = (size_t)NI*NJ*sizeof(float);
    size_t szR = (size_t)NJ*NJ*sizeof(float);

    float *h_a=(float*)malloc(szA);
    for (int i=0;i<NI*NJ;i++) h_a[i]=(float)(i%1024)/1024.0f + 0.001f;

    float *d_a,*d_q,*d_r;
    cudaMalloc(&d_a,szA); cudaMalloc(&d_q,szA); cudaMalloc(&d_r,szR);
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);

    dim3 block1(256);
    dim3 block2(16,16);

    // Warmup: compute the column norm on GPU, matching the timed dataflow.
    {
        gs_norm<<<1,1>>>(d_a,d_r,NI,NJ,0);
        gs_normalize<<<(NI+255)/256,block1>>>(d_q,d_a,d_r,NI,NJ,0);
        cudaDeviceSynchronize();
    }

    // Reset
    cudaMemcpy(d_a,h_a,szA,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int ITERS=1;
    cudaEventRecord(start);
    for (int it=0;it<ITERS;it++) {
        for (int k=0;k<NJ;k++) {
            gs_norm<<<1,1>>>(d_a,d_r,NI,NJ,k);
            gs_normalize<<<(NI+255)/256,block1>>>(d_q,d_a,d_r,NI,NJ,k);
            cudaDeviceSynchronize();

            int rem = NJ - k - 1;
            if (rem > 0) {
                gs_dot<<<(rem+255)/256,block1>>>(d_q,d_a,d_r,NI,NJ,k);
                cudaDeviceSynchronize();

                dim3 gridU((rem+15)/16,(NI+15)/16);
                gs_update<<<gridU,block2>>>(d_q,d_r,d_a,NI,NJ,k);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("gramschm CUDA: %.3f us/iter (NI=%d, NJ=%d, %d iters)\n", ms*1000.0f/ITERS, NI, NJ, ITERS);

    cudaFree(d_a);cudaFree(d_q);cudaFree(d_r);
    free(h_a);
    return 0;
}
