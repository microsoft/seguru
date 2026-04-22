#include <stdio.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float *A, float *B, int NI, int NJ, int NK) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int qr = blockIdx.y * blockDim.y + threadIdx.y;
    int i = qr / NJ;
    int j = qr % NJ;

    float c11=2.0f, c12=5.0f, c13=-8.0f;
    float c21=-3.0f, c22=6.0f, c23=-9.0f;
    float c31=4.0f, c32=7.0f, c33=10.0f;

    if (i > 0 && i < NI-1 && j > 0 && j < NJ-1 && k > 0 && k < NK-1) {
        B[i*NJ*NK + j*NK + k] =
            c11*A[(i-1)*NJ*NK + (j-1)*NK + (k-1)] + c12*A[(i-1)*NJ*NK + (j-1)*NK + k] + c13*A[(i-1)*NJ*NK + (j-1)*NK + (k+1)]
          + c21*A[(i-1)*NJ*NK + j*NK + (k-1)]     + c22*A[(i-1)*NJ*NK + j*NK + k]     + c23*A[(i-1)*NJ*NK + j*NK + (k+1)]
          + c31*A[(i-1)*NJ*NK + (j+1)*NK + (k-1)] + c32*A[(i-1)*NJ*NK + (j+1)*NK + k] + c33*A[(i-1)*NJ*NK + (j+1)*NK + (k+1)]
          + c11*A[i*NJ*NK + (j-1)*NK + (k-1)]     + c12*A[i*NJ*NK + (j-1)*NK + k]     + c13*A[i*NJ*NK + (j-1)*NK + (k+1)]
          + c21*A[i*NJ*NK + j*NK + (k-1)]         + c22*A[i*NJ*NK + j*NK + k]         + c23*A[i*NJ*NK + j*NK + (k+1)]
          + c31*A[i*NJ*NK + (j+1)*NK + (k-1)]     + c32*A[i*NJ*NK + (j+1)*NK + k]     + c33*A[i*NJ*NK + (j+1)*NK + (k+1)]
          + c11*A[(i+1)*NJ*NK + (j-1)*NK + (k-1)] + c12*A[(i+1)*NJ*NK + (j-1)*NK + k] + c13*A[(i+1)*NJ*NK + (j-1)*NK + (k+1)]
          + c21*A[(i+1)*NJ*NK + j*NK + (k-1)]     + c22*A[(i+1)*NJ*NK + j*NK + k]     + c23*A[(i+1)*NJ*NK + j*NK + (k+1)]
          + c31*A[(i+1)*NJ*NK + (j+1)*NK + (k-1)] + c32*A[(i+1)*NJ*NK + (j+1)*NK + k] + c33*A[(i+1)*NJ*NK + (j+1)*NK + (k+1)];
    }
}

int main() {
    const int NI = 256, NJ = 256, NK = 256;
    size_t sz = (size_t)NI * NJ * NK * sizeof(float);
    float *h_a = (float*)malloc(sz);
    float *h_b = (float*)malloc(sz);
    for (int i = 0; i < NI*NJ*NK; i++) h_a[i] = (float)(i % 1024) / 1024.0f;

    float *d_a, *d_b;
    cudaMalloc(&d_a, sz); cudaMalloc(&d_b, sz);
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);

    dim3 block(32, 8);
    dim3 grid((NK+31)/32, (NI*NJ+7)/8);

    conv3d_kernel<<<grid, block>>>(d_a, d_b, NI, NJ, NK);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        conv3d_kernel<<<grid, block>>>(d_a, d_b, NI, NJ, NK);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("conv3d CUDA: %.3f us/iter (NI=%d, NJ=%d, NK=%d, %d iters)\n", ms*1000.0f/ITERS, NI, NJ, NK, ITERS);

    cudaFree(d_a); cudaFree(d_b);
    free(h_a); free(h_b);
    return 0;
}
