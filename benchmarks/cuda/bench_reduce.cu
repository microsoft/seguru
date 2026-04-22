#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float *input, float *output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = smem[0];
}

int main() {
    const int N = 1 << 20;
    const int BS = 256;
    const int NB = (N + BS*2 - 1) / (BS*2);
    float *h_in = (float*)malloc(N*sizeof(float));
    float *h_out = (float*)malloc(NB*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, NB*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    reduce_sum<<<NB, BS, BS*sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 100;
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++) {
        reduce_sum<<<NB, BS, BS*sizeof(float)>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("reduce CUDA: %.3f us/iter (N=%d, %d iters)\n", ms*1000.0f/ITERS, N, ITERS);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
