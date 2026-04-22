// Classic CUDA parallel reduction (sum) kernel
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float *input, float *output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread into shared memory
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    smem[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

int main() {
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(NUM_BLOCKS * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, NUM_BLOCKS * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; i++) total += h_output[i];
    printf("Sum = %f (expected %f)\n", total, (float)N);

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    return 0;
}
