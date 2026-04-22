// Classic CUDA histogram kernel with shared memory + atomics
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void histogram(const unsigned int *data, unsigned int *bins, int n) {
    __shared__ unsigned int smem_bins[NUM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory bins to zero
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        smem_bins[i] = 0;
    }
    __syncthreads();

    // Accumulate into shared memory bins
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&smem_bins[data[i]], 1);
    }
    __syncthreads();

    // Merge shared memory bins into global bins
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], smem_bins[i]);
    }
}

int main() {
    const int N = 1 << 20;
    unsigned int *h_data = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned int *h_bins = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));

    // Fill with values 0..255
    for (int i = 0; i < N; i++) h_data[i] = i % NUM_BINS;

    unsigned int *d_data, *d_bins;
    cudaMalloc(&d_data, N * sizeof(unsigned int));
    cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int));

    int blockSize = 256;
    int numBlocks = min((N + blockSize - 1) / blockSize, 1024);
    histogram<<<numBlocks, blockSize>>>(d_data, d_bins, N);

    cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Verify: each bin should have N / NUM_BINS counts
    int expected = N / NUM_BINS;
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_bins[i] != expected) {
            printf("Error: bin[%d] = %u, expected %d\n", i, h_bins[i], expected);
            return 1;
        }
    }
    printf("PASSED\n");

    cudaFree(d_data); cudaFree(d_bins);
    free(h_data); free(h_bins);
    return 0;
}
