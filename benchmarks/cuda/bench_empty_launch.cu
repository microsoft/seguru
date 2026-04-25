#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void empty_launch_kernel(float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = 0.0f;
    }
}

int main(int argc, char **argv) {
    int iters = 20000;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters <= 0) {
            iters = 20000;
        }
    }

    float *d_out;
    cudaMalloc(&d_out, sizeof(float));

    for (int i = 0; i < 100; i++) {
        empty_launch_kernel<<<1, 1>>>(d_out);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        empty_launch_kernel<<<1, 1>>>(d_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("empty_launch CUDA: %.3f us/iter (%d iters)\n", ms * 1000.0f / iters, iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    return 0;
}
