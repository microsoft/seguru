#include <cstdio>
#include <ctime>

#include <cuda_runtime.h>

#define BLOCK_SIZE      1

__global__ void innerProductKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i, j;

    for (i = 0; i < N / BLOCK_SIZE / blockDim.y; i++) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (j = 0; j < N / BLOCK_SIZE / blockDim.x; j++) {
            if(row < N && col < N) {
                float sum = 0;
                int64_t a_idx = row * N;
                int64_t b_idx = col;
                for(int k = 0; k < N; ++k) {
                    sum += A[a_idx] * B[b_idx];
                    a_idx += 1;
                    b_idx += N;
                }
                C[row * N + col] = sum;
            }
            col += BLOCK_SIZE * blockDim.x;
        }
        row += BLOCK_SIZE * blockDim.y;
    }
}

void timespec_diff(struct timespec *start, struct timespec *stop,
                   struct timespec *result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
}

int main(int argc, char **argv) {
    int N = 1024;
    int THREAD_SIZE = 16;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int cudaErrorCode;

    struct timespec ts1, ts2, ts3;

    if (argc > 1) {
        N = atoi(argv[1]);
        if (argc > 2) {
            THREAD_SIZE = atoi(argv[2]);
        }
    }

    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices A and B here

    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    
    cudaDeviceSynchronize();
    if (cudaGetLastError()) {
        printf("cudaMalloc failed\n");
    }
    
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    if (cudaGetLastError()) {
        printf("cudaMemcpy failed\n");
    }

    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    //  blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);
    dim3 blocksPerGrid(BLOCK_SIZE, BLOCK_SIZE);

    //innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //cudaDeviceSynchronize();

    printf("Now starting kernel...\n");
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    
    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    timespec_diff(&ts1, &ts2, &ts3);
    printf("Kernel done. Time spent: %ld.%06ld s\n", ts3.tv_sec, ts3.tv_nsec / 1000);

    cudaErrorCode = cudaGetLastError();
    if (cudaErrorCode) {
        printf("kernel invocation failed with %d\n", cudaErrorCode);
    }

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}