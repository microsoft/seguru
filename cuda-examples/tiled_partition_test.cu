#include <cstdio>
#include <ctime>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void innerProductKernel() {
    printf("Enter\n");
    cg::thread_block block = cg::this_thread_block();
    printf("Done thread_block\n");
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    printf("Block: (%d, %d, %d), Thread: (%d, %d, %d), Warp ID: %d, Thread Rank: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, warp.meta_group_rank(), warp.thread_rank());
    //printf("Done Thread: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    int cudaErrorCode;

    dim3 threadsPerBlock(4, 4, 4);
    dim3 blocksPerGrid(2, 2, 2);

    innerProductKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    cudaErrorCode = cudaGetLastError();
    if (cudaErrorCode) {
        printf("kernel invocation failed with %d\n", cudaErrorCode);
    }

    return 0;
}