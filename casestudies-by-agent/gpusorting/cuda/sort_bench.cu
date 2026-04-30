// Thin FFI wrapper around GPUSortingCUDA's DeviceRadixSort for benchmarking.
#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "../cuda-ref/GPUSortingCUDA/Sort/DeviceRadixSort.cuh"

extern "C" {

// Sort `size` uint32_t keys, returning average elapsed milliseconds per sort.
// h_keys: HOST pointer (input, copied to device each iteration).
// h_out:  HOST pointer (output, receives sorted result from last iteration).
float cuda_bench_sort(
    const uint32_t* h_keys,
    uint32_t*       h_out,
    uint32_t        size,
    uint32_t        warmup,
    uint32_t        iters)
{
    const uint32_t partitionSize = 7680;
    const uint32_t threadblocks = (size + partitionSize - 1) / partitionSize;

    uint32_t *d_sort, *d_alt, *d_globalHist, *d_passHist;
    cudaMalloc(&d_sort, size * sizeof(uint32_t));
    cudaMalloc(&d_alt,  size * sizeof(uint32_t));
    cudaMalloc(&d_globalHist, 256 * 4 * sizeof(uint32_t));
    cudaMalloc(&d_passHist,   threadblocks * 256 * sizeof(uint32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Warmup ---
    for (uint32_t w = 0; w < warmup; w++) {
        cudaMemcpy(d_sort, h_keys, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_globalHist, 0, 256 * 4 * sizeof(uint32_t));
        cudaDeviceSynchronize();
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            uint32_t *src = (shift / 8) % 2 == 0 ? d_sort : d_alt;
            uint32_t *dst = (shift / 8) % 2 == 0 ? d_alt  : d_sort;
            DeviceRadixSort::Upsweep<<<threadblocks, 128>>>(src, d_globalHist, d_passHist, size, shift);
            DeviceRadixSort::Scan<<<256, 128>>>(d_passHist, threadblocks);
            DeviceRadixSort::DownsweepKeysOnly<<<threadblocks, 512>>>(src, dst, d_globalHist, d_passHist, size, shift);
        }
        cudaDeviceSynchronize();
    }

    // --- Timed runs ---
    float totalMs = 0.0f;
    for (uint32_t i = 0; i < iters; i++) {
        cudaMemcpy(d_sort, h_keys, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_globalHist, 0, 256 * 4 * sizeof(uint32_t));
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            uint32_t *src = (shift / 8) % 2 == 0 ? d_sort : d_alt;
            uint32_t *dst = (shift / 8) % 2 == 0 ? d_alt  : d_sort;
            DeviceRadixSort::Upsweep<<<threadblocks, 128>>>(src, d_globalHist, d_passHist, size, shift);
            DeviceRadixSort::Scan<<<256, 128>>>(d_passHist, threadblocks);
            DeviceRadixSort::DownsweepKeysOnly<<<threadblocks, 512>>>(src, dst, d_globalHist, d_passHist, size, shift);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;
    }

    // Copy result back (after 4 passes, result is in d_sort)
    cudaMemcpy(h_out, d_sort, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sort);
    cudaFree(d_alt);
    cudaFree(d_globalHist);
    cudaFree(d_passHist);

    return totalMs / (float)iters;
}

} // extern "C"
