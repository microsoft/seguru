#include <cstdint>
#include <cuda_runtime.h>

// Barrett multiplication (matches modular.rs)
__device__ __forceinline__ uint64_t barrett_mul(uint64_t a, uint64_t b,
    uint64_t mod_val, uint64_t bit, uint64_t mu)
{
    unsigned __int128 z = (unsigned __int128)a * b;
    unsigned __int128 w = z >> (bit - 2);
    w = (w * (unsigned __int128)mu) >> (bit + 3);
    w = w * (unsigned __int128)mod_val;
    uint64_t r = (uint64_t)(z - w);
    if (r >= mod_val) r -= mod_val;
    return r;
}

// Addition kernel
__global__ void cuda_addition_kernel(
    const uint64_t* in1, const uint64_t* in2, uint64_t* output,
    const uint64_t* mod_values, uint32_t n_power, uint32_t rns_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = (gid >> n_power) % rns_count;
    uint64_t mod_val = mod_values[idy];
    uint64_t sum = in1[gid] + in2[gid];
    output[gid] = (sum >= mod_val) ? sum - mod_val : sum;
}

// Barrett multiply kernel
__global__ void cuda_multiply_kernel(
    const uint64_t* in1, const uint64_t* in2, uint64_t* output,
    const uint64_t* mod_values, const uint64_t* mod_bits, const uint64_t* mod_mus,
    uint32_t n_power, uint32_t rns_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = (gid >> n_power) % rns_count;
    output[gid] = barrett_mul(in1[gid], in2[gid],
        mod_values[idy], mod_bits[idy], mod_mus[idy]);
}

// SK multiplication kernel
__global__ void cuda_sk_multiply_kernel(
    const uint64_t* ct1, const uint64_t* sk, uint64_t* output,
    const uint64_t* mod_values, const uint64_t* mod_bits, const uint64_t* mod_mus,
    uint32_t n_power, uint32_t rns_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = (gid >> n_power) % rns_count;
    output[gid] = barrett_mul(ct1[gid], sk[gid],
        mod_values[idy], mod_bits[idy], mod_mus[idy]);
}

// Cipher-plain multiply kernel
__global__ void cuda_cipher_plain_mul_kernel(
    const uint64_t* cipher, const uint64_t* plain, uint64_t* output,
    const uint64_t* mod_values, const uint64_t* mod_bits, const uint64_t* mod_mus,
    uint32_t n_power, uint32_t rns_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pt_size = rns_count << n_power;
    uint32_t pt_idx = gid % pt_size;
    uint32_t idy = (pt_idx >> n_power) % rns_count;
    output[gid] = barrett_mul(cipher[gid], plain[pt_idx],
        mod_values[idy], mod_bits[idy], mod_mus[idy]);
}

// C-linkage wrapper functions that handle launch + timing
extern "C" {

double cuda_bench_addition(
    const uint64_t* d_in1, const uint64_t* d_in2, uint64_t* d_out,
    const uint64_t* d_mod_values,
    uint32_t total_elements, uint32_t n_power, uint32_t rns_count,
    uint32_t block_size, int iters)
{
    uint32_t grid = (total_elements + block_size - 1) / block_size;

    for (int i = 0; i < 5; i++) {
        cuda_addition_kernel<<<grid, block_size>>>(
            d_in1, d_in2, d_out, d_mod_values, n_power, rns_count);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cuda_addition_kernel<<<grid, block_size>>>(
            d_in1, d_in2, d_out, d_mod_values, n_power, rns_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms * 1000.0 / iters;
}

double cuda_bench_multiply(
    const uint64_t* d_in1, const uint64_t* d_in2, uint64_t* d_out,
    const uint64_t* d_mod_values, const uint64_t* d_mod_bits, const uint64_t* d_mod_mus,
    uint32_t total_elements, uint32_t n_power, uint32_t rns_count,
    uint32_t block_size, int iters)
{
    uint32_t grid = (total_elements + block_size - 1) / block_size;
    for (int i = 0; i < 5; i++) {
        cuda_multiply_kernel<<<grid, block_size>>>(
            d_in1, d_in2, d_out, d_mod_values, d_mod_bits, d_mod_mus, n_power, rns_count);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cuda_multiply_kernel<<<grid, block_size>>>(
            d_in1, d_in2, d_out, d_mod_values, d_mod_bits, d_mod_mus, n_power, rns_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms * 1000.0 / iters;
}

double cuda_bench_sk_multiply(
    const uint64_t* d_ct1, const uint64_t* d_sk, uint64_t* d_out,
    const uint64_t* d_mod_values, const uint64_t* d_mod_bits, const uint64_t* d_mod_mus,
    uint32_t total_elements, uint32_t n_power, uint32_t rns_count,
    uint32_t block_size, int iters)
{
    uint32_t grid = (total_elements + block_size - 1) / block_size;
    for (int i = 0; i < 5; i++) {
        cuda_sk_multiply_kernel<<<grid, block_size>>>(
            d_ct1, d_sk, d_out, d_mod_values, d_mod_bits, d_mod_mus, n_power, rns_count);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cuda_sk_multiply_kernel<<<grid, block_size>>>(
            d_ct1, d_sk, d_out, d_mod_values, d_mod_bits, d_mod_mus, n_power, rns_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms * 1000.0 / iters;
}

double cuda_bench_cipher_plain_mul(
    const uint64_t* d_cipher, const uint64_t* d_plain, uint64_t* d_out,
    const uint64_t* d_mod_values, const uint64_t* d_mod_bits, const uint64_t* d_mod_mus,
    uint32_t total_elements, uint32_t n_power, uint32_t rns_count,
    uint32_t block_size, int iters)
{
    uint32_t grid = (total_elements + block_size - 1) / block_size;
    for (int i = 0; i < 5; i++) {
        cuda_cipher_plain_mul_kernel<<<grid, block_size>>>(
            d_cipher, d_plain, d_out, d_mod_values, d_mod_bits, d_mod_mus,
            n_power, rns_count);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cuda_cipher_plain_mul_kernel<<<grid, block_size>>>(
            d_cipher, d_plain, d_out, d_mod_values, d_mod_bits, d_mod_mus,
            n_power, rns_count);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms * 1000.0 / iters;
}

void cuda_malloc(void** ptr, size_t size) { cudaMalloc(ptr, size); }
void cuda_free(void* ptr) { cudaFree(ptr); }
void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
void cuda_device_sync() { cudaDeviceSynchronize(); }

} // extern "C"
