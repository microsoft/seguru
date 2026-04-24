#ifndef AES_KERNELS_H
#define AES_KERNELS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Benchmark functions: allocate device memory, run warmup + 100 iterations,
// return average kernel time in microseconds.
// round_keys: 44 uint32_t from host AES-128 key expansion.
float bench_aes128_encrypt_ttable(
    const uint8_t* host_input, uint8_t* host_output,
    const uint32_t* host_round_keys, uint32_t num_blocks,
    int warmup, int iters);

float bench_aes128_decrypt_ttable(
    const uint8_t* host_input, uint8_t* host_output,
    const uint32_t* host_round_keys, uint32_t num_blocks,
    int warmup, int iters);

float bench_aes128_encrypt_textbook(
    const uint8_t* host_input, uint8_t* host_output,
    const uint32_t* host_round_keys, uint32_t num_blocks,
    int warmup, int iters);

float bench_aes128_decrypt_textbook(
    const uint8_t* host_input, uint8_t* host_output,
    const uint32_t* host_round_keys, uint32_t num_blocks,
    int warmup, int iters);

#ifdef __cplusplus
}
#endif

#endif // AES_KERNELS_H
