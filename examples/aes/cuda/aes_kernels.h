#ifndef AES_KERNELS_H
#define AES_KERNELS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Textbook AES-128 ECB
float bench_aes128_ecb_encrypt_textbook(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks);

float bench_aes128_ecb_decrypt_textbook(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks);

// T-table AES-128 ECB
float bench_aes128_ecb_encrypt_ttable(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks);

float bench_aes128_ecb_decrypt_ttable(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks);

// Correctness test wrappers (allocate GPU memory, copy data, run kernel, copy back)
void aes128_ecb_encrypt_textbook_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks);

void aes128_ecb_decrypt_textbook_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks);

void aes128_ecb_encrypt_ttable_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks);

void aes128_ecb_decrypt_ttable_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks);

#ifdef __cplusplus
}
#endif

#endif // AES_KERNELS_H
