#include "aes_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================
// AES-128 S-box (FIPS 197)
// ============================================================
__constant__ uint8_t d_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ uint8_t d_inv_sbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

// ============================================================
// Forward T-tables (precomputed SubBytes+ShiftRows+MixColumns)
// ============================================================
#define XTIME(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))

// Te0[s] = [2s, s, s, 3s] big-endian
#define TE0_ENTRY(s) ( (XTIME(s) << 24) | ((s) << 16) | ((s) << 8) | (XTIME(s) ^ (s)) )
// Te1 = ror(Te0, 8), Te2 = ror(Te0, 16), Te3 = ror(Te0, 24)

// We'll generate them at compile time using a helper

// Use __constant__ memory for T-tables (read-only, cached)
// These are initialized from host before first use
__constant__ uint32_t d_te0[256];
__constant__ uint32_t d_te1[256];
__constant__ uint32_t d_te2[256];
__constant__ uint32_t d_te3[256];

__constant__ uint32_t d_td0[256];
__constant__ uint32_t d_td1[256];
__constant__ uint32_t d_td2[256];
__constant__ uint32_t d_td3[256];

// Host-side T-table generation
static uint8_t h_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static uint8_t h_inv_sbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

static uint8_t xtime(uint8_t x) {
    return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
}

static uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) r ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= 0x1b;
        b >>= 1;
    }
    return r;
}

static bool tables_initialized = false;

static void init_tables() {
    if (tables_initialized) return;
    tables_initialized = true;

    uint32_t h_te0[256], h_te1[256], h_te2[256], h_te3[256];
    uint32_t h_td0[256], h_td1[256], h_td2[256], h_td3[256];

    for (int i = 0; i < 256; i++) {
        uint8_t s = h_sbox[i];
        uint8_t s2 = xtime(s);
        uint8_t s3 = s2 ^ s;
        uint32_t te0 = ((uint32_t)s2 << 24) | ((uint32_t)s << 16) | ((uint32_t)s << 8) | (uint32_t)s3;
        h_te0[i] = te0;
        h_te1[i] = (te0 >> 8) | (te0 << 24);
        h_te2[i] = (te0 >> 16) | (te0 << 16);
        h_te3[i] = (te0 >> 24) | (te0 << 8);

        uint8_t si = h_inv_sbox[i];
        uint8_t se = gmul(si, 0x0e);
        uint8_t s9 = gmul(si, 0x09);
        uint8_t sd = gmul(si, 0x0d);
        uint8_t sb = gmul(si, 0x0b);
        uint32_t td0 = ((uint32_t)se << 24) | ((uint32_t)s9 << 16) | ((uint32_t)sd << 8) | (uint32_t)sb;
        h_td0[i] = td0;
        h_td1[i] = (td0 >> 8) | (td0 << 24);
        h_td2[i] = (td0 >> 16) | (td0 << 16);
        h_td3[i] = (td0 >> 24) | (td0 << 8);
    }

    cudaMemcpyToSymbol(d_te0, h_te0, sizeof(h_te0));
    cudaMemcpyToSymbol(d_te1, h_te1, sizeof(h_te1));
    cudaMemcpyToSymbol(d_te2, h_te2, sizeof(h_te2));
    cudaMemcpyToSymbol(d_te3, h_te3, sizeof(h_te3));
    cudaMemcpyToSymbol(d_td0, h_td0, sizeof(h_td0));
    cudaMemcpyToSymbol(d_td1, h_td1, sizeof(h_td1));
    cudaMemcpyToSymbol(d_td2, h_td2, sizeof(h_td2));
    cudaMemcpyToSymbol(d_td3, h_td3, sizeof(h_td3));
}

// ============================================================
// Helper: load 16 bytes as 4 big-endian uint32
// ============================================================
// Device gmul for textbook decrypt
__device__ uint8_t gmul_dev(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) r ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= 0x1b;
        b >>= 1;
    }
    return r;
}

__device__ __forceinline__ void load_block(const uint8_t* p, uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3) {
    s0 = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
    s1 = ((uint32_t)p[4] << 24) | ((uint32_t)p[5] << 16) | ((uint32_t)p[6] << 8) | p[7];
    s2 = ((uint32_t)p[8] << 24) | ((uint32_t)p[9] << 16) | ((uint32_t)p[10] << 8) | p[11];
    s3 = ((uint32_t)p[12] << 24) | ((uint32_t)p[13] << 16) | ((uint32_t)p[14] << 8) | p[15];
}

__device__ __forceinline__ void store_block(uint8_t* p, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    p[0] = s0 >> 24; p[1] = (s0 >> 16) & 0xff; p[2] = (s0 >> 8) & 0xff; p[3] = s0 & 0xff;
    p[4] = s1 >> 24; p[5] = (s1 >> 16) & 0xff; p[6] = (s1 >> 8) & 0xff; p[7] = s1 & 0xff;
    p[8] = s2 >> 24; p[9] = (s2 >> 16) & 0xff; p[10] = (s2 >> 8) & 0xff; p[11] = s2 & 0xff;
    p[12] = s3 >> 24; p[13] = (s3 >> 16) & 0xff; p[14] = (s3 >> 8) & 0xff; p[15] = s3 & 0xff;
}

// ============================================================
// Textbook AES-128 ECB encrypt kernel
// ============================================================
__global__ void aes128_ecb_encrypt_textbook_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ round_keys,
    uint32_t num_blocks)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    // Load S-box into shared memory
    __shared__ uint8_t s_sbox[256];
    if (threadIdx.x < 256) {
        s_sbox[threadIdx.x] = d_sbox[threadIdx.x];
    }
    __syncthreads();

    const uint8_t* in_block = input + tid * 16;
    uint8_t* out_block = output + tid * 16;

    uint32_t s0, s1, s2, s3;
    load_block(in_block, s0, s1, s2, s3);

    // Round 0: AddRoundKey
    s0 ^= round_keys[0];
    s1 ^= round_keys[1];
    s2 ^= round_keys[2];
    s3 ^= round_keys[3];

    // Rounds 1-9: SubBytes + ShiftRows + MixColumns + AddRoundKey
    for (int r = 1; r < 10; r++) {
        // SubBytes + ShiftRows: extract bytes in shifted order
        uint8_t b00 = s_sbox[(s0 >> 24) & 0xff];
        uint8_t b01 = s_sbox[(s1 >> 16) & 0xff];
        uint8_t b02 = s_sbox[(s2 >> 8) & 0xff];
        uint8_t b03 = s_sbox[s3 & 0xff];

        uint8_t b10 = s_sbox[(s1 >> 24) & 0xff];
        uint8_t b11 = s_sbox[(s2 >> 16) & 0xff];
        uint8_t b12 = s_sbox[(s3 >> 8) & 0xff];
        uint8_t b13 = s_sbox[s0 & 0xff];

        uint8_t b20 = s_sbox[(s2 >> 24) & 0xff];
        uint8_t b21 = s_sbox[(s3 >> 16) & 0xff];
        uint8_t b22 = s_sbox[(s0 >> 8) & 0xff];
        uint8_t b23 = s_sbox[s1 & 0xff];

        uint8_t b30 = s_sbox[(s3 >> 24) & 0xff];
        uint8_t b31 = s_sbox[(s0 >> 16) & 0xff];
        uint8_t b32 = s_sbox[(s1 >> 8) & 0xff];
        uint8_t b33 = s_sbox[s2 & 0xff];

        // MixColumns
        #define MC(a,b,c,d) (XTIME(a) ^ (XTIME(b) ^ (b)) ^ (c) ^ (d))
        s0 = ((uint32_t)MC(b00,b01,b02,b03) << 24) |
             ((uint32_t)MC(b03,b00,b01,b02) << 16) |
             ((uint32_t)MC(b02,b03,b00,b01) << 8) |
             (uint32_t)MC(b01,b02,b03,b00);
        s1 = ((uint32_t)MC(b10,b11,b12,b13) << 24) |
             ((uint32_t)MC(b13,b10,b11,b12) << 16) |
             ((uint32_t)MC(b12,b13,b10,b11) << 8) |
             (uint32_t)MC(b11,b12,b13,b10);
        s2 = ((uint32_t)MC(b20,b21,b22,b23) << 24) |
             ((uint32_t)MC(b23,b20,b21,b22) << 16) |
             ((uint32_t)MC(b22,b23,b20,b21) << 8) |
             (uint32_t)MC(b21,b22,b23,b20);
        s3 = ((uint32_t)MC(b30,b31,b32,b33) << 24) |
             ((uint32_t)MC(b33,b30,b31,b32) << 16) |
             ((uint32_t)MC(b32,b33,b30,b31) << 8) |
             (uint32_t)MC(b31,b32,b33,b30);
        #undef MC

        // AddRoundKey
        s0 ^= round_keys[4*r];
        s1 ^= round_keys[4*r+1];
        s2 ^= round_keys[4*r+2];
        s3 ^= round_keys[4*r+3];
    }

    // Round 10: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    uint32_t t0 = ((uint32_t)s_sbox[(s0>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s1>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s2>>8)&0xff] << 8)  | (uint32_t)s_sbox[s3&0xff];
    uint32_t t1 = ((uint32_t)s_sbox[(s1>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s2>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s3>>8)&0xff] << 8)  | (uint32_t)s_sbox[s0&0xff];
    uint32_t t2 = ((uint32_t)s_sbox[(s2>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s3>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s0>>8)&0xff] << 8)  | (uint32_t)s_sbox[s1&0xff];
    uint32_t t3 = ((uint32_t)s_sbox[(s3>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s0>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s1>>8)&0xff] << 8)  | (uint32_t)s_sbox[s2&0xff];

    store_block(out_block, t0 ^ round_keys[40], t1 ^ round_keys[41], t2 ^ round_keys[42], t3 ^ round_keys[43]);
}

// ============================================================
// Textbook AES-128 ECB decrypt kernel
// ============================================================
__global__ void aes128_ecb_decrypt_textbook_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ inv_round_keys,
    uint32_t num_blocks)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    __shared__ uint8_t s_inv_sbox[256];
    if (threadIdx.x < 256) {
        s_inv_sbox[threadIdx.x] = d_inv_sbox[threadIdx.x];
    }
    __syncthreads();

    const uint8_t* in_block = input + tid * 16;
    uint8_t* out_block = output + tid * 16;

    uint32_t s0, s1, s2, s3;
    load_block(in_block, s0, s1, s2, s3);

    // AddRoundKey (round 10)
    s0 ^= inv_round_keys[40];
    s1 ^= inv_round_keys[41];
    s2 ^= inv_round_keys[42];
    s3 ^= inv_round_keys[43];

    // Rounds 9..1: InvShiftRows + InvSubBytes + AddRoundKey + InvMixColumns
    for (int r = 9; r >= 1; r--) {
        // InvShiftRows + InvSubBytes
        uint8_t b00 = s_inv_sbox[(s0 >> 24) & 0xff];
        uint8_t b01 = s_inv_sbox[(s3 >> 16) & 0xff];
        uint8_t b02 = s_inv_sbox[(s2 >> 8) & 0xff];
        uint8_t b03 = s_inv_sbox[s1 & 0xff];

        uint8_t b10 = s_inv_sbox[(s1 >> 24) & 0xff];
        uint8_t b11 = s_inv_sbox[(s0 >> 16) & 0xff];
        uint8_t b12 = s_inv_sbox[(s3 >> 8) & 0xff];
        uint8_t b13 = s_inv_sbox[s2 & 0xff];

        uint8_t b20 = s_inv_sbox[(s2 >> 24) & 0xff];
        uint8_t b21 = s_inv_sbox[(s1 >> 16) & 0xff];
        uint8_t b22 = s_inv_sbox[(s0 >> 8) & 0xff];
        uint8_t b23 = s_inv_sbox[s3 & 0xff];

        uint8_t b30 = s_inv_sbox[(s3 >> 24) & 0xff];
        uint8_t b31 = s_inv_sbox[(s2 >> 16) & 0xff];
        uint8_t b32 = s_inv_sbox[(s1 >> 8) & 0xff];
        uint8_t b33 = s_inv_sbox[s0 & 0xff];

        // AddRoundKey (before InvMixColumns, using equivalent decryption round keys)
        uint32_t a0 = ((uint32_t)b00 << 24) | ((uint32_t)b01 << 16) | ((uint32_t)b02 << 8) | b03;
        uint32_t a1 = ((uint32_t)b10 << 24) | ((uint32_t)b11 << 16) | ((uint32_t)b12 << 8) | b13;
        uint32_t a2 = ((uint32_t)b20 << 24) | ((uint32_t)b21 << 16) | ((uint32_t)b22 << 8) | b23;
        uint32_t a3 = ((uint32_t)b30 << 24) | ((uint32_t)b31 << 16) | ((uint32_t)b32 << 8) | b33;

        a0 ^= inv_round_keys[4*r];
        a1 ^= inv_round_keys[4*r+1];
        a2 ^= inv_round_keys[4*r+2];
        a3 ^= inv_round_keys[4*r+3];

        // InvMixColumns
        #define IMC_E(a) gmul_dev(a, 0x0e)
        #define IMC_B(a) gmul_dev(a, 0x0b)
        #define IMC_D(a) gmul_dev(a, 0x0d)
        #define IMC_9(a) gmul_dev(a, 0x09)
        // Actually, let's use a simpler approach for the textbook version
        // We already have inv_round_keys with InvMixColumns applied, so we use
        // the equivalent decryption schedule. But for textbook, let's do it directly.
        // For textbook decrypt, use the standard approach:
        // s = InvMixColumns(a) where a = InvSubBytes(InvShiftRows(state)) ^ round_key
        // Actually for equivalent inverse cipher, we apply InvMixColumns after AddRoundKey.

        // InvMixColumns on a0..a3
        uint8_t c0 = (a0 >> 24) & 0xff, c1 = (a0 >> 16) & 0xff, c2 = (a0 >> 8) & 0xff, c3 = a0 & 0xff;
        s0 = ((uint32_t)(gmul_dev(c0,0x0e)^gmul_dev(c1,0x0b)^gmul_dev(c2,0x0d)^gmul_dev(c3,0x09)) << 24) |
             ((uint32_t)(gmul_dev(c0,0x09)^gmul_dev(c1,0x0e)^gmul_dev(c2,0x0b)^gmul_dev(c3,0x0d)) << 16) |
             ((uint32_t)(gmul_dev(c0,0x0d)^gmul_dev(c1,0x09)^gmul_dev(c2,0x0e)^gmul_dev(c3,0x0b)) << 8) |
             (uint32_t)(gmul_dev(c0,0x0b)^gmul_dev(c1,0x0d)^gmul_dev(c2,0x09)^gmul_dev(c3,0x0e));

        c0 = (a1 >> 24) & 0xff; c1 = (a1 >> 16) & 0xff; c2 = (a1 >> 8) & 0xff; c3 = a1 & 0xff;
        s1 = ((uint32_t)(gmul_dev(c0,0x0e)^gmul_dev(c1,0x0b)^gmul_dev(c2,0x0d)^gmul_dev(c3,0x09)) << 24) |
             ((uint32_t)(gmul_dev(c0,0x09)^gmul_dev(c1,0x0e)^gmul_dev(c2,0x0b)^gmul_dev(c3,0x0d)) << 16) |
             ((uint32_t)(gmul_dev(c0,0x0d)^gmul_dev(c1,0x09)^gmul_dev(c2,0x0e)^gmul_dev(c3,0x0b)) << 8) |
             (uint32_t)(gmul_dev(c0,0x0b)^gmul_dev(c1,0x0d)^gmul_dev(c2,0x09)^gmul_dev(c3,0x0e));

        c0 = (a2 >> 24) & 0xff; c1 = (a2 >> 16) & 0xff; c2 = (a2 >> 8) & 0xff; c3 = a2 & 0xff;
        s2 = ((uint32_t)(gmul_dev(c0,0x0e)^gmul_dev(c1,0x0b)^gmul_dev(c2,0x0d)^gmul_dev(c3,0x09)) << 24) |
             ((uint32_t)(gmul_dev(c0,0x09)^gmul_dev(c1,0x0e)^gmul_dev(c2,0x0b)^gmul_dev(c3,0x0d)) << 16) |
             ((uint32_t)(gmul_dev(c0,0x0d)^gmul_dev(c1,0x09)^gmul_dev(c2,0x0e)^gmul_dev(c3,0x0b)) << 8) |
             (uint32_t)(gmul_dev(c0,0x0b)^gmul_dev(c1,0x0d)^gmul_dev(c2,0x09)^gmul_dev(c3,0x0e));

        c0 = (a3 >> 24) & 0xff; c1 = (a3 >> 16) & 0xff; c2 = (a3 >> 8) & 0xff; c3 = a3 & 0xff;
        s3 = ((uint32_t)(gmul_dev(c0,0x0e)^gmul_dev(c1,0x0b)^gmul_dev(c2,0x0d)^gmul_dev(c3,0x09)) << 24) |
             ((uint32_t)(gmul_dev(c0,0x09)^gmul_dev(c1,0x0e)^gmul_dev(c2,0x0b)^gmul_dev(c3,0x0d)) << 16) |
             ((uint32_t)(gmul_dev(c0,0x0d)^gmul_dev(c1,0x09)^gmul_dev(c2,0x0e)^gmul_dev(c3,0x0b)) << 8) |
             (uint32_t)(gmul_dev(c0,0x0b)^gmul_dev(c1,0x0d)^gmul_dev(c2,0x09)^gmul_dev(c3,0x0e));

        #undef IMC_E
        #undef IMC_B
        #undef IMC_D
        #undef IMC_9
    }

    // Round 0: InvShiftRows + InvSubBytes + AddRoundKey (no InvMixColumns)
    uint32_t t0 = ((uint32_t)s_inv_sbox[(s0>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s3>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s2>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s1&0xff];
    uint32_t t1 = ((uint32_t)s_inv_sbox[(s1>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s0>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s3>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s2&0xff];
    uint32_t t2 = ((uint32_t)s_inv_sbox[(s2>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s1>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s0>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s3&0xff];
    uint32_t t3 = ((uint32_t)s_inv_sbox[(s3>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s2>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s1>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s0&0xff];

    store_block(out_block, t0 ^ inv_round_keys[0], t1 ^ inv_round_keys[1],
                t2 ^ inv_round_keys[2], t3 ^ inv_round_keys[3]);
}

// ============================================================
// T-table AES-128 ECB encrypt kernel
// ============================================================
__global__ void aes128_ecb_encrypt_ttable_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ round_keys,
    uint32_t num_blocks)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    // Load T-tables into shared memory (4KB total)
    __shared__ uint32_t s_te0[256], s_te1[256], s_te2[256], s_te3[256];
    __shared__ uint8_t s_sbox[256];
    // Cooperative load: 256 threads load 256 entries each
    if (threadIdx.x < 256) {
        s_te0[threadIdx.x] = d_te0[threadIdx.x];
        s_te1[threadIdx.x] = d_te1[threadIdx.x];
        s_te2[threadIdx.x] = d_te2[threadIdx.x];
        s_te3[threadIdx.x] = d_te3[threadIdx.x];
        s_sbox[threadIdx.x] = d_sbox[threadIdx.x];
    }
    __syncthreads();

    const uint8_t* in_block = input + tid * 16;
    uint8_t* out_block = output + tid * 16;

    uint32_t s0, s1, s2, s3;
    load_block(in_block, s0, s1, s2, s3);

    s0 ^= round_keys[0];
    s1 ^= round_keys[1];
    s2 ^= round_keys[2];
    s3 ^= round_keys[3];

    // Rounds 1-9: T-table lookups
    for (int r = 1; r < 10; r++) {
        uint32_t t0 = s_te0[(s0>>24)&0xff] ^ s_te1[(s1>>16)&0xff] ^ s_te2[(s2>>8)&0xff] ^ s_te3[s3&0xff] ^ round_keys[4*r];
        uint32_t t1 = s_te0[(s1>>24)&0xff] ^ s_te1[(s2>>16)&0xff] ^ s_te2[(s3>>8)&0xff] ^ s_te3[s0&0xff] ^ round_keys[4*r+1];
        uint32_t t2 = s_te0[(s2>>24)&0xff] ^ s_te1[(s3>>16)&0xff] ^ s_te2[(s0>>8)&0xff] ^ s_te3[s1&0xff] ^ round_keys[4*r+2];
        uint32_t t3 = s_te0[(s3>>24)&0xff] ^ s_te1[(s0>>16)&0xff] ^ s_te2[(s1>>8)&0xff] ^ s_te3[s2&0xff] ^ round_keys[4*r+3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Round 10: S-box only (no MixColumns)
    uint32_t t0 = ((uint32_t)s_sbox[(s0>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s1>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s2>>8)&0xff] << 8)  | (uint32_t)s_sbox[s3&0xff];
    uint32_t t1 = ((uint32_t)s_sbox[(s1>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s2>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s3>>8)&0xff] << 8)  | (uint32_t)s_sbox[s0&0xff];
    uint32_t t2 = ((uint32_t)s_sbox[(s2>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s3>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s0>>8)&0xff] << 8)  | (uint32_t)s_sbox[s1&0xff];
    uint32_t t3 = ((uint32_t)s_sbox[(s3>>24)&0xff] << 24) | ((uint32_t)s_sbox[(s0>>16)&0xff] << 16) |
                  ((uint32_t)s_sbox[(s1>>8)&0xff] << 8)  | (uint32_t)s_sbox[s2&0xff];

    store_block(out_block, t0 ^ round_keys[40], t1 ^ round_keys[41], t2 ^ round_keys[42], t3 ^ round_keys[43]);
}

// ============================================================
// T-table AES-128 ECB decrypt kernel
// ============================================================
__global__ void aes128_ecb_decrypt_ttable_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ inv_round_keys,
    uint32_t num_blocks)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    __shared__ uint32_t s_td0[256], s_td1[256], s_td2[256], s_td3[256];
    __shared__ uint8_t s_inv_sbox[256];
    if (threadIdx.x < 256) {
        s_td0[threadIdx.x] = d_td0[threadIdx.x];
        s_td1[threadIdx.x] = d_td1[threadIdx.x];
        s_td2[threadIdx.x] = d_td2[threadIdx.x];
        s_td3[threadIdx.x] = d_td3[threadIdx.x];
        s_inv_sbox[threadIdx.x] = d_inv_sbox[threadIdx.x];
    }
    __syncthreads();

    const uint8_t* in_block = input + tid * 16;
    uint8_t* out_block = output + tid * 16;

    uint32_t s0, s1, s2, s3;
    load_block(in_block, s0, s1, s2, s3);

    // AddRoundKey (round 10)
    s0 ^= inv_round_keys[40];
    s1 ^= inv_round_keys[41];
    s2 ^= inv_round_keys[42];
    s3 ^= inv_round_keys[43];

    // Rounds 9..1: Inverse T-table lookups
    for (int r = 9; r >= 1; r--) {
        uint32_t t0 = s_td0[(s0>>24)&0xff] ^ s_td1[(s3>>16)&0xff] ^ s_td2[(s2>>8)&0xff] ^ s_td3[s1&0xff] ^ inv_round_keys[4*r];
        uint32_t t1 = s_td0[(s1>>24)&0xff] ^ s_td1[(s0>>16)&0xff] ^ s_td2[(s3>>8)&0xff] ^ s_td3[s2&0xff] ^ inv_round_keys[4*r+1];
        uint32_t t2 = s_td0[(s2>>24)&0xff] ^ s_td1[(s1>>16)&0xff] ^ s_td2[(s0>>8)&0xff] ^ s_td3[s3&0xff] ^ inv_round_keys[4*r+2];
        uint32_t t3 = s_td0[(s3>>24)&0xff] ^ s_td1[(s2>>16)&0xff] ^ s_td2[(s1>>8)&0xff] ^ s_td3[s0&0xff] ^ inv_round_keys[4*r+3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Round 0: InvShiftRows + InvSubBytes + AddRoundKey
    uint32_t t0 = ((uint32_t)s_inv_sbox[(s0>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s3>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s2>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s1&0xff];
    uint32_t t1 = ((uint32_t)s_inv_sbox[(s1>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s0>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s3>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s2&0xff];
    uint32_t t2 = ((uint32_t)s_inv_sbox[(s2>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s1>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s0>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s3&0xff];
    uint32_t t3 = ((uint32_t)s_inv_sbox[(s3>>24)&0xff] << 24) | ((uint32_t)s_inv_sbox[(s2>>16)&0xff] << 16) |
                  ((uint32_t)s_inv_sbox[(s1>>8)&0xff] << 8)  | (uint32_t)s_inv_sbox[s0&0xff];

    store_block(out_block, t0 ^ inv_round_keys[0], t1 ^ inv_round_keys[1],
                t2 ^ inv_round_keys[2], t3 ^ inv_round_keys[3]);
}

// ============================================================
// Host wrapper functions (allocate, copy, run, copy back)
// ============================================================
extern "C" void aes128_ecb_encrypt_textbook_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks)
{
    size_t data_size = (size_t)num_blocks * 16;
    uint8_t *d_in, *d_out;
    uint32_t *d_rk;

    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMalloc(&d_rk, 44 * sizeof(uint32_t));

    cudaMemcpy(d_in, input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, round_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;
    aes128_ecb_encrypt_textbook_kernel<<<grid_size, block_size>>>(d_in, d_out, d_rk, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_rk);
}

extern "C" void aes128_ecb_decrypt_textbook_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks)
{
    size_t data_size = (size_t)num_blocks * 16;
    uint8_t *d_in, *d_out;
    uint32_t *d_rk;

    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMalloc(&d_rk, 44 * sizeof(uint32_t));

    cudaMemcpy(d_in, input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, round_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;
    aes128_ecb_decrypt_textbook_kernel<<<grid_size, block_size>>>(d_in, d_out, d_rk, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_rk);
}

extern "C" void aes128_ecb_encrypt_ttable_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks)
{
    init_tables();
    size_t data_size = (size_t)num_blocks * 16;
    uint8_t *d_in, *d_out;
    uint32_t *d_rk;

    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMalloc(&d_rk, 44 * sizeof(uint32_t));

    cudaMemcpy(d_in, input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, round_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;
    aes128_ecb_encrypt_ttable_kernel<<<grid_size, block_size>>>(d_in, d_out, d_rk, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_rk);
}

extern "C" void aes128_ecb_decrypt_ttable_host(
    const uint8_t* input, uint8_t* output,
    const uint32_t* round_keys, uint32_t num_blocks)
{
    init_tables();
    size_t data_size = (size_t)num_blocks * 16;
    uint8_t *d_in, *d_out;
    uint32_t *d_rk;

    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMalloc(&d_rk, 44 * sizeof(uint32_t));

    cudaMemcpy(d_in, input, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk, round_keys, 44 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;
    aes128_ecb_decrypt_ttable_kernel<<<grid_size, block_size>>>(d_in, d_out, d_rk, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_rk);
}

// ============================================================
// Benchmark wrappers (data already on GPU)
// ============================================================
extern "C" float bench_aes128_ecb_encrypt_textbook(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    aes128_ecb_encrypt_textbook_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        aes128_ecb_encrypt_textbook_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

extern "C" float bench_aes128_ecb_encrypt_ttable(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks)
{
    init_tables();
    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    aes128_ecb_encrypt_ttable_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        aes128_ecb_encrypt_ttable_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

extern "C" float bench_aes128_ecb_decrypt_textbook(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks)
{
    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    aes128_ecb_decrypt_textbook_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        aes128_ecb_decrypt_textbook_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}

extern "C" float bench_aes128_ecb_decrypt_ttable(
    const uint8_t* d_input, uint8_t* d_output,
    const uint32_t* d_round_keys, uint32_t num_blocks)
{
    init_tables();
    uint32_t block_size = 256;
    uint32_t grid_size = (num_blocks + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    aes128_ecb_decrypt_ttable_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        aes128_ecb_decrypt_ttable_kernel<<<grid_size, block_size>>>(d_input, d_output, d_round_keys, num_blocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 100.0f;
}
