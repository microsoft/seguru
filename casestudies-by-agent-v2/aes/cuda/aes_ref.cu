// CUDA C++ reference for the SeGuRu AES-128 case study.
//
// `aes128_encrypt_opt` / `aes128_decrypt_opt` mirror the SeGuRu kernels
// one-for-one (uint4 vector I/O, four AES blocks per thread, a single shared
// TE0/TD0 table plus rotations, round keys staged in shared memory) so the
// SeGuRu-vs-CUDA number isolates code generation rather than algorithm choice.
//
// `aes128_encrypt_classic` is the textbook CUDA formulation (one AES block per
// thread, four T-tables in __constant__ memory) kept as a baseline.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(expr)                                                                       \
  do {                                                                                         \
    cudaError_t _e = (expr);                                                                   \
    if (_e != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(_e), __FILE__,          \
              __LINE__, cudaGetErrorString(_e));                                               \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

#define BLOCK_DIM 256
#define BLOCKS_PER_THREAD 4

__constant__ uint32_t c_te[1024];  // TE0..TE3
__constant__ uint32_t c_rk[44];

__device__ __forceinline__ uint32_t rotr8(uint32_t x) { return (x >> 8) | (x << 24); }
__device__ __forceinline__ uint32_t rotr16(uint32_t x) { return (x >> 16) | (x << 16); }
__device__ __forceinline__ uint32_t rotr24(uint32_t x) { return (x >> 24) | (x << 8); }

__device__ __forceinline__ uint32_t round_col(const uint32_t *t, uint32_t a, uint32_t b,
                                              uint32_t c, uint32_t d, uint32_t k) {
  return t[a] ^ rotr8(t[b]) ^ rotr16(t[c]) ^ rotr24(t[d]) ^ k;
}

__device__ __forceinline__ uint32_t sbox_from_te(const uint32_t *t, uint32_t x) {
  return (t[x] >> 16) & 0xff;
}

// ---------------------------------------------------------------------------
// Optimised kernels (mirror of the SeGuRu implementation)
// ---------------------------------------------------------------------------

__global__ void aes128_encrypt_opt(const uint4 *__restrict__ in, uint4 *__restrict__ out,
                                   const uint32_t *__restrict__ te0,
                                   const uint32_t *__restrict__ rks) {
  __shared__ uint32_t te[BLOCK_DIM];
  __shared__ uint32_t rk[BLOCK_DIM];
  const uint32_t tid = threadIdx.x;
  te[tid] = te0[tid];
  rk[tid] = rks[tid];
  __syncthreads();

  const uint32_t nthreads = gridDim.x * BLOCK_DIM;
  const uint32_t gid = blockIdx.x * BLOCK_DIM + tid;

  uint32_t st[BLOCKS_PER_THREAD][4];
#pragma unroll
  for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
    uint4 v = in[gid + k * nthreads];
    st[k][0] = v.x ^ rk[0];
    st[k][1] = v.y ^ rk[1];
    st[k][2] = v.z ^ rk[2];
    st[k][3] = v.w ^ rk[3];
  }

#pragma unroll
  for (int r = 1; r < 10; ++r) {
#pragma unroll
    for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
      uint32_t s0 = st[k][0], s1 = st[k][1], s2 = st[k][2], s3 = st[k][3];
      st[k][0] = round_col(te, s0 >> 24, (s1 >> 16) & 0xff, (s2 >> 8) & 0xff, s3 & 0xff, rk[4 * r]);
      st[k][1] =
          round_col(te, s1 >> 24, (s2 >> 16) & 0xff, (s3 >> 8) & 0xff, s0 & 0xff, rk[4 * r + 1]);
      st[k][2] =
          round_col(te, s2 >> 24, (s3 >> 16) & 0xff, (s0 >> 8) & 0xff, s1 & 0xff, rk[4 * r + 2]);
      st[k][3] =
          round_col(te, s3 >> 24, (s0 >> 16) & 0xff, (s1 >> 8) & 0xff, s2 & 0xff, rk[4 * r + 3]);
    }
  }

#pragma unroll
  for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
    uint32_t s0 = st[k][0], s1 = st[k][1], s2 = st[k][2], s3 = st[k][3];
    uint4 v;
    v.x = ((sbox_from_te(te, s0 >> 24) << 24) | (sbox_from_te(te, (s1 >> 16) & 0xff) << 16) |
           (sbox_from_te(te, (s2 >> 8) & 0xff) << 8) | sbox_from_te(te, s3 & 0xff)) ^
          rk[40];
    v.y = ((sbox_from_te(te, s1 >> 24) << 24) | (sbox_from_te(te, (s2 >> 16) & 0xff) << 16) |
           (sbox_from_te(te, (s3 >> 8) & 0xff) << 8) | sbox_from_te(te, s0 & 0xff)) ^
          rk[41];
    v.z = ((sbox_from_te(te, s2 >> 24) << 24) | (sbox_from_te(te, (s3 >> 16) & 0xff) << 16) |
           (sbox_from_te(te, (s0 >> 8) & 0xff) << 8) | sbox_from_te(te, s1 & 0xff)) ^
          rk[42];
    v.w = ((sbox_from_te(te, s3 >> 24) << 24) | (sbox_from_te(te, (s0 >> 16) & 0xff) << 16) |
           (sbox_from_te(te, (s1 >> 8) & 0xff) << 8) | sbox_from_te(te, s2 & 0xff)) ^
          rk[43];
    out[gid + k * nthreads] = v;
  }
}

__global__ void aes128_decrypt_opt(const uint4 *__restrict__ in, uint4 *__restrict__ out,
                                   const uint32_t *__restrict__ td0,
                                   const uint32_t *__restrict__ isbox,
                                   const uint32_t *__restrict__ rks) {
  __shared__ uint32_t td[BLOCK_DIM];
  __shared__ uint32_t isb[BLOCK_DIM];
  __shared__ uint32_t rk[BLOCK_DIM];
  const uint32_t tid = threadIdx.x;
  td[tid] = td0[tid];
  isb[tid] = isbox[tid];
  rk[tid] = rks[tid];
  __syncthreads();

  const uint32_t nthreads = gridDim.x * BLOCK_DIM;
  const uint32_t gid = blockIdx.x * BLOCK_DIM + tid;

  uint32_t st[BLOCKS_PER_THREAD][4];
#pragma unroll
  for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
    uint4 v = in[gid + k * nthreads];
    st[k][0] = v.x ^ rk[40];
    st[k][1] = v.y ^ rk[41];
    st[k][2] = v.z ^ rk[42];
    st[k][3] = v.w ^ rk[43];
  }

#pragma unroll
  for (int r = 9; r >= 1; --r) {
#pragma unroll
    for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
      uint32_t s0 = st[k][0], s1 = st[k][1], s2 = st[k][2], s3 = st[k][3];
      st[k][0] = round_col(td, s0 >> 24, (s3 >> 16) & 0xff, (s2 >> 8) & 0xff, s1 & 0xff, rk[4 * r]);
      st[k][1] =
          round_col(td, s1 >> 24, (s0 >> 16) & 0xff, (s3 >> 8) & 0xff, s2 & 0xff, rk[4 * r + 1]);
      st[k][2] =
          round_col(td, s2 >> 24, (s1 >> 16) & 0xff, (s0 >> 8) & 0xff, s3 & 0xff, rk[4 * r + 2]);
      st[k][3] =
          round_col(td, s3 >> 24, (s2 >> 16) & 0xff, (s1 >> 8) & 0xff, s0 & 0xff, rk[4 * r + 3]);
    }
  }

#pragma unroll
  for (int k = 0; k < BLOCKS_PER_THREAD; ++k) {
    uint32_t s0 = st[k][0], s1 = st[k][1], s2 = st[k][2], s3 = st[k][3];
    uint4 v;
    v.x = ((isb[s0 >> 24] << 24) | (isb[(s3 >> 16) & 0xff] << 16) | (isb[(s2 >> 8) & 0xff] << 8) |
           isb[s1 & 0xff]) ^
          rk[0];
    v.y = ((isb[s1 >> 24] << 24) | (isb[(s0 >> 16) & 0xff] << 16) | (isb[(s3 >> 8) & 0xff] << 8) |
           isb[s2 & 0xff]) ^
          rk[1];
    v.z = ((isb[s2 >> 24] << 24) | (isb[(s1 >> 16) & 0xff] << 16) | (isb[(s0 >> 8) & 0xff] << 8) |
           isb[s3 & 0xff]) ^
          rk[2];
    v.w = ((isb[s3 >> 24] << 24) | (isb[(s2 >> 16) & 0xff] << 16) | (isb[(s1 >> 8) & 0xff] << 8) |
           isb[s0 & 0xff]) ^
          rk[3];
    out[gid + k * nthreads] = v;
  }
}

// ---------------------------------------------------------------------------
// Classic baseline: one block per thread, four T-tables in constant memory
// ---------------------------------------------------------------------------

__global__ void aes128_encrypt_classic(const uint32_t *__restrict__ in, uint32_t *__restrict__ out,
                                       uint32_t n_blocks) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_blocks) return;
  const uint32_t *p = in + 4 * tid;
  uint32_t s0 = p[0] ^ c_rk[0];
  uint32_t s1 = p[1] ^ c_rk[1];
  uint32_t s2 = p[2] ^ c_rk[2];
  uint32_t s3 = p[3] ^ c_rk[3];
#pragma unroll
  for (int r = 1; r < 10; ++r) {
    uint32_t t0 = c_te[s0 >> 24] ^ c_te[256 + ((s1 >> 16) & 0xff)] ^
                  c_te[512 + ((s2 >> 8) & 0xff)] ^ c_te[768 + (s3 & 0xff)] ^ c_rk[4 * r];
    uint32_t t1 = c_te[s1 >> 24] ^ c_te[256 + ((s2 >> 16) & 0xff)] ^
                  c_te[512 + ((s3 >> 8) & 0xff)] ^ c_te[768 + (s0 & 0xff)] ^ c_rk[4 * r + 1];
    uint32_t t2 = c_te[s2 >> 24] ^ c_te[256 + ((s3 >> 16) & 0xff)] ^
                  c_te[512 + ((s0 >> 8) & 0xff)] ^ c_te[768 + (s1 & 0xff)] ^ c_rk[4 * r + 2];
    uint32_t t3 = c_te[s3 >> 24] ^ c_te[256 + ((s0 >> 16) & 0xff)] ^
                  c_te[512 + ((s1 >> 8) & 0xff)] ^ c_te[768 + (s2 & 0xff)] ^ c_rk[4 * r + 3];
    s0 = t0;
    s1 = t1;
    s2 = t2;
    s3 = t3;
  }
  uint32_t *q = out + 4 * tid;
  q[0] = ((sbox_from_te(c_te, s0 >> 24) << 24) | (sbox_from_te(c_te, (s1 >> 16) & 0xff) << 16) |
          (sbox_from_te(c_te, (s2 >> 8) & 0xff) << 8) | sbox_from_te(c_te, s3 & 0xff)) ^
         c_rk[40];
  q[1] = ((sbox_from_te(c_te, s1 >> 24) << 24) | (sbox_from_te(c_te, (s2 >> 16) & 0xff) << 16) |
          (sbox_from_te(c_te, (s3 >> 8) & 0xff) << 8) | sbox_from_te(c_te, s0 & 0xff)) ^
         c_rk[41];
  q[2] = ((sbox_from_te(c_te, s2 >> 24) << 24) | (sbox_from_te(c_te, (s3 >> 16) & 0xff) << 16) |
          (sbox_from_te(c_te, (s0 >> 8) & 0xff) << 8) | sbox_from_te(c_te, s1 & 0xff)) ^
         c_rk[42];
  q[3] = ((sbox_from_te(c_te, s3 >> 24) << 24) | (sbox_from_te(c_te, (s0 >> 16) & 0xff) << 16) |
          (sbox_from_te(c_te, (s1 >> 8) & 0xff) << 8) | sbox_from_te(c_te, s2 & 0xff)) ^
         c_rk[43];
}

// ---------------------------------------------------------------------------
// Host-side benchmark entry points. All timings are kernel-only (device
// buffers are allocated and filled once, outside the timed region).
// ---------------------------------------------------------------------------

struct CudaAesCtx {
  uint32_t *d_in;
  uint32_t *d_out;
  uint32_t *d_te0;
  uint32_t *d_td0;
  uint32_t *d_isb;
  uint32_t *d_enc_rk;
  uint32_t *d_dec_rk;
  uint32_t padded_blocks;
};

extern "C" CudaAesCtx *cuda_aes_create(const uint32_t *h_in, uint32_t padded_blocks,
                                       const uint32_t *te0, const uint32_t *td0,
                                       const uint32_t *isb, const uint32_t *enc_rk,
                                       const uint32_t *dec_rk, const uint32_t *te_all) {
  CudaAesCtx *c = new CudaAesCtx();
  c->padded_blocks = padded_blocks;
  size_t bytes = (size_t)padded_blocks * 16;
  CUDA_CHECK(cudaMalloc(&c->d_in, bytes));
  CUDA_CHECK(cudaMalloc(&c->d_out, bytes));
  CUDA_CHECK(cudaMalloc(&c->d_te0, BLOCK_DIM * 4));
  CUDA_CHECK(cudaMalloc(&c->d_td0, BLOCK_DIM * 4));
  CUDA_CHECK(cudaMalloc(&c->d_isb, BLOCK_DIM * 4));
  CUDA_CHECK(cudaMalloc(&c->d_enc_rk, BLOCK_DIM * 4));
  CUDA_CHECK(cudaMalloc(&c->d_dec_rk, BLOCK_DIM * 4));
  CUDA_CHECK(cudaMemcpy(c->d_in, h_in, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->d_te0, te0, BLOCK_DIM * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->d_td0, td0, BLOCK_DIM * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->d_isb, isb, BLOCK_DIM * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->d_enc_rk, enc_rk, BLOCK_DIM * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->d_dec_rk, dec_rk, BLOCK_DIM * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(c_te, te_all, 1024 * 4));
  CUDA_CHECK(cudaMemcpyToSymbol(c_rk, enc_rk, 44 * 4));
  return c;
}

extern "C" void cuda_aes_destroy(CudaAesCtx *c) {
  cudaFree(c->d_in);
  cudaFree(c->d_out);
  cudaFree(c->d_te0);
  cudaFree(c->d_td0);
  cudaFree(c->d_isb);
  cudaFree(c->d_enc_rk);
  cudaFree(c->d_dec_rk);
  delete c;
}

extern "C" void cuda_aes_copy_out(CudaAesCtx *c, uint32_t *h_out) {
  CUDA_CHECK(cudaMemcpy(h_out, c->d_out, (size_t)c->padded_blocks * 16, cudaMemcpyDeviceToHost));
}

// kind: 0 = optimised encrypt, 1 = optimised decrypt, 2 = classic encrypt.
// Returns the mean kernel time in milliseconds.
extern "C" float cuda_aes_bench(CudaAesCtx *c, int kind, int warmup, int iters) {
  uint32_t grid = c->padded_blocks / (BLOCK_DIM * BLOCKS_PER_THREAD);
  uint32_t classic_grid = (c->padded_blocks + BLOCK_DIM - 1) / BLOCK_DIM;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Decryption must be timed on real ciphertext, not on the plaintext buffer.
  // AES T-table indices are the data bytes, so the shared-memory bank-conflict
  // rate depends on the input distribution. The synthetic plaintext used here is
  // structured (byte i is i*31 ^ (i>>8)), which makes consecutive lanes' indices
  // differ by 240 mod 256; since 240 % 32 == 16, a whole warp collides onto two
  // banks in the first round. Real ciphertext is pseudorandom and spreads over
  // all 32 banks. Decrypting the plaintext buffer therefore measures a workload
  // ~12% slower than the real one. Encrypt once and swap so `d_in` holds
  // ciphertext; `d_out` then correctly receives the recovered plaintext.
  if (kind == 1) {
    aes128_encrypt_opt<<<grid, BLOCK_DIM>>>((const uint4 *)c->d_in, (uint4 *)c->d_out, c->d_te0,
                                            c->d_enc_rk);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t *tmp = c->d_in;
    c->d_in = c->d_out;
    c->d_out = tmp;
  }

  for (int i = 0; i < warmup + iters; ++i) {
    if (i == warmup) {
      cudaDeviceSynchronize();
      cudaEventRecord(start);
    }
    if (kind == 0) {
      aes128_encrypt_opt<<<grid, BLOCK_DIM>>>((const uint4 *)c->d_in, (uint4 *)c->d_out, c->d_te0,
                                              c->d_enc_rk);
    } else if (kind == 1) {
      aes128_decrypt_opt<<<grid, BLOCK_DIM>>>((const uint4 *)c->d_in, (uint4 *)c->d_out, c->d_td0,
                                              c->d_isb, c->d_dec_rk);
    } else {
      aes128_encrypt_classic<<<classic_grid, BLOCK_DIM>>>(c->d_in, c->d_out, c->padded_blocks);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(stop);
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}
