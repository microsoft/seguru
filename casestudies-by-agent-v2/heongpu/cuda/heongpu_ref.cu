// CUDA C++ reference for the SeGuRu HEonGPU case study.
//
// Every kernel here is an instruction-for-instruction mirror of the SeGuRu
// kernel of the same name: same tiling (4096 coefficients per CTA in shared
// memory, 512 threads, 8 coefficients per thread, four register-resident
// radix-8 rounds), same Shoup butterflies, same global-pass decomposition.
// The comparison therefore measures code generation, not algorithm choice.

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

#define TILE 4096
#define NTT_BDIM 512
#define EPT 8

#define ELEM_BDIM 256
#define ELEMS_PER_THREAD 4

typedef unsigned long long u64;

__device__ __forceinline__ u64 add_mod(u64 a, u64 b, u64 q) {
  u64 s = a + b;
  return s >= q ? s - q : s;
}

__device__ __forceinline__ u64 sub_mod(u64 a, u64 b, u64 q) {
  return a >= b ? a - b : a + q - b;
}

__device__ __forceinline__ u64 mul_mod_shoup(u64 a, u64 w, u64 wp, u64 q) {
  u64 hi = __umul64hi(a, wp);
  u64 r = a * w - hi * q;
  return r >= q ? r - q : r;
}

__device__ __forceinline__ u64 mul_mod(u64 a, u64 b, u64 q, u64 mu, uint32_t bit) {
  __uint128_t z = (__uint128_t)a * (__uint128_t)b;
  __uint128_t w = z >> (bit - 2);
  w = (w * (__uint128_t)mu) >> (bit + 3);
  u64 r = (u64)(z - w * (__uint128_t)q);
  return r >= q ? r - q : r;
}

// ---------------------------------------------------------------------------
// Register-resident radix-8 sub-transforms
// ---------------------------------------------------------------------------

__device__ __forceinline__ void fwd_radix8(u64 *v, const u64 *__restrict__ w,
                                           const u64 *__restrict__ ws, u64 q, uint32_t h0) {
  {
    u64 s = w[h0], sp = ws[h0];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      u64 u = v[j];
      u64 t = mul_mod_shoup(v[j + 4], s, sp, q);
      v[j] = add_mod(u, t, q);
      v[j + 4] = sub_mod(u, t, q);
    }
  }
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    uint32_t idx = 2 * h0 + g;
    u64 s = w[idx], sp = ws[idx];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      int a = g * 4 + jj;
      u64 u = v[a];
      u64 t = mul_mod_shoup(v[a + 2], s, sp, q);
      v[a] = add_mod(u, t, q);
      v[a + 2] = sub_mod(u, t, q);
    }
  }
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t idx = 4 * h0 + g;
    u64 s = w[idx], sp = ws[idx];
    int a = g * 2;
    u64 u = v[a];
    u64 t = mul_mod_shoup(v[a + 1], s, sp, q);
    v[a] = add_mod(u, t, q);
    v[a + 1] = sub_mod(u, t, q);
  }
}

__device__ __forceinline__ void inv_radix8(u64 *v, const u64 *__restrict__ w,
                                           const u64 *__restrict__ ws, u64 q, uint32_t a0) {
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t idx = a0 + g;
    u64 s = w[idx], sp = ws[idx];
    int a = g * 2;
    u64 u = v[a], t = v[a + 1];
    v[a] = add_mod(u, t, q);
    v[a + 1] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
  }
  uint32_t a1 = a0 >> 1;
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    uint32_t idx = a1 + g;
    u64 s = w[idx], sp = ws[idx];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      int a = g * 4 + jj;
      u64 u = v[a], t = v[a + 2];
      v[a] = add_mod(u, t, q);
      v[a + 2] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
    }
  }
  {
    uint32_t a2 = a0 >> 2;
    u64 s = w[a2], sp = ws[a2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      u64 u = v[j], t = v[j + 4];
      v[j] = add_mod(u, t, q);
      v[j + 4] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
    }
  }
}

// ---------------------------------------------------------------------------
// Shared-memory-resident tile kernels
// ---------------------------------------------------------------------------

__global__ void ntt_forward_tile(const u64 *__restrict__ inp, u64 *__restrict__ out,
                                 const u64 *__restrict__ w, const u64 *__restrict__ ws, u64 q,
                                 uint32_t m0) {
  __shared__ u64 smem[TILE];
  const uint32_t tid = threadIdx.x;
  const uint32_t blk = blockIdx.x;
  const uint32_t tile_base = blk * TILE;
  const uint32_t base = m0 + (blk & (m0 - 1));

  u64 v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = inp[tile_base + tid + j * 512];
  fwd_radix8(v, w, ws, q, base);
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[tid + j * 512] = v[j];
  __syncthreads();

  uint32_t t_lo = tid & 63, t_hi = tid >> 6;
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[t_lo + j * 64 + t_hi * 512];
  fwd_radix8(v, w, ws, q, (base << 3) + t_hi);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[t_lo + j * 64 + t_hi * 512] = v[j];
  __syncthreads();

  t_lo = tid & 7;
  t_hi = tid >> 3;
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[t_lo + j * 8 + t_hi * 64];
  fwd_radix8(v, w, ws, q, (base << 6) + t_hi);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[t_lo + j * 8 + t_hi * 64] = v[j];
  __syncthreads();

#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[tid * 8 + j];
  fwd_radix8(v, w, ws, q, (base << 9) + tid);
#pragma unroll
  for (int j = 0; j < 8; ++j) out[tile_base + tid * 8 + j] = v[j];
}

__global__ void ntt_inverse_tile(const u64 *__restrict__ inp, u64 *__restrict__ out,
                                 const u64 *__restrict__ w, const u64 *__restrict__ ws, u64 q,
                                 uint32_t log_n, u64 scale, u64 scale_shoup) {
  __shared__ u64 smem[TILE];
  const uint32_t tid = threadIdx.x;
  const uint32_t blk = blockIdx.x;
  const uint32_t n = 1u << log_n;
  const uint32_t m0 = n / TILE;
  const uint32_t tile_base = blk * TILE;
  const uint32_t cp = blk & (m0 - 1);

  u64 v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = inp[tile_base + tid * 8 + j];
  inv_radix8(v, w, ws, q, (n >> 1) + (cp << 11) + 4 * tid);
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[tid * 8 + j] = v[j];
  __syncthreads();

  uint32_t t_lo = tid & 7, t_hi = tid >> 3;
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[t_lo + j * 8 + t_hi * 64];
  inv_radix8(v, w, ws, q, (n >> 4) + (cp << 8) + 4 * t_hi);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[t_lo + j * 8 + t_hi * 64] = v[j];
  __syncthreads();

  t_lo = tid & 63;
  t_hi = tid >> 6;
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[t_lo + j * 64 + t_hi * 512];
  inv_radix8(v, w, ws, q, (n >> 7) + (cp << 5) + 4 * t_hi);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) smem[t_lo + j * 64 + t_hi * 512] = v[j];
  __syncthreads();

#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = smem[tid + j * 512];
  inv_radix8(v, w, ws, q, (n >> 10) + (cp << 2));
#pragma unroll
  for (int j = 0; j < 8; ++j)
    out[tile_base + tid + j * 512] = mul_mod_shoup(v[j], scale, scale_shoup, q);
}

// ---------------------------------------------------------------------------
// Global passes (butterfly distance larger than a tile)
// ---------------------------------------------------------------------------

__global__ void ntt_stage_forward(const u64 *__restrict__ inp, u64 *__restrict__ out,
                                  const u64 *__restrict__ w, const u64 *__restrict__ ws, u64 q,
                                  uint32_t log_t, uint32_t log_n) {
  const uint32_t t = 1u << log_t;
  const uint32_t tq = t >> 2;
  const uint32_t m_blocks = 1u << (log_n - 1 - log_t);
  const uint32_t lin = blockIdx.x * NTT_BDIM + threadIdx.x;
  const uint32_t low = lin & (tq - 1);
  const uint32_t i = (lin >> (log_t - 2)) & (m_blocks - 1);
  const uint32_t poly = lin >> (log_n - 3);
  const uint32_t base = (poly << log_n) + i * (t << 1) + low * 4;
  const u64 s = w[m_blocks + i], sp = ws[m_blocks + i];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    u64 u = inp[base + e];
    u64 x = mul_mod_shoup(inp[base + e + t], s, sp, q);
    out[base + e] = add_mod(u, x, q);
    out[base + e + t] = sub_mod(u, x, q);
  }
}

__global__ void ntt_stage_inverse(const u64 *__restrict__ inp, u64 *__restrict__ out,
                                  const u64 *__restrict__ w, const u64 *__restrict__ ws, u64 q,
                                  uint32_t log_t, uint32_t log_n, u64 scale, u64 scale_shoup) {
  const uint32_t t = 1u << log_t;
  const uint32_t tq = t >> 2;
  const uint32_t m_blocks = 1u << (log_n - 1 - log_t);
  const uint32_t lin = blockIdx.x * NTT_BDIM + threadIdx.x;
  const uint32_t low = lin & (tq - 1);
  const uint32_t i = (lin >> (log_t - 2)) & (m_blocks - 1);
  const uint32_t poly = lin >> (log_n - 3);
  const uint32_t base = (poly << log_n) + i * (t << 1) + low * 4;
  const u64 s = w[m_blocks + i], sp = ws[m_blocks + i];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    u64 u = inp[base + e];
    u64 x = inp[base + e + t];
    u64 lo = add_mod(u, x, q);
    u64 hi = mul_mod_shoup(sub_mod(u, x, q), s, sp, q);
    out[base + e] = mul_mod_shoup(lo, scale, scale_shoup, q);
    out[base + e + t] = mul_mod_shoup(hi, scale, scale_shoup, q);
  }
}

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------

__global__ void poly_add(const u64 *__restrict__ a, const u64 *__restrict__ b,
                         u64 *__restrict__ out, u64 q) {
  const uint32_t nthreads = gridDim.x * ELEM_BDIM;
  const uint32_t gid = blockIdx.x * ELEM_BDIM + threadIdx.x;
#pragma unroll
  for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
    uint32_t i = gid + k * nthreads;
    out[i] = add_mod(a[i], b[i], q);
  }
}

__global__ void poly_mul(const u64 *__restrict__ a, const u64 *__restrict__ b,
                         u64 *__restrict__ out, u64 q, u64 mu, uint32_t bit) {
  const uint32_t nthreads = gridDim.x * ELEM_BDIM;
  const uint32_t gid = blockIdx.x * ELEM_BDIM + threadIdx.x;
#pragma unroll
  for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
    uint32_t i = gid + k * nthreads;
    out[i] = mul_mod(a[i], b[i], q, mu, bit);
  }
}

__global__ void cipher_plain_mul(const u64 *__restrict__ c, const u64 *__restrict__ p,
                                 u64 *__restrict__ out, uint32_t n_mask, u64 q, u64 mu,
                                 uint32_t bit) {
  const uint32_t nthreads = gridDim.x * ELEM_BDIM;
  const uint32_t gid = blockIdx.x * ELEM_BDIM + threadIdx.x;
#pragma unroll
  for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
    uint32_t i = gid + k * nthreads;
    out[i] = mul_mod(c[i], p[i & n_mask], q, mu, bit);
  }
}

// ---------------------------------------------------------------------------
// Host-side context
// ---------------------------------------------------------------------------

struct CudaNttCtx {
  uint32_t n;
  uint32_t log_n;
  uint32_t batch;
  u64 q, mu;
  uint32_t bit;
  u64 n_inv, n_inv_shoup, one_shoup;
  size_t elems;  // n * batch
  u64 *a, *b;    // ping-pong data buffers
  u64 *aux;      // second operand for the element-wise kernels
  u64 *wf, *wfs, *wi, *wis;
  u64 *result;   // whichever buffer holds the last result
};

extern "C" CudaNttCtx *cuda_ntt_create(uint32_t n, uint32_t log_n, uint32_t batch, u64 q, u64 mu,
                                       uint32_t bit, u64 n_inv, u64 n_inv_shoup, u64 one_shoup,
                                       const u64 *h_data, const u64 *h_aux, const u64 *wf,
                                       const u64 *wfs, const u64 *wi, const u64 *wis) {
  CudaNttCtx *c = (CudaNttCtx *)calloc(1, sizeof(CudaNttCtx));
  c->n = n;
  c->log_n = log_n;
  c->batch = batch;
  c->q = q;
  c->mu = mu;
  c->bit = bit;
  c->n_inv = n_inv;
  c->n_inv_shoup = n_inv_shoup;
  c->one_shoup = one_shoup;
  c->elems = (size_t)n * batch;
  size_t bytes = c->elems * sizeof(u64);
  size_t tbytes = (size_t)n * sizeof(u64);
  CUDA_CHECK(cudaMalloc(&c->a, bytes));
  CUDA_CHECK(cudaMalloc(&c->b, bytes));
  CUDA_CHECK(cudaMalloc(&c->aux, bytes));
  CUDA_CHECK(cudaMalloc(&c->wf, tbytes));
  CUDA_CHECK(cudaMalloc(&c->wfs, tbytes));
  CUDA_CHECK(cudaMalloc(&c->wi, tbytes));
  CUDA_CHECK(cudaMalloc(&c->wis, tbytes));
  CUDA_CHECK(cudaMemcpy(c->a, h_data, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->aux, h_aux, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(c->b, 0, bytes));
  CUDA_CHECK(cudaMemcpy(c->wf, wf, tbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->wfs, wfs, tbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->wi, wi, tbytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c->wis, wis, tbytes, cudaMemcpyHostToDevice));
  c->result = c->a;
  return c;
}

extern "C" void cuda_ntt_destroy(CudaNttCtx *c) {
  CUDA_CHECK(cudaFree(c->a));
  CUDA_CHECK(cudaFree(c->b));
  CUDA_CHECK(cudaFree(c->aux));
  CUDA_CHECK(cudaFree(c->wf));
  CUDA_CHECK(cudaFree(c->wfs));
  CUDA_CHECK(cudaFree(c->wi));
  CUDA_CHECK(cudaFree(c->wis));
  free(c);
}

extern "C" void cuda_ntt_copy_out(CudaNttCtx *c, u64 *h_out) {
  CUDA_CHECK(cudaMemcpy(h_out, c->result, c->elems * sizeof(u64), cudaMemcpyDeviceToHost));
}

extern "C" void cuda_ntt_reset(CudaNttCtx *c, const u64 *h_data) {
  CUDA_CHECK(cudaMemcpy(c->a, h_data, c->elems * sizeof(u64), cudaMemcpyHostToDevice));
  c->result = c->a;
}

// kind: 0 = forward NTT, 1 = inverse NTT, 2 = poly_add, 3 = poly_mul,
//       4 = cipher_plain_mul
static void run_once(CudaNttCtx *c, int kind) {
  const uint32_t grid = (uint32_t)(c->elems / TILE);
  const uint32_t egrid = (uint32_t)(c->elems / (ELEM_BDIM * ELEMS_PER_THREAD));
  const uint32_t passes = c->log_n - 12;
  u64 *src = c->a, *dst = c->b, *tmp;
  switch (kind) {
    case 0:
      for (uint32_t s = 0; s < passes; ++s) {
        ntt_stage_forward<<<grid, NTT_BDIM>>>(src, dst, c->wf, c->wfs, c->q, c->log_n - 1 - s,
                                              c->log_n);
        tmp = src;
        src = dst;
        dst = tmp;
      }
      ntt_forward_tile<<<grid, NTT_BDIM>>>(src, dst, c->wf, c->wfs, c->q, c->n / TILE);
      c->result = dst;
      break;
    case 1:
      ntt_inverse_tile<<<grid, NTT_BDIM>>>(src, dst, c->wi, c->wis, c->q, c->log_n,
                                           passes == 0 ? c->n_inv : 1,
                                           passes == 0 ? c->n_inv_shoup : c->one_shoup);
      tmp = src;
      src = dst;
      dst = tmp;
      for (uint32_t s = 0; s < passes; ++s) {
        bool last = (s + 1 == passes);
        ntt_stage_inverse<<<grid, NTT_BDIM>>>(src, dst, c->wi, c->wis, c->q, 12 + s, c->log_n,
                                              last ? c->n_inv : 1,
                                              last ? c->n_inv_shoup : c->one_shoup);
        tmp = src;
        src = dst;
        dst = tmp;
      }
      c->result = src;
      break;
    case 2:
      poly_add<<<egrid, ELEM_BDIM>>>(c->a, c->aux, c->b, c->q);
      c->result = c->b;
      break;
    case 3:
      poly_mul<<<egrid, ELEM_BDIM>>>(c->a, c->aux, c->b, c->q, c->mu, c->bit);
      c->result = c->b;
      break;
    case 4:
      cipher_plain_mul<<<egrid, ELEM_BDIM>>>(c->a, c->aux, c->b, c->n - 1, c->q, c->mu, c->bit);
      c->result = c->b;
      break;
    default:
      abort();
  }
}

extern "C" float cuda_ntt_bench(CudaNttCtx *c, int kind, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) run_once(c, kind);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) run_once(c, kind);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / (float)iters;
}
