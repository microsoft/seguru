// CUDA C++ reference for the SeGuRu PolyBench/GPU case study.
//
// Every kernel below is a deliberate one-for-one mirror of the corresponding
// SeGuRu kernel in `../src/`: same block shape, same tile geometry, same
// work-per-thread, same shared-memory layout (including the exact index
// expressions produced by the Rust `reshape_map!`s), same loop structure and
// unroll factors. The point of the comparison is to isolate *code generation*,
// so any algorithmic difference would invalidate it. Where a mirror is not
// exact it is called out in a comment.
//
// One extra, non-mirrored baseline is provided: cuBLAS SGEMM, so the reader can
// see how far both hand-written implementations are from the vendor library on
// this machine. That is reported as an additional column, never as "the" CUDA
// number.
//
// All timings are kernel-only: device buffers are allocated and filled once,
// outside the timed region, and the timed region is bracketed by CUDA events.

#include <cublas_v2.h>
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

#define CUBLAS_CHECK(expr)                                                                     \
  do {                                                                                         \
    cublasStatus_t _s = (expr);                                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                         \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__);              \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Small host helpers
// ---------------------------------------------------------------------------

static float *dev_alloc(size_t n) {
  float *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
  return p;
}

static void up(float *d, const float *h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
}

static void down(float *h, const float *d, size_t n) {
  CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// Mean time per iteration, in milliseconds, of `iters` back-to-back launches of
// `launch` after `warmup` untimed ones.
template <typename F>
static float time_kernel(int warmup, int iters, F launch) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// ---------------------------------------------------------------------------
// GEMM family: 16x16 CTA owning a 64x64 tile of C, 4x4 register micro-tile per
// thread, K walked in 16-wide slabs staged through shared memory.
//
// Shared-memory indexing mirrors the Rust `reshape_map!`s exactly:
//   As[k * 64 + r] with the staging store As[tx * 64 + ty + 16 * j]
//   Bs[k * 64 + c] with the staging store Bs[ty * 64 + tx + 16 * j]
// ---------------------------------------------------------------------------

#define G_BDIM 16
#define G_TILE 64
#define G_KTILE 16
#define G_SMEM (G_KTILE * G_TILE)

__global__ void gemm_kernel(const float *__restrict__ a, const float *__restrict__ b,
                            float *__restrict__ c, unsigned nk, float alpha, float beta) {
  __shared__ float as_[G_SMEM];
  __shared__ float bs_[G_SMEM];

  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const unsigned nj = gridDim.x * G_TILE;
  const unsigned row0 = blockIdx.y * G_TILE;
  const unsigned col0 = blockIdx.x * G_TILE;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (unsigned slab = 0; slab < nk / G_KTILE; ++slab) {
    const unsigned kt = slab * G_KTILE;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned jj = (unsigned)j;
      as_[tx * G_TILE + ty + G_BDIM * jj] = a[(row0 + ty + G_BDIM * jj) * nk + kt + tx];
      bs_[ty * G_TILE + tx + G_BDIM * jj] = b[(kt + ty) * nj + col0 + tx + G_BDIM * jj];
    }
    __syncthreads();

    for (unsigned kh = 0; kh < G_KTILE / 4; ++kh) {
#pragma unroll
      for (int kl = 0; kl < 4; ++kl) {
        const unsigned base = (kh * 4 + (unsigned)kl) * G_TILE;
        float af[4], bf[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          af[i] = as_[base + ty + G_BDIM * (unsigned)i];
          bf[i] = bs_[base + tx + G_BDIM * (unsigned)i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned r = row0 + ty + G_BDIM * (unsigned)i;
      const unsigned cl = col0 + tx + G_BDIM * (unsigned)j;
      c[r * nj + cl] = alpha * acc[i][j] + beta * c[r * nj + cl];
    }
}

// SYRK: C = alpha * A * A^T + beta * C. Both shared tiles are *rows* of A, so
// the two staging loads have an identical access pattern.
__global__ void syrk_kernel(const float *__restrict__ a, float *__restrict__ c, unsigned mm,
                            float alpha, float beta) {
  __shared__ float as_[G_SMEM];
  __shared__ float bs_[G_SMEM];

  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const unsigned nn = gridDim.x * G_TILE;
  const unsigned row0 = blockIdx.y * G_TILE;
  const unsigned col0 = blockIdx.x * G_TILE;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (unsigned slab = 0; slab < mm / G_KTILE; ++slab) {
    const unsigned kt = slab * G_KTILE;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned jj = (unsigned)j;
      as_[tx * G_TILE + ty + G_BDIM * jj] = a[(row0 + ty + G_BDIM * jj) * mm + kt + tx];
      bs_[tx * G_TILE + ty + G_BDIM * jj] = a[(col0 + ty + G_BDIM * jj) * mm + kt + tx];
    }
    __syncthreads();

    for (unsigned kh = 0; kh < G_KTILE / 4; ++kh) {
#pragma unroll
      for (int kl = 0; kl < 4; ++kl) {
        const unsigned base = (kh * 4 + (unsigned)kl) * G_TILE;
        float af[4], bf[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          af[i] = as_[base + ty + G_BDIM * (unsigned)i];
          bf[i] = bs_[base + tx + G_BDIM * (unsigned)i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned r = row0 + ty + G_BDIM * (unsigned)i;
      const unsigned cl = col0 + tx + G_BDIM * (unsigned)j;
      c[r * nn + cl] = alpha * acc[i][j] + beta * c[r * nn + cl];
    }
}

// SYR2K: four shared tiles per K slab (row and column blocks of both A and B).
__global__ void syr2k_kernel(const float *__restrict__ a, const float *__restrict__ b,
                             float *__restrict__ c, unsigned mm, float alpha, float beta) {
  __shared__ float ar_[G_SMEM];
  __shared__ float ac_[G_SMEM];
  __shared__ float br_[G_SMEM];
  __shared__ float bc_[G_SMEM];

  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const unsigned nn = gridDim.x * G_TILE;
  const unsigned row0 = blockIdx.y * G_TILE;
  const unsigned col0 = blockIdx.x * G_TILE;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (unsigned slab = 0; slab < mm / G_KTILE; ++slab) {
    const unsigned kt = slab * G_KTILE;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned jj = (unsigned)j;
      const unsigned s = tx * G_TILE + ty + G_BDIM * jj;
      const size_t ri = (size_t)(row0 + ty + G_BDIM * jj) * mm + kt + tx;
      const size_t ci = (size_t)(col0 + ty + G_BDIM * jj) * mm + kt + tx;
      ar_[s] = a[ri];
      ac_[s] = a[ci];
      br_[s] = b[ri];
      bc_[s] = b[ci];
    }
    __syncthreads();

    for (unsigned kh = 0; kh < G_KTILE / 4; ++kh) {
#pragma unroll
      for (int kl = 0; kl < 4; ++kl) {
        const unsigned base = (kh * 4 + (unsigned)kl) * G_TILE;
        float ar[4], br[4], acol[4], bcol[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned ro = base + ty + G_BDIM * (unsigned)i;
          const unsigned co = base + tx + G_BDIM * (unsigned)i;
          ar[i] = ar_[ro];
          br[i] = br_[ro];
          acol[i] = ac_[co];
          bcol[i] = bc_[co];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[i][j] += ar[i] * bcol[j] + br[i] * acol[j];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned r = row0 + ty + G_BDIM * (unsigned)i;
      const unsigned cl = col0 + tx + G_BDIM * (unsigned)j;
      c[r * nn + cl] = alpha * acc[i][j] + beta * c[r * nn + cl];
    }
}

// ---------------------------------------------------------------------------
// Mat-vec family: warp-per-row float4 reductions and thread-per-column sweeps.
// ---------------------------------------------------------------------------

#define MV_BX 32
#define MV_BY 8
#define COL_BDIM 256

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int s = 0; s < 5; ++s) v += __shfl_xor_sync(0xffffffffu, v, 1 << s, 32);
  return v;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// tmp[i] = sum_j A[i][j] * x[j], one warp per row.
__global__ void mv_row(const float4 *__restrict__ a, const float4 *__restrict__ x,
                       float *__restrict__ out, unsigned ny4) {
  const unsigned tx = threadIdx.x;
  const unsigned row = blockIdx.y * MV_BY + threadIdx.y;
  const size_t base = (size_t)row * ny4;
  float acc = 0.f;
  for (unsigned j = tx; j < ny4; j += MV_BX) acc += dot4(a[base + j], x[j]);
  const float s = warp_sum(acc);
  if (tx == 0) out[row] = s;
}

// x1[i] += sum_j A[i][j] * y1[j] (the accumulating variant used by MVT).
__global__ void mv_row_acc(const float4 *__restrict__ a, const float4 *__restrict__ x,
                           float *__restrict__ out, unsigned ny4) {
  const unsigned tx = threadIdx.x;
  const unsigned row = blockIdx.y * MV_BY + threadIdx.y;
  const size_t base = (size_t)row * ny4;
  float acc = 0.f;
  for (unsigned j = tx; j < ny4; j += MV_BX) acc += dot4(a[base + j], x[j]);
  const float s = warp_sum(acc);
  if (tx == 0) out[row] = out[row] + s;
}

// y[j] = sum_i A[i][j] * v[i], one thread per column, 4-way unrolled.
__global__ void mv_col(const float *__restrict__ a, const float *__restrict__ v,
                       float *__restrict__ out, unsigned nx, unsigned ny) {
  const unsigned j = blockIdx.x * COL_BDIM + threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (unsigned i = 0; i < nx; i += 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const unsigned ii = i + (unsigned)u;
      acc[u] += a[(size_t)ii * ny + j] * v[ii];
    }
  }
  out[j] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

// x2[i] += sum_j A[j][i] * y2[j] (accumulating column sweep used by MVT).
__global__ void mv_col_acc(const float *__restrict__ a, const float *__restrict__ v,
                           float *__restrict__ out, unsigned n) {
  const unsigned i = blockIdx.x * COL_BDIM + threadIdx.x;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (unsigned j = 0; j < n; j += 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const unsigned jj = j + (unsigned)u;
      acc[u] += a[(size_t)jj * n + i] * v[jj];
    }
  }
  out[i] = out[i] + (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

// y = alpha * A x + beta * B x, both products fused into one pass.
__global__ void gesummv_kernel(const float4 *__restrict__ a, const float4 *__restrict__ b,
                               const float4 *__restrict__ x, float *__restrict__ y, unsigned n4,
                               float alpha, float beta) {
  const unsigned tx = threadIdx.x;
  const unsigned row = blockIdx.y * MV_BY + threadIdx.y;
  const size_t base = (size_t)row * n4;
  float sa = 0.f, sb = 0.f;
  for (unsigned j = tx; j < n4; j += MV_BX) {
    const float4 xv = x[j];
    sa += dot4(a[base + j], xv);
    sb += dot4(b[base + j], xv);
  }
  sa = warp_sum(sa);
  sb = warp_sum(sb);
  if (tx == 0) y[row] = alpha * sa + beta * sb;
}

// ---------------------------------------------------------------------------
// Stencils
// ---------------------------------------------------------------------------

#define S_BX 32
#define S_BY 8
#define CONV2D_ROWS 4
#define CONV2D_CTA_ROWS (S_BY * CONV2D_ROWS)

// Immediate constants, exactly as in the Rust kernel (not `__constant__`
// memory, which would add a load the SeGuRu side does not perform).
#define C2_11 0.2f
#define C2_21 0.5f
#define C2_31 -0.8f
#define C2_12 -0.3f
#define C2_22 0.6f
#define C2_32 -0.9f
#define C2_13 0.4f
#define C2_23 0.7f
#define C2_33 0.10f

__device__ __forceinline__ void row3(const float *__restrict__ a, unsigned ir, unsigned nj,
                                     unsigned jm, unsigned j, unsigned jp, float *o) {
  const size_t base = (size_t)ir * nj;
  o[0] = a[base + jm];
  o[1] = a[base + j];
  o[2] = a[base + jp];
}

__global__ void conv2d_kernel(const float *__restrict__ a, float *__restrict__ b, unsigned ni,
                              unsigned nj) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned row0 = blockIdx.y * CONV2D_CTA_ROWS + threadIdx.y * CONV2D_ROWS;

  const unsigned jm = (j > 0 ? j : 1u) - 1u;
  const unsigned jp = min(j + 1u, nj - 1u);

  float w[3][3];
  row3(a, (row0 > 0 ? row0 : 1u) - 1u, nj, jm, j, jp, w[0]);
  row3(a, row0, nj, jm, j, jp, w[1]);
  row3(a, min(row0 + 1u, ni - 1u), nj, jm, j, jp, w[2]);

#pragma unroll
  for (int r = 0; r < CONV2D_ROWS; ++r) {
    const unsigned i = row0 + (unsigned)r;
    const bool interior = i > 0 && i + 1 < ni && j > 0 && j + 1 < nj;
    const float v = C2_11 * w[0][0] + C2_21 * w[0][1] + C2_31 * w[0][2] +
                    C2_12 * w[1][0] + C2_22 * w[1][1] + C2_32 * w[1][2] +
                    C2_13 * w[2][0] + C2_23 * w[2][1] + C2_33 * w[2][2];
    b[(size_t)i * nj + j] = interior ? v : 0.f;
    w[0][0] = w[1][0];
    w[0][1] = w[1][1];
    w[0][2] = w[1][2];
    w[1][0] = w[2][0];
    w[1][1] = w[2][1];
    w[1][2] = w[2][2];
    row3(a, min(i + 2u, ni - 1u), nj, jm, j, jp, w[2]);
  }
}

__device__ __forceinline__ float at3(const float *__restrict__ a, unsigned plane, unsigned nk,
                                     unsigned i, unsigned j, unsigned k) {
  return a[(size_t)i * plane + (size_t)j * nk + k];
}

// The weights (and the repeated terms) follow the PolyBench/GPU source, which
// the SeGuRu kernel reproduces verbatim.
__global__ void conv3d_kernel(const float *__restrict__ a, float *__restrict__ b, unsigned ni,
                              unsigned nj, unsigned nk) {
  const float C11 = 2.f, C21 = 5.f, C31 = -8.f;
  const float C12 = -3.f, C22 = 6.f, C32 = -9.f;
  const float C13 = 4.f, C23 = 7.f, C33 = 10.f;

  const unsigned k = blockIdx.x * S_BX + threadIdx.x;
  const unsigned j = blockIdx.y * S_BY + threadIdx.y;
  const unsigned i = blockIdx.z;

  const bool interior = i > 0 && i + 1 < ni && j > 0 && j + 1 < nj && k > 0 && k + 1 < nk;
  const unsigned im = (i > 0 ? i : 1u) - 1u;
  const unsigned ip = min(i + 1u, ni - 1u);
  const unsigned jm = (j > 0 ? j : 1u) - 1u;
  const unsigned jp = min(j + 1u, nj - 1u);
  const unsigned km = (k > 0 ? k : 1u) - 1u;
  const unsigned kp = min(k + 1u, nk - 1u);
  const unsigned plane = nj * nk;

  const float v = C11 * at3(a, plane, nk, im, jm, km) + C13 * at3(a, plane, nk, ip, jm, km) +
                  C21 * at3(a, plane, nk, im, jm, km) + C23 * at3(a, plane, nk, ip, jm, km) +
                  C31 * at3(a, plane, nk, im, jm, km) + C33 * at3(a, plane, nk, ip, jm, km) +
                  C12 * at3(a, plane, nk, i, jm, k) + C22 * at3(a, plane, nk, i, j, k) +
                  C32 * at3(a, plane, nk, i, jp, k) + C11 * at3(a, plane, nk, im, jm, kp) +
                  C13 * at3(a, plane, nk, ip, jm, kp) + C21 * at3(a, plane, nk, im, j, kp) +
                  C23 * at3(a, plane, nk, ip, j, kp) + C31 * at3(a, plane, nk, im, jp, kp) +
                  C33 * at3(a, plane, nk, ip, jp, kp);

  b[(size_t)i * plane + (size_t)j * nk + k] = interior ? v : 0.f;
}

#define J1_BDIM 256

__global__ void jacobi1d_step(const float *__restrict__ a, float *__restrict__ b, unsigned n) {
  const unsigned i = blockIdx.x * J1_BDIM + threadIdx.x;
  const unsigned im = (i > 0 ? i : 1u) - 1u;
  const unsigned ip = min(i + 1u, n - 1u);
  const float v = 0.33333f * (a[im] + a[i] + a[ip]);
  const bool interior = i > 0 && i + 1 < n;
  b[i] = interior ? v : b[i];
}

__global__ void jacobi1d_copy(const float *__restrict__ b, float *__restrict__ a, unsigned n) {
  const unsigned i = blockIdx.x * J1_BDIM + threadIdx.x;
  const bool interior = i > 0 && i + 1 < n;
  a[i] = interior ? b[i] : a[i];
}

__global__ void jacobi2d_step(const float *__restrict__ a, float *__restrict__ b, unsigned n) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned i = blockIdx.y * S_BY + threadIdx.y;
  const unsigned jm = (j > 0 ? j : 1u) - 1u;
  const unsigned jp = min(j + 1u, n - 1u);
  const unsigned im = (i > 0 ? i : 1u) - 1u;
  const unsigned ip = min(i + 1u, n - 1u);
  const float v = 0.2f * (a[(size_t)i * n + j] + a[(size_t)i * n + jm] + a[(size_t)i * n + jp] +
                          a[(size_t)ip * n + j] + a[(size_t)im * n + j]);
  const bool interior = i > 0 && i + 1 < n && j > 0 && j + 1 < n;
  b[(size_t)i * n + j] = interior ? v : b[(size_t)i * n + j];
}

__global__ void jacobi2d_copy(const float *__restrict__ b, float *__restrict__ a, unsigned n) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned i = blockIdx.y * S_BY + threadIdx.y;
  const bool interior = i > 0 && i + 1 < n && j > 0 && j + 1 < n;
  a[(size_t)i * n + j] = interior ? b[(size_t)i * n + j] : a[(size_t)i * n + j];
}

__global__ void fdtd_ey(float *__restrict__ ey, const float *__restrict__ hz, unsigned ny,
                        float fict) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned i = blockIdx.y * S_BY + threadIdx.y;
  const unsigned im = (i > 0 ? i : 1u) - 1u;
  const size_t o = (size_t)i * ny + j;
  const float v = ey[o] - 0.5f * (hz[o] - hz[(size_t)im * ny + j]);
  ey[o] = (i == 0) ? fict : v;
}

__global__ void fdtd_ex(float *__restrict__ ex, const float *__restrict__ hz, unsigned ny) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned i = blockIdx.y * S_BY + threadIdx.y;
  const unsigned jm = (j > 0 ? j : 1u) - 1u;
  const size_t o = (size_t)i * ny + j;
  const float v = ex[o] - 0.5f * (hz[o] - hz[(size_t)i * ny + jm]);
  ex[o] = (j == 0) ? ex[o] : v;
}

__global__ void fdtd_hz(float *__restrict__ hz, const float *__restrict__ ex,
                        const float *__restrict__ ey, unsigned nx, unsigned ny) {
  const unsigned j = blockIdx.x * S_BX + threadIdx.x;
  const unsigned i = blockIdx.y * S_BY + threadIdx.y;
  const unsigned jp = min(j + 1u, ny - 1u);
  const unsigned ip = min(i + 1u, nx - 1u);
  const size_t o = (size_t)i * ny + j;
  const float v =
      hz[o] - 0.7f * (ex[(size_t)i * ny + jp] - ex[o] + ey[(size_t)ip * ny + j] - ey[o]);
  hz[o] = (i + 1 < nx && j + 1 < ny) ? v : hz[o];
}

// ---------------------------------------------------------------------------
// Host-side benchmark entry points.
//
// Each returns the mean kernel time per iteration in milliseconds. After the
// timed loop the mutable inputs are re-uploaded and the sequence is run exactly
// once more, so the buffer copied back to the host is the result of a single
// clean application of the kernel (the timed loop deliberately leaves state
// evolving, which is fine for timing but not for verification).
// ---------------------------------------------------------------------------

extern "C" float cuda_gemm_bench(const float *ha, const float *hb, const float *hc, float *hout,
                                 int ni, int nj, int nk, float alpha, float beta, int warmup,
                                 int iters) {
  const size_t na = (size_t)ni * nk, nb = (size_t)nk * nj, ncn = (size_t)ni * nj;
  float *da = dev_alloc(na), *db = dev_alloc(nb), *dc = dev_alloc(ncn);
  up(da, ha, na);
  up(db, hb, nb);
  up(dc, hc, ncn);

  dim3 grid(nj / G_TILE, ni / G_TILE), block(G_BDIM, G_BDIM);
  auto launch = [&] { gemm_kernel<<<grid, block>>>(da, db, dc, nk, alpha, beta); };
  const float ms = time_kernel(warmup, iters, launch);

  up(dc, hc, ncn);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dc, ncn);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return ms;
}

// cuBLAS SGEMM on the same problem. Row-major C = A*B is computed as the
// column-major C^T = B^T * A^T, which needs no transposes at all.
extern "C" float cuda_gemm_cublas_bench(const float *ha, const float *hb, const float *hc,
                                        float *hout, int ni, int nj, int nk, float alpha,
                                        float beta, int warmup, int iters) {
  const size_t na = (size_t)ni * nk, nb = (size_t)nk * nj, ncn = (size_t)ni * nj;
  float *da = dev_alloc(na), *db = dev_alloc(nb), *dc = dev_alloc(ncn);
  up(da, ha, na);
  up(db, hb, nb);
  up(dc, hc, ncn);

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  auto launch = [&] {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, nj, ni, nk, &alpha, db, nj, da, nk,
                             &beta, dc, nj));
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(dc, hc, ncn);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dc, ncn);
  CUBLAS_CHECK(cublasDestroy(h));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return ms;
}

extern "C" float cuda_twomm_bench(const float *ha, const float *hb, const float *hc,
                                  const float *hd, float *hout, int ni, int nj, int nk, int nl,
                                  float alpha, float beta, int warmup, int iters) {
  const size_t na = (size_t)ni * nk, nb = (size_t)nk * nj, ncn = (size_t)nj * nl;
  const size_t nd = (size_t)ni * nl, nt = (size_t)ni * nj;
  float *da = dev_alloc(na), *db = dev_alloc(nb), *dc = dev_alloc(ncn);
  float *dd = dev_alloc(nd), *dt = dev_alloc(nt);
  up(da, ha, na);
  up(db, hb, nb);
  up(dc, hc, ncn);
  up(dd, hd, nd);
  CUDA_CHECK(cudaMemset(dt, 0, nt * sizeof(float)));

  dim3 block(G_BDIM, G_BDIM);
  dim3 g1(nj / G_TILE, ni / G_TILE), g2(nl / G_TILE, ni / G_TILE);
  auto launch = [&] {
    gemm_kernel<<<g1, block>>>(da, db, dt, nk, alpha, 0.f);
    gemm_kernel<<<g2, block>>>(dt, dc, dd, nj, 1.f, beta);
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(dd, hd, nd);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dd, nd);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  CUDA_CHECK(cudaFree(dd));
  CUDA_CHECK(cudaFree(dt));
  return ms;
}

extern "C" float cuda_threemm_bench(const float *ha, const float *hb, const float *hc,
                                    const float *hd, float *hout, int ni, int nj, int nk, int nl,
                                    int nm, int warmup, int iters) {
  const size_t na = (size_t)ni * nk, nb = (size_t)nk * nj;
  const size_t ncn = (size_t)nj * nm, nd = (size_t)nm * nl;
  const size_t ne = (size_t)ni * nj, nf = (size_t)nj * nl, ng = (size_t)ni * nl;
  float *da = dev_alloc(na), *db = dev_alloc(nb), *dc = dev_alloc(ncn), *dd = dev_alloc(nd);
  float *de = dev_alloc(ne), *df = dev_alloc(nf), *dg = dev_alloc(ng);
  up(da, ha, na);
  up(db, hb, nb);
  up(dc, hc, ncn);
  up(dd, hd, nd);

  dim3 block(G_BDIM, G_BDIM);
  dim3 g1(nj / G_TILE, ni / G_TILE), g2(nl / G_TILE, nj / G_TILE), g3(nl / G_TILE, ni / G_TILE);
  auto launch = [&] {
    gemm_kernel<<<g1, block>>>(da, db, de, nk, 1.f, 0.f);
    gemm_kernel<<<g2, block>>>(dc, dd, df, nm, 1.f, 0.f);
    gemm_kernel<<<g3, block>>>(de, df, dg, nj, 1.f, 0.f);
  };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dg, ng);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  CUDA_CHECK(cudaFree(dd));
  CUDA_CHECK(cudaFree(de));
  CUDA_CHECK(cudaFree(df));
  CUDA_CHECK(cudaFree(dg));
  return ms;
}

extern "C" float cuda_syrk_bench(const float *ha, const float *hc, float *hout, int n, int m,
                                 float alpha, float beta, int warmup, int iters) {
  const size_t na = (size_t)n * m, ncn = (size_t)n * n;
  float *da = dev_alloc(na), *dc = dev_alloc(ncn);
  up(da, ha, na);
  up(dc, hc, ncn);

  dim3 grid(n / G_TILE, n / G_TILE), block(G_BDIM, G_BDIM);
  auto launch = [&] { syrk_kernel<<<grid, block>>>(da, dc, m, alpha, beta); };
  const float ms = time_kernel(warmup, iters, launch);

  up(dc, hc, ncn);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dc, ncn);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dc));
  return ms;
}

extern "C" float cuda_syr2k_bench(const float *ha, const float *hb, const float *hc, float *hout,
                                  int n, int m, float alpha, float beta, int warmup, int iters) {
  const size_t na = (size_t)n * m, ncn = (size_t)n * n;
  float *da = dev_alloc(na), *db = dev_alloc(na), *dc = dev_alloc(ncn);
  up(da, ha, na);
  up(db, hb, na);
  up(dc, hc, ncn);

  dim3 grid(n / G_TILE, n / G_TILE), block(G_BDIM, G_BDIM);
  auto launch = [&] { syr2k_kernel<<<grid, block>>>(da, db, dc, m, alpha, beta); };
  const float ms = time_kernel(warmup, iters, launch);

  up(dc, hc, ncn);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dc, ncn);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return ms;
}

extern "C" float cuda_atax_bench(const float *ha, const float *hx, float *hout, int nx, int ny,
                                 int warmup, int iters) {
  const size_t na = (size_t)nx * ny;
  float *da = dev_alloc(na), *dx = dev_alloc(ny), *dt = dev_alloc(nx), *dy = dev_alloc(ny);
  up(da, ha, na);
  up(dx, hx, ny);

  dim3 b1(MV_BX, MV_BY), g1(1, nx / MV_BY);
  dim3 b2(COL_BDIM), g2(ny / COL_BDIM);
  auto launch = [&] {
    mv_row<<<g1, b1>>>((const float4 *)da, (const float4 *)dx, dt, ny / 4);
    mv_col<<<g2, b2>>>(da, dt, dy, nx, ny);
  };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dy, ny);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dt));
  CUDA_CHECK(cudaFree(dy));
  return ms;
}

extern "C" float cuda_bicg_bench(const float *ha, const float *hp, const float *hr, float *hs,
                                 float *hq, int nx, int ny, int warmup, int iters) {
  const size_t na = (size_t)nx * ny;
  float *da = dev_alloc(na), *dp = dev_alloc(ny), *dr = dev_alloc(nx);
  float *dq = dev_alloc(nx), *ds = dev_alloc(ny);
  up(da, ha, na);
  up(dp, hp, ny);
  up(dr, hr, nx);

  dim3 b1(MV_BX, MV_BY), g1(1, nx / MV_BY);
  dim3 b2(COL_BDIM), g2(ny / COL_BDIM);
  auto launch = [&] {
    mv_row<<<g1, b1>>>((const float4 *)da, (const float4 *)dp, dq, ny / 4);
    mv_col<<<g2, b2>>>(da, dr, ds, nx, ny);
  };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hs, ds, ny);
  down(hq, dq, nx);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dp));
  CUDA_CHECK(cudaFree(dr));
  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(ds));
  return ms;
}

extern "C" float cuda_gesummv_bench(const float *ha, const float *hb, const float *hx, float *hout,
                                    int n, float alpha, float beta, int warmup, int iters) {
  const size_t na = (size_t)n * n;
  float *da = dev_alloc(na), *db = dev_alloc(na), *dx = dev_alloc(n), *dy = dev_alloc(n);
  up(da, ha, na);
  up(db, hb, na);
  up(dx, hx, n);

  dim3 block(MV_BX, MV_BY), grid(1, n / MV_BY);
  auto launch = [&] {
    gesummv_kernel<<<grid, block>>>((const float4 *)da, (const float4 *)db, (const float4 *)dx, dy,
                                    n / 4, alpha, beta);
  };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, dy, n);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  return ms;
}

extern "C" float cuda_mvt_bench(const float *ha, const float *hx1, const float *hx2,
                                const float *hy1, const float *hy2, float *ox1, float *ox2, int n,
                                int warmup, int iters) {
  const size_t na = (size_t)n * n;
  float *da = dev_alloc(na), *dy1 = dev_alloc(n), *dy2 = dev_alloc(n);
  float *dx1 = dev_alloc(n), *dx2 = dev_alloc(n);
  up(da, ha, na);
  up(dy1, hy1, n);
  up(dy2, hy2, n);
  up(dx1, hx1, n);
  up(dx2, hx2, n);

  dim3 b1(MV_BX, MV_BY), g1(1, n / MV_BY);
  dim3 b2(COL_BDIM), g2(n / COL_BDIM);
  auto launch = [&] {
    mv_row_acc<<<g1, b1>>>((const float4 *)da, (const float4 *)dy1, dx1, n / 4);
    mv_col_acc<<<g2, b2>>>(da, dy2, dx2, n);
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(dx1, hx1, n);
  up(dx2, hx2, n);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(ox1, dx1, n);
  down(ox2, dx2, n);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dy1));
  CUDA_CHECK(cudaFree(dy2));
  CUDA_CHECK(cudaFree(dx1));
  CUDA_CHECK(cudaFree(dx2));
  return ms;
}

extern "C" float cuda_conv2d_bench(const float *ha, float *hout, int ni, int nj, int warmup,
                                   int iters) {
  const size_t n = (size_t)ni * nj;
  float *da = dev_alloc(n), *db = dev_alloc(n);
  up(da, ha, n);
  CUDA_CHECK(cudaMemset(db, 0, n * sizeof(float)));

  dim3 block(S_BX, S_BY), grid(nj / S_BX, ni / CONV2D_CTA_ROWS);
  auto launch = [&] { conv2d_kernel<<<grid, block>>>(da, db, ni, nj); };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, db, n);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  return ms;
}

extern "C" float cuda_conv3d_bench(const float *ha, float *hout, int ni, int nj, int nk,
                                   int warmup, int iters) {
  const size_t n = (size_t)ni * nj * nk;
  float *da = dev_alloc(n), *db = dev_alloc(n);
  up(da, ha, n);
  CUDA_CHECK(cudaMemset(db, 0, n * sizeof(float)));

  dim3 block(S_BX, S_BY), grid(nk / S_BX, nj / S_BY, ni);
  auto launch = [&] { conv3d_kernel<<<grid, block>>>(da, db, ni, nj, nk); };
  const float ms = time_kernel(warmup, iters, launch);

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(hout, db, n);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  return ms;
}

extern "C" float cuda_jacobi1d_bench(const float *ha, const float *hb, float *oa, float *ob, int n,
                                     int tsteps, int warmup, int iters) {
  float *da = dev_alloc(n), *db = dev_alloc(n);
  up(da, ha, n);
  up(db, hb, n);

  const int grid = n / J1_BDIM;
  auto launch = [&] {
    for (int t = 0; t < tsteps; ++t) {
      jacobi1d_step<<<grid, J1_BDIM>>>(da, db, n);
      jacobi1d_copy<<<grid, J1_BDIM>>>(db, da, n);
    }
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(da, ha, n);
  up(db, hb, n);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(oa, da, n);
  down(ob, db, n);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  return ms;
}

extern "C" float cuda_jacobi2d_bench(const float *ha, const float *hb, float *oa, float *ob, int n,
                                     int tsteps, int warmup, int iters) {
  const size_t nn = (size_t)n * n;
  float *da = dev_alloc(nn), *db = dev_alloc(nn);
  up(da, ha, nn);
  up(db, hb, nn);

  dim3 block(S_BX, S_BY), grid(n / S_BX, n / S_BY);
  auto launch = [&] {
    for (int t = 0; t < tsteps; ++t) {
      jacobi2d_step<<<grid, block>>>(da, db, n);
      jacobi2d_copy<<<grid, block>>>(db, da, n);
    }
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(da, ha, nn);
  up(db, hb, nn);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(oa, da, nn);
  down(ob, db, nn);
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  return ms;
}

extern "C" float cuda_fdtd2d_bench(const float *hex, const float *hey, const float *hhz,
                                   const float *hfict, float *oex, float *oey, float *ohz, int nx,
                                   int ny, int tmax, int warmup, int iters) {
  const size_t n = (size_t)nx * ny;
  float *dex = dev_alloc(n), *dey = dev_alloc(n), *dhz = dev_alloc(n);
  up(dex, hex, n);
  up(dey, hey, n);
  up(dhz, hhz, n);

  dim3 block(S_BX, S_BY), grid(ny / S_BX, nx / S_BY);
  auto launch = [&] {
    for (int t = 0; t < tmax; ++t) {
      fdtd_ey<<<grid, block>>>(dey, dhz, ny, hfict[t]);
      fdtd_ex<<<grid, block>>>(dex, dhz, ny);
      fdtd_hz<<<grid, block>>>(dhz, dex, dey, nx, ny);
    }
  };
  const float ms = time_kernel(warmup, iters, launch);

  up(dex, hex, n);
  up(dey, hey, n);
  up(dhz, hhz, n);
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  down(oex, dex, n);
  down(oey, dey, n);
  down(ohz, dhz, n);
  CUDA_CHECK(cudaFree(dex));
  CUDA_CHECK(cudaFree(dey));
  CUDA_CHECK(cudaFree(dhz));
  return ms;
}
