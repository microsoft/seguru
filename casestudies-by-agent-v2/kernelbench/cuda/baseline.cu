#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

constexpr int BLOCK = 256;
constexpr int WARMUP = 20;
constexpr int ITERS = 200;

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

// Appends one row to $BENCH_CSV, matching the schema and the workload/parameter
// split used by kernelbench/src/bin/bench.rs: a label like "max_pool1d k=4 s=4"
// becomes workload "max_pool1d" with the config folded into the parameter, so
// the SeGuRu and CUDA rows join on (suite, workload, parameter).
static void csv_row(const char* op, int rows, int cols,
                    const char* metric, double value, const char* units) {
  const char* path = std::getenv("BENCH_CSV");
  if (!path) return;
  std::string label(op);
  size_t sp = label.find(' ');
  std::string workload = label.substr(0, sp);
  char shape[64];
  std::snprintf(shape, sizeof(shape), "%dx%d", rows, cols);
  std::string parameter(shape);
  if (sp != std::string::npos) {
    std::string extra = label.substr(sp + 1);
    for (char& c : extra)
      if (c == ' ') c = '/';
    parameter += "/" + extra;
  }
  FILE* f = std::fopen(path, "a");
  if (!f) return;
  std::fprintf(f, "kernelbench,%s,%s,cuda,%s,%.6f,%s\n",
               workload.c_str(), parameter.c_str(), metric, value, units);
  std::fclose(f);
}

enum class Ewise { Relu, Gelu, Sigmoid, Tanh, Swish, Softplus, Leaky };

template <Ewise OP>
__global__ void ewise_kernel(const float* x, float* y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float v = x[i];
    if constexpr (OP == Ewise::Relu) v = fmaxf(v, 0.0f);
    if constexpr (OP == Ewise::Gelu) v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    if constexpr (OP == Ewise::Sigmoid) v = 1.0f / (1.0f + expf(-v));
    if constexpr (OP == Ewise::Tanh) v = tanhf(v);
    if constexpr (OP == Ewise::Swish) v = v / (1.0f + expf(-v));
    if constexpr (OP == Ewise::Softplus) v = log1pf(expf(-fabsf(v))) + fmaxf(v, 0.0f);
    if constexpr (OP == Ewise::Leaky) v = v > 0.0f ? v : 0.01f * v;
    y[i] = v;
  }
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
  return v;
}

template <int MODE>
__global__ void row_kernel(const float* x, const float* weight, const float* bias,
                           float* y, float* scalar, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  __shared__ float smem[8], aux;
  const float* xr = x + row * cols;
  float* yr = y + row * cols;
  float a = (MODE == 3 || MODE == 4 || MODE == 10 || MODE == 13) ? -INFINITY : 0.0f;
  if constexpr (MODE == 0 || MODE == 1 || MODE == 2 || MODE == 3 || MODE == 4 ||
                MODE == 5 || MODE == 6 || MODE == 7 || MODE == 8 || MODE == 9 ||
                MODE == 10 || MODE == 11 || MODE == 12 || MODE == 13) {
    for (int i = tid; i < cols; i += BLOCK) {
      float v = xr[i];
      if constexpr (MODE == 0) a += v;
      if constexpr (MODE == 1) a += fabsf(v);
      if constexpr (MODE == 2) a += v * v;
      if constexpr (MODE == 3 || MODE == 4 || MODE == 10 || MODE == 13) a = fmaxf(a, v);
      if constexpr (MODE == 5) a += v;
      if constexpr (MODE == 6) a += v * v;
      if constexpr (MODE == 7) a += v;
      if constexpr (MODE == 8 || MODE == 9 || MODE == 11 || MODE == 12) a += v;
    }
  }
  float ra = (MODE == 3 || MODE == 4 || MODE == 10 || MODE == 13) ? warp_max(a) : warp_sum(a);
  if (lane == 0) smem[wid] = ra;
  __syncthreads();
  if (wid == 0) {
    float v = lane < 8 ? smem[lane] : ((MODE == 3 || MODE == 4 || MODE == 10 || MODE == 13) ? -INFINITY : 0.0f);
    v = (MODE == 3 || MODE == 4 || MODE == 10 || MODE == 13) ? warp_max(v) : warp_sum(v);
    if (lane == 0) aux = v;
  }
  __syncthreads();
  float total = aux;

  if constexpr (MODE == 3) {
    float s = 0.0f;
    for (int i = tid; i < cols; i += BLOCK) s += expf(xr[i] - total);
    s = warp_sum(s);
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    if (wid == 0) {
      float v = lane < 8 ? smem[lane] : 0.0f;
      v = warp_sum(v);
      if (lane == 0) aux = v;
    }
    __syncthreads();
    for (int i = tid; i < cols; i += BLOCK) yr[i] = expf(xr[i] - total) / aux;
  } else if constexpr (MODE == 5) {
    float mean = total / cols;
    for (int i = tid; i < cols; i += BLOCK) yr[i] = xr[i] - mean;
  } else if constexpr (MODE == 1) {
    float scale = 1.0f / (total + 1e-5f);
    for (int i = tid; i < cols; i += BLOCK) yr[i] = xr[i] * scale;
  } else if constexpr (MODE == 2) {
    float scale = rsqrtf(total + 1e-5f);
    for (int i = tid; i < cols; i += BLOCK) yr[i] = xr[i] * scale;
  } else if constexpr (MODE == 6) {
    float mean = total / cols, v = 0.0f;
    for (int i = tid; i < cols; i += BLOCK) { float d = xr[i] - mean; v += d * d; }
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    if (wid == 0) { float z = lane < 8 ? smem[lane] : 0.0f; z = warp_sum(z); if (!lane) aux = rsqrtf(z / cols + 1e-5f); }
    __syncthreads();
    for (int i = tid; i < cols; i += BLOCK)
      yr[i] = (xr[i] - mean) * aux * weight[i] + bias[i];
  } else if constexpr (MODE == 7) {
    float inv = rsqrtf(total / cols + 1e-5f);
    for (int i = tid; i < cols; i += BLOCK) yr[i] = xr[i] * inv;
  } else if constexpr (MODE == 8 || MODE == 9 || MODE == 10 || MODE == 11) {
    if (tid == 0) scalar[row] = MODE == 8 ? total : MODE == 9 ? total / cols : total;
  } else if constexpr (MODE == 12) {
    if (tid == 0) scalar[row] = total;
  } else if constexpr (MODE == 13) {
    float m = total; int idx = cols;
    for (int i = tid; i < cols; i += BLOCK) if (xr[i] == m) idx = min(idx, i);
    __shared__ int inds[8];
    if (lane == 0) inds[wid] = idx;
    __syncthreads();
    if (tid == 0) { int r = cols; for (int i = 0; i < 8; ++i) r = min(r, inds[i]); scalar[row] = (float)r; }
  }
}

__device__ __forceinline__ float warp_scan_inclusive(float v) {
  for (int off = 1; off < 32; off <<= 1) {
    float n = __shfl_up_sync(0xffffffff, v, off);
    if ((threadIdx.x & 31) >= off) v += n;
  }
  return v;
}

// Row-wise inclusive scan, structured like the SeGuRu kernel it is compared
// against: every thread in the block participates in each tile and a running
// carry links the tiles. A single-threaded scan would make the comparison an
// algorithm difference rather than a compiler one.
__global__ void cumsum_kernel(const float* x, float* y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * cols;
  float* yr = y + (size_t)row * cols;
  __shared__ float wsum[BLOCK / 32];
  __shared__ float carry;
  int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  if (tid == 0) carry = 0.0f;
  __syncthreads();
  for (int base = 0; base < cols; base += BLOCK) {
    int i = base + tid;
    float v = (i < cols) ? xr[i] : 0.0f;
    float inc = warp_scan_inclusive(v);
    if (lane == 31) wsum[wid] = inc;
    __syncthreads();
    if (wid == 0) {
      float t = (lane < BLOCK / 32) ? wsum[lane] : 0.0f;
      float s = warp_scan_inclusive(t);
      if (lane < BLOCK / 32) wsum[lane] = s - t;
    }
    __syncthreads();
    float off = carry + wsum[wid];
    if (i < cols) yr[i] = inc + off;
    __syncthreads();
    if (tid == BLOCK - 1) carry = off + inc;
    __syncthreads();
  }
}

__global__ void log_softmax_kernel(const float* x, float* y, int rows, int cols) {
  int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  if (row >= rows) return;
  __shared__ float sm[8], m, s;
  const float* xr = x + row * cols; float* yr = y + row * cols;
  float a = -INFINITY, z = 0;
  for (int i = tid; i < cols; i += BLOCK) a = fmaxf(a, xr[i]);
  a = warp_max(a); if (!lane) sm[wid] = a; __syncthreads();
  if (!wid) { float v = lane < 8 ? sm[lane] : -INFINITY; v = warp_max(v); if (!lane) m = v; }
  __syncthreads();
  for (int i = tid; i < cols; i += BLOCK) z += expf(xr[i] - m);
  z = warp_sum(z); if (!lane) sm[wid] = z; __syncthreads();
  if (!wid) { float v = lane < 8 ? sm[lane] : 0; v = warp_sum(v); if (!lane) s = logf(v); }
  __syncthreads();
  for (int i = tid; i < cols; i += BLOCK) yr[i] = xr[i] - m - s;
}

__global__ void mse_kernel(const float* a, const float* b, float* out, int n) {
  __shared__ float sm[BLOCK / 32]; float v = 0;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
    float d = a[i] - b[i]; v += d * d;
  }
  v = warp_sum(v); int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (!lane) sm[wid] = v; __syncthreads();
  if (!wid) { v = lane < 8 ? sm[lane] : 0; v = warp_sum(v); if (!threadIdx.x) out[0] = v / n; }
}

__global__ void pool_kernel(const float* x, float* y, int nout) {
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < nout; i += gridDim.x * BLOCK) {
    int j = i * 4; y[i] = fmaxf(fmaxf(x[j], x[j + 1]), fmaxf(x[j + 2], x[j + 3]));
  }
}

struct Measurement { const char* op; double us; size_t bytes; };

template <class F>
double measure(F launch) {
  for (int i = 0; i < WARMUP; ++i) launch();
  CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t start, stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  CUDA_OK(cudaEventRecord(start));
  for (int i = 0; i < ITERS; ++i) launch();
  CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop));
  float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  return ms * 1000.0 / ITERS;
}

template <Ewise OP>
double bench_ewise(const float* x, float* y, int n) {
  int grid = (n + BLOCK - 1) / BLOCK;
  return measure([&] { ewise_kernel<OP><<<grid, BLOCK>>>(x, y, n); });
}

int main() {
  CUDA_OK(cudaSetDevice(0));
  const int shapes[][2] = {{2048, 2048}, {8192, 8192}, {32768, 32768}};
  std::printf("| Operator | Shape | Time (us) | GB/s |\n|---|---|---:|---:|\n");
  for (const auto& shape : shapes) {
    int rows = shape[0], cols = shape[1], n = rows * cols;
    std::vector<float> hx(n), hx2(n), hw(cols), hb(cols);
    for (int i = 0; i < n; ++i) { hx[i] = std::sin(i * 0.001f); hx2[i] = std::cos(i * 0.002f); }
    for (int i = 0; i < cols; ++i) { hw[i] = 1.0f; hb[i] = 0.0f; }
    float *x, *x2, *y, *w, *b, *scalar;
    CUDA_OK(cudaMalloc(&x, n * sizeof(float))); CUDA_OK(cudaMalloc(&x2, n * sizeof(float)));
    CUDA_OK(cudaMalloc(&y, n * sizeof(float))); CUDA_OK(cudaMalloc(&w, cols * sizeof(float)));
    CUDA_OK(cudaMalloc(&b, cols * sizeof(float))); CUDA_OK(cudaMalloc(&scalar, rows * sizeof(float)));
    CUDA_OK(cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(x2, hx2.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(w, hw.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b, hb.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    auto print = [&](const char* op, double us, size_t bytes) {
      double gbps = bytes / (us * 1e-6) / 1e9;
      std::printf("| %s | %dx%d | %.1f | %.0f |\n", op, rows, cols, us, gbps);
      csv_row(op, rows, cols, "time", us, "us");
      csv_row(op, rows, cols, "throughput", gbps, "GB/s");
    };
    print("relu", bench_ewise<Ewise::Relu>(x, y, n), 2ull * n * 4);
    print("gelu", bench_ewise<Ewise::Gelu>(x, y, n), 2ull * n * 4);
    print("sigmoid", bench_ewise<Ewise::Sigmoid>(x, y, n), 2ull * n * 4);
    print("tanh", bench_ewise<Ewise::Tanh>(x, y, n), 2ull * n * 4);
    print("swish", bench_ewise<Ewise::Swish>(x, y, n), 2ull * n * 4);
    print("softplus", bench_ewise<Ewise::Softplus>(x, y, n), 2ull * n * 4);
    print("leaky_relu", bench_ewise<Ewise::Leaky>(x, y, n), 2ull * n * 4);
    int grid = rows;
    print("softmax", measure([&] { row_kernel<3><<<grid, BLOCK>>>(x,w,b,y,scalar,rows,cols); }), 2ull*n*4);
    print("log_softmax", measure([&] { log_softmax_kernel<<<grid,BLOCK>>>(x,y,rows,cols); }), 2ull*n*4);
    print("cumsum", measure([&] { cumsum_kernel<<<rows,BLOCK>>>(x,y,rows,cols); }), 2ull*n*4);
    print("rms_norm", measure([&] { row_kernel<7><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), 2ull*n*4);
    print("l1_norm", measure([&] { row_kernel<1><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), 2ull*n*4);
    print("l2_norm", measure([&] { row_kernel<2><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), 2ull*n*4);
    print("layer_norm", measure([&] { row_kernel<6><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), 2ull*n*4);
    print("sum_dim", measure([&] { row_kernel<8><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), n*4 + rows*4);
    print("mean_dim", measure([&] { row_kernel<9><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), n*4 + rows*4);
    print("max_dim", measure([&] { row_kernel<10><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), n*4 + rows*4);
    print("argmax_dim", measure([&] { row_kernel<13><<<grid,BLOCK>>>(x,w,b,y,scalar,rows,cols); }), n*4 + rows*4);
    print("mse_loss", measure([&] { mse_kernel<<<grid,BLOCK>>>(x,x2,scalar,n); }), 2ull*n*4);
    int nout = rows * cols / 4;
    print("max_pool1d k=4 s=4", measure([&] { pool_kernel<<<(nout+BLOCK-1)/BLOCK,BLOCK>>>(x,y,nout); }), (n+nout)*4);
    CUDA_OK(cudaFree(x)); CUDA_OK(cudaFree(x2)); CUDA_OK(cudaFree(y)); CUDA_OK(cudaFree(w));
    CUDA_OK(cudaFree(b)); CUDA_OK(cudaFree(scalar));
  }
}
