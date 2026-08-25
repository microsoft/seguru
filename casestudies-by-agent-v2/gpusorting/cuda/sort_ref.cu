// CUDA reference baselines for the SeGuRu radix sort benchmark.
//
// Two baselines are provided:
//   * CUB `DeviceRadixSort::SortKeys` — NVIDIA's production sort, and the
//     algorithm the SeGuRu kernels are modelled on. This is the honest bar.
//   * Thrust `sort` — the convenience API most applications actually reach for.
//
// Timings are kernel-only: allocation, the temporary-storage query and the host
// transfers all happen outside the timed region.

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t _e = (expr);                                                      \
    if (_e != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,        \
                   __LINE__, cudaGetErrorString(_e));                             \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

struct SortCtx {
  unsigned int *d_src;   // pristine copy of the unsorted keys
  unsigned int *d_a;     // working buffer / CUB "in"
  unsigned int *d_b;     // CUB "out"
  void *d_temp;
  size_t temp_bytes;
  unsigned int n;
};

extern "C" SortCtx *cuda_sort_create(const unsigned int *h_keys, unsigned int n) {
  SortCtx *c = new SortCtx();
  c->n = n;
  size_t bytes = (size_t)n * sizeof(unsigned int);
  CUDA_CHECK(cudaMalloc(&c->d_src, bytes));
  CUDA_CHECK(cudaMalloc(&c->d_a, bytes));
  CUDA_CHECK(cudaMalloc(&c->d_b, bytes));
  CUDA_CHECK(cudaMemcpy(c->d_src, h_keys, bytes, cudaMemcpyHostToDevice));

  c->d_temp = nullptr;
  c->temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, c->temp_bytes, c->d_a, c->d_b,
                                            (int)n));
  CUDA_CHECK(cudaMalloc(&c->d_temp, c->temp_bytes));
  return c;
}

extern "C" void cuda_sort_destroy(SortCtx *c) {
  CUDA_CHECK(cudaFree(c->d_src));
  CUDA_CHECK(cudaFree(c->d_a));
  CUDA_CHECK(cudaFree(c->d_b));
  CUDA_CHECK(cudaFree(c->d_temp));
  delete c;
}

// kind 0 = CUB DeviceRadixSort, kind 1 = Thrust sort.
// Returns the mean milliseconds of one sort.
extern "C" float cuda_sort_bench(SortCtx *c, int kind, int warmup, int iters) {
  size_t bytes = (size_t)c->n * sizeof(unsigned int);
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemcpy(c->d_a, c->d_src, bytes, cudaMemcpyDeviceToDevice));
    if (kind == 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(c->d_temp, c->temp_bytes, c->d_a,
                                                c->d_b, (int)c->n));
    } else {
      thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                   thrust::device_ptr<unsigned int>(c->d_a + c->n));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  float total = 0.0f;
  for (int i = 0; i < iters; ++i) {
    // Restore the unsorted input outside the timed window.
    CUDA_CHECK(cudaMemcpy(c->d_a, c->d_src, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(beg));
    if (kind == 0) {
      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(c->d_temp, c->temp_bytes, c->d_a,
                                                c->d_b, (int)c->n));
    } else {
      thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                   thrust::device_ptr<unsigned int>(c->d_a + c->n));
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    total += ms;
  }
  CUDA_CHECK(cudaEventDestroy(beg));
  CUDA_CHECK(cudaEventDestroy(end));
  return total / (float)iters;
}

// Copy the most recent sorted result back to the host, so the harness can check
// that the baselines and SeGuRu agree.
extern "C" void cuda_sort_copy_out(SortCtx *c, int kind, unsigned int *h_out) {
  size_t bytes = (size_t)c->n * sizeof(unsigned int);
  CUDA_CHECK(cudaMemcpy(c->d_a, c->d_src, bytes, cudaMemcpyDeviceToDevice));
  if (kind == 0) {
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(c->d_temp, c->temp_bytes, c->d_a,
                                              c->d_b, (int)c->n));
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUB may leave the result in either buffer; SortKeys without a
    // DoubleBuffer always writes to d_out.
    CUDA_CHECK(cudaMemcpy(h_out, c->d_b, bytes, cudaMemcpyDeviceToHost));
  } else {
    thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                 thrust::device_ptr<unsigned int>(c->d_a + c->n));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, c->d_a, bytes, cudaMemcpyDeviceToHost));
  }
}
