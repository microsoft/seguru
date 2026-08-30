// CUDA reference baselines for the SeGuRu radix sort benchmark.
//
// Four baselines are provided:
//   * CUB `DeviceRadixSort::SortKeys` — NVIDIA's production sort. Note this is
//     *not* the same algorithm: on CUDA 13.3 / sm_80 CUB dispatches
//     `DeviceRadixSortOnesweepKernel`, i.e. onesweep with decoupled look-back.
//   * Thrust `sort` — the convenience API most applications actually reach for.
//   * The vendored upstream reduce-then-scan `DeviceRadixSort.cu`, which is the
//     file our Rust kernels are a transliteration of, at two tunings: upstream's
//     own (7680 keys/tile) and ours (4096 keys/tile). These are the like-for-like
//     bar, and the only one from which a cost of safety can be read off.
//
// Timings are kernel-only: allocation, the temporary-storage query and the host
// transfers all happen outside the timed region.

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cstdlib>

// Same-algorithm baselines, compiled from cuda/drs_variant.cu.
extern "C" unsigned int drs_dispatch_up_part_size(void);
extern "C" unsigned int drs_dispatch_ours_part_size(void);
extern "C" void drs_dispatch_up(unsigned int *, unsigned int *, unsigned int *,
                                unsigned int *, unsigned int);
extern "C" void drs_dispatch_ours(unsigned int *, unsigned int *, unsigned int *,
                                  unsigned int *, unsigned int);

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
  // Scratch for the same-algorithm baselines. `d_pass_hist` is sized for the
  // smaller tile, which needs more thread blocks and so a larger histogram.
  unsigned int *d_global_hist;
  unsigned int *d_pass_hist;
};

// Run one same-algorithm sort. Result lands back in `c->d_a`.
static void drs_run(SortCtx *c, int kind) {
  if (kind == 2) {
    drs_dispatch_up(c->d_a, c->d_b, c->d_global_hist, c->d_pass_hist, c->n);
  } else {
    drs_dispatch_ours(c->d_a, c->d_b, c->d_global_hist, c->d_pass_hist, c->n);
  }
}

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

  unsigned int part_up = drs_dispatch_up_part_size();
  unsigned int part_ours = drs_dispatch_ours_part_size();
  unsigned int part_min = part_up < part_ours ? part_up : part_ours;
  unsigned int max_blocks = (n + part_min - 1) / part_min;
  CUDA_CHECK(cudaMalloc(&c->d_global_hist, 256 * 4 * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&c->d_pass_hist,
                        (size_t)max_blocks * 256 * sizeof(unsigned int)));
  return c;
}

extern "C" void cuda_sort_destroy(SortCtx *c) {
  CUDA_CHECK(cudaFree(c->d_src));
  CUDA_CHECK(cudaFree(c->d_a));
  CUDA_CHECK(cudaFree(c->d_b));
  CUDA_CHECK(cudaFree(c->d_temp));
  CUDA_CHECK(cudaFree(c->d_global_hist));
  CUDA_CHECK(cudaFree(c->d_pass_hist));
  delete c;
}

// kind 0 = CUB, 1 = Thrust, 2 = upstream DeviceRadixSort at upstream tuning
// (7680 keys/tile), 3 = the same kernels at our port's tuning (4096 keys/tile).
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
    } else if (kind == 1) {
      thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                   thrust::device_ptr<unsigned int>(c->d_a + c->n));
    } else {
      drs_run(c, kind);
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
    } else if (kind == 1) {
      thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                   thrust::device_ptr<unsigned int>(c->d_a + c->n));
    } else {
      drs_run(c, kind);
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
  } else if (kind >= 2) {
    // Four passes ping-pong, so the sorted keys end up back in d_a.
    drs_run(c, kind);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, c->d_a, bytes, cudaMemcpyDeviceToHost));
  } else {
    thrust::sort(thrust::device_ptr<unsigned int>(c->d_a),
                 thrust::device_ptr<unsigned int>(c->d_a + c->n));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, c->d_a, bytes, cudaMemcpyDeviceToHost));
  }
}
