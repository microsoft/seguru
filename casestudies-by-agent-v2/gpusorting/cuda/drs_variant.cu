// Same-algorithm CUDA baseline for the SeGuRu radix sort.
//
// CUB is *not* a like-for-like comparison: on CUDA 13.3 / sm_80 it dispatches
// `DeviceRadixSortOnesweepKernel`, i.e. onesweep with decoupled look-back. Our
// Rust port is a transliteration of Thomas Smith's reduce-then-scan
// `DeviceRadixSort.cu` (MIT, https://github.com/b0nes164/GPUSorting), which is a
// different algorithm. Measuring SeGuRu against CUB therefore measures the
// choice of algorithm, not the cost of safety.
//
// This translation unit compiles the *unmodified upstream kernels* (vendored in
// `upstream/`, MIT, see `upstream/LICENSE`) and drives them with the same kernel
// sequence the upstream dispatcher uses, so the only difference against our Rust
// port is the compiler and the safety checks.
//
// It is compiled twice by `build.rs` with different tuning macros, so we can
// separate two effects that would otherwise be confounded:
//
//   * `DRS_DISPATCH=drs_dispatch_up`  — upstream tuning, 7680 keys per tile,
//     15 keys per thread. The reference implementation as its author tuned it.
//   * `DRS_DISPATCH=drs_dispatch_ours` — our port's tuning, 4096 keys per tile,
//     8 keys per thread.
//
// The gap between those two is a tuning cost that has nothing to do with SeGuRu;
// the gap between `drs_dispatch_ours` and the Rust port is the SeGuRu cost.

#include "upstream/DeviceRadixSort.cu"

#include <cstdio>
#include <cstdlib>

#ifndef DRS_DISPATCH
#error "define DRS_DISPATCH to the exported symbol name"
#endif

#define DRS_CAT_(a, b) a##b
#define DRS_CAT(a, b) DRS_CAT_(a, b)
#define DRS_PART_FN DRS_CAT(DRS_DISPATCH, _part_size)

#define DRS_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,    \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Keys per partition tile, so the caller can size `passHist` correctly.
extern "C" unsigned int DRS_PART_FN(void) { return PART_SIZE; }

// One full 4-pass keys-only sort. Mirrors
// `DeviceRadixSortDispatcher::DispatchKernelsKeysOnly`. Four passes ping-pong
// sort -> alt -> sort -> alt -> sort, so the result lands back in `sort`.
extern "C" void DRS_DISPATCH(unsigned int *sort, unsigned int *alt,
                             unsigned int *globalHist, unsigned int *passHist,
                             unsigned int size) {
  const unsigned int threadblocks = (size + PART_SIZE - 1) / PART_SIZE;
  const unsigned int upsweepThreads = 128;
  const unsigned int scanThreads = 128;
  const unsigned int downsweepThreads = 512;

  DRS_CHECK(cudaMemsetAsync(globalHist, 0, RADIX * 4 * sizeof(unsigned int)));

  for (unsigned int pass = 0; pass < 4; ++pass) {
    const unsigned int shift = pass * 8;
    unsigned int *in = (pass & 1) ? alt : sort;
    unsigned int *out = (pass & 1) ? sort : alt;
    DeviceRadixSort::Upsweep<<<threadblocks, upsweepThreads>>>(
        in, globalHist, passHist, size, shift);
    DeviceRadixSort::Scan<<<RADIX, scanThreads>>>(passHist, threadblocks);
    DeviceRadixSort::DownsweepKeysOnly<<<threadblocks, downsweepThreads>>>(
        in, out, globalHist, passHist, size, shift);
  }
}
