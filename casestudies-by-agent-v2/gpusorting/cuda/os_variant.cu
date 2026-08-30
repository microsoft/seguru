// Same-algorithm CUDA baseline for the SeGuRu **onesweep** sort.
//
// Companion to `drs_variant.cu`, which does the same job for reduce-then-scan.
// Between them the two files give a like-for-like CUDA C++ baseline for each of
// the two algorithms the Rust port implements, so that neither ratio is
// contaminated by a difference of algorithm.
//
// This matters because CUB is not a usable stand-in. CUB is onesweep, but it is
// a heavily tuned production implementation; comparing our first-draft port
// against it measures tuning effort as much as anything else. The kernels here
// are the exact `OneSweep.cu` (MIT, see `upstream/LICENSE`) that the Rust
// `onesweep.rs` transliterates.
//
// Compiled twice by `build.rs`:
//
//   * `OS_DISPATCH=os_dispatch_up`   — upstream tuning, 7680 keys per tile,
//     15 keys per thread, 16 warps.
//   * `OS_DISPATCH=os_dispatch_ours` — our port's tuning, 4096 keys per tile,
//     8 keys per thread, 16 warps.
//
// The `os_dispatch_ours` column is the one to read against the Rust onesweep:
// same algorithm, same tile size, same launch geometry.

#include "upstream/OneSweep.cu"

#include <cstdio>
#include <cstdlib>

#ifndef OS_DISPATCH
#error "define OS_DISPATCH to the exported symbol name"
#endif

#define OS_CAT_(a, b) a##b
#define OS_CAT(a, b) OS_CAT_(a, b)
#define OS_PART_FN OS_CAT(OS_DISPATCH, _part_size)

#define OS_CHECK(expr)                                                         \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,    \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Keys per binning tile, so the caller can size the look-back buffers.
extern "C" unsigned int OS_PART_FN(void) { return BIN_PART_SIZE; }

// One full 4-pass keys-only onesweep sort. Mirrors
// `OneSweepDispatcher::DispatchKernelsKeysOnly`.
//
// Upstream keeps four separate pass histograms; `pass_hist` here is one buffer
// of four consecutive `RADIX * tiles` regions, which is the same layout with a
// stride instead of four pointers. Four passes ping-pong sort -> alt -> sort ->
// alt -> sort, so the result lands back in `sort`.
//
// The clears are part of the algorithm, not setup: a stale FLAG_INCLUSIVE left
// over from a previous sort would let a tile terminate its look-back on garbage.
// They are inside the timed region on both sides for the same reason.
extern "C" void OS_DISPATCH(unsigned int *sort, unsigned int *alt,
                            unsigned int *globalHist, unsigned int *passHist,
                            unsigned int *index, unsigned int size) {
  const unsigned int binBlocks = (size + BIN_PART_SIZE - 1) / BIN_PART_SIZE;
  const unsigned int histBlocks =
      (size + G_HIST_PART_SIZE - 1) / G_HIST_PART_SIZE;
  const unsigned int histThreads = 128;
  const unsigned int binThreads = 512;
  const unsigned int radix = 256;
  const unsigned int passes = 4;

  OS_CHECK(cudaMemsetAsync(index, 0, passes * sizeof(unsigned int)));
  OS_CHECK(cudaMemsetAsync(globalHist, 0, radix * passes * sizeof(unsigned int)));
  // One row per tile *plus one*: `Scan` seeds row 0 with the global digit base
  // tagged FLAG_INCLUSIVE, and tile `i` publishes into row `i + 1`, so the last
  // tile touches row `binBlocks`. Upstream's dispatcher allocates only
  // `binBlocks` rows and gets away with it because each pass has its own
  // allocation and cudaMalloc rounds up; packing the four passes into one buffer
  // makes the missing row collide with the next pass's seed row, which corrupts
  // that pass's look-back. Hence the `+ 1`.
  const size_t stride = (size_t)radix * (binBlocks + 1);
  OS_CHECK(cudaMemsetAsync(passHist, 0, stride * passes * sizeof(unsigned int)));

  unsigned int *ph0 = passHist;
  unsigned int *ph1 = passHist + stride;
  unsigned int *ph2 = passHist + stride * 2;
  unsigned int *ph3 = passHist + stride * 3;

  OneSweep::GlobalHistogram<<<histBlocks, histThreads>>>(sort, globalHist, size);
  OneSweep::Scan<<<passes, radix>>>(globalHist, ph0, ph1, ph2, ph3);
  OneSweep::DigitBinningPassKeysOnly<<<binBlocks, binThreads>>>(sort, alt, ph0,
                                                                index, size, 0);
  OneSweep::DigitBinningPassKeysOnly<<<binBlocks, binThreads>>>(alt, sort, ph1,
                                                                index, size, 8);
  OneSweep::DigitBinningPassKeysOnly<<<binBlocks, binThreads>>>(sort, alt, ph2,
                                                                index, size, 16);
  OneSweep::DigitBinningPassKeysOnly<<<binBlocks, binThreads>>>(alt, sort, ph3,
                                                                index, size, 24);
}
