//! Safe wrapper around the CUDA C++ reference baselines.
//!
//! This is the only module in the crate that contains `unsafe`, and it exists
//! solely so the benchmark can compare against vendor-tuned CUDA. The SeGuRu
//! kernels and their host code are entirely safe Rust.

#[repr(C)]
struct SortCtxRaw {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn cuda_sort_create(h_keys: *const u32, n: u32) -> *mut SortCtxRaw;
    fn cuda_sort_destroy(ctx: *mut SortCtxRaw);
    fn cuda_sort_bench(ctx: *mut SortCtxRaw, kind: i32, warmup: i32, iters: i32) -> f32;
    fn cuda_sort_copy_out(ctx: *mut SortCtxRaw, kind: i32, h_out: *mut u32);
}

/// Which CUDA baseline to time.
///
/// Only [`CudaSort::DrsOurTuning`] is a like-for-like comparison against the
/// SeGuRu kernels. CUB is a *different algorithm* (onesweep with decoupled
/// look-back on CUDA 13.3 / sm_80), so `SeGuRu / CUB` measures the choice of
/// algorithm rather than the cost of safety.
#[derive(Clone, Copy)]
pub enum CudaSort {
    /// `cub::DeviceRadixSort::SortKeys`. Different algorithm — see above.
    Cub = 0,
    /// `thrust::sort`.
    Thrust = 1,
    /// The upstream reduce-then-scan `DeviceRadixSort.cu` our kernels are a
    /// transliteration of, at upstream's own tuning (7680 keys per tile,
    /// 15 keys per thread).
    DrsUpstreamTuning = 2,
    /// The same upstream kernels rebuilt at our port's tuning (4096 keys per
    /// tile, 8 keys per thread). Same algorithm *and* same tuning as the Rust
    /// code, so this ratio isolates the cost of SeGuRu.
    DrsOurTuning = 3,
}

pub struct CudaSorter {
    raw: *mut SortCtxRaw,
    n: usize,
}

impl CudaSorter {
    pub fn new(keys: &[u32]) -> Self {
        let raw = unsafe { cuda_sort_create(keys.as_ptr(), keys.len() as u32) };
        assert!(!raw.is_null(), "cuda_sort_create returned null");
        Self { raw, n: keys.len() }
    }

    /// Mean milliseconds of one sort, measured with CUDA events.
    pub fn bench(&self, kind: CudaSort, warmup: u32, iters: u32) -> f64 {
        unsafe { cuda_sort_bench(self.raw, kind as i32, warmup as i32, iters as i32) as f64 }
    }

    pub fn sorted(&self, kind: CudaSort) -> Vec<u32> {
        let mut out = vec![0u32; self.n];
        unsafe { cuda_sort_copy_out(self.raw, kind as i32, out.as_mut_ptr()) };
        out
    }
}

impl Drop for CudaSorter {
    fn drop(&mut self) {
        unsafe { cuda_sort_destroy(self.raw) };
    }
}
