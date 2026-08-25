//! Safe wrapper around the CUDA C++ reference implementation.
//!
//! This is the only module in the crate containing `unsafe`. It exists solely
//! so the benchmark can compare the SeGuRu kernels against hand-written CUDA
//! that implements exactly the same algorithm and tiling.

use crate::ntt::NttTables;

#[repr(C)]
struct CudaNttCtxRaw {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn cuda_ntt_create(
        n: u32,
        log_n: u32,
        batch: u32,
        q: u64,
        mu: u64,
        bit: u32,
        n_inv: u64,
        n_inv_shoup: u64,
        one_shoup: u64,
        h_data: *const u64,
        h_aux: *const u64,
        wf: *const u64,
        wfs: *const u64,
        wi: *const u64,
        wis: *const u64,
    ) -> *mut CudaNttCtxRaw;
    fn cuda_ntt_destroy(ctx: *mut CudaNttCtxRaw);
    fn cuda_ntt_copy_out(ctx: *mut CudaNttCtxRaw, h_out: *mut u64);
    fn cuda_ntt_reset(ctx: *mut CudaNttCtxRaw, h_data: *const u64);
    fn cuda_ntt_bench(ctx: *mut CudaNttCtxRaw, kind: i32, warmup: i32, iters: i32) -> f32;
}

/// Which CUDA reference kernel (or kernel sequence) to run.
#[derive(Clone, Copy)]
pub enum CudaKernel {
    Forward = 0,
    Inverse = 1,
    PolyAdd = 2,
    PolyMul = 3,
    CipherPlainMul = 4,
}

/// Owns the CUDA-side device buffers of the reference implementation.
pub struct CudaNtt {
    raw: *mut CudaNttCtxRaw,
    elems: usize,
}

impl CudaNtt {
    /// Upload the twiddle tables, the working polynomials `data` and the second
    /// operand `aux` used by the element-wise kernels.
    pub fn new(tables: &NttTables, data: &[u64], aux: &[u64]) -> Self {
        assert_eq!(data.len(), aux.len());
        assert_eq!(data.len() % tables.n, 0);
        let batch = data.len() / tables.n;
        let m = &tables.modulus;
        // SAFETY: every pointer refers to a live host allocation of the length
        // the C side reads: `n * batch` for `h_data`/`h_aux`, `n` for each of
        // the four twiddle tables.
        let raw = unsafe {
            cuda_ntt_create(
                tables.n as u32,
                tables.log_n,
                batch as u32,
                m.q,
                m.mu,
                m.bit,
                tables.n_inv,
                tables.n_inv_shoup,
                m.shoup(1),
                data.as_ptr(),
                aux.as_ptr(),
                tables.w_fwd.as_ptr(),
                tables.w_fwd_shoup.as_ptr(),
                tables.w_inv.as_ptr(),
                tables.w_inv_shoup.as_ptr(),
            )
        };
        assert!(!raw.is_null(), "cuda_ntt_create returned null");
        Self { raw, elems: data.len() }
    }

    /// Mean time of one operation in microseconds (CUDA events, kernel only).
    pub fn bench(&mut self, kind: CudaKernel, warmup: u32, iters: u32) -> f64 {
        // SAFETY: `self.raw` is a valid context for the lifetime of `self`.
        let ms = unsafe { cuda_ntt_bench(self.raw, kind as i32, warmup as i32, iters as i32) };
        ms as f64 * 1000.0
    }

    /// Restore the input buffer, so a timing run does not feed on its own output.
    pub fn reset(&mut self, data: &[u64]) {
        assert_eq!(data.len(), self.elems);
        // SAFETY: `data` holds exactly `elems` words, matching the device buffer.
        unsafe { cuda_ntt_reset(self.raw, data.as_ptr()) };
    }

    /// Copy the buffer holding the most recent result back to the host.
    pub fn output(&mut self) -> Vec<u64> {
        let mut out = vec![0u64; self.elems];
        // SAFETY: `out` has exactly `elems` words, matching the device buffer.
        unsafe { cuda_ntt_copy_out(self.raw, out.as_mut_ptr()) };
        out
    }
}

impl Drop for CudaNtt {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from `cuda_ntt_create` and is dropped once.
        unsafe { cuda_ntt_destroy(self.raw) };
    }
}
