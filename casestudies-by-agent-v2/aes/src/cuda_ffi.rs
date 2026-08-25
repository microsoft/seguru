//! Safe wrapper around the CUDA C++ reference implementation.
//!
//! This is the only module in the crate that contains `unsafe`, and it exists
//! solely so the benchmark can compare against hand-written CUDA. The SeGuRu
//! kernels and their host code are entirely safe Rust.

use crate::{BLOCK_DIM, U32_4, tables};

#[repr(C)]
struct CudaAesCtxRaw {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn cuda_aes_create(
        h_in: *const u32,
        padded_blocks: u32,
        te0: *const u32,
        td0: *const u32,
        isb: *const u32,
        enc_rk: *const u32,
        dec_rk: *const u32,
        te_all: *const u32,
    ) -> *mut CudaAesCtxRaw;
    fn cuda_aes_destroy(ctx: *mut CudaAesCtxRaw);
    fn cuda_aes_copy_out(ctx: *mut CudaAesCtxRaw, h_out: *mut u32);
    fn cuda_aes_bench(ctx: *mut CudaAesCtxRaw, kind: i32, warmup: i32, iters: i32) -> f32;
}

/// Which CUDA reference kernel to time.
#[derive(Clone, Copy)]
pub enum CudaKernel {
    /// Mirror of the SeGuRu encrypt kernel.
    EncryptOpt = 0,
    /// Mirror of the SeGuRu decrypt kernel.
    DecryptOpt = 1,
    /// Textbook one-block-per-thread encrypt with `__constant__` T-tables.
    EncryptClassic = 2,
}

/// Owns the CUDA-side device buffers for the reference implementation.
pub struct CudaAes {
    raw: *mut CudaAesCtxRaw,
    padded_blocks: usize,
}

impl CudaAes {
    /// Upload `input` (already padded to a whole number of CTA tiles) plus the
    /// tables and round keys used by the reference kernels.
    pub fn new(input: &[U32_4], enc_rk: &[u32; 44], dec_rk: &[u32; 44]) -> Self {
        let te0: Vec<u32> = pad_to_block_dim(&tables::TE0);
        let td0: Vec<u32> = pad_to_block_dim(&tables::TD0);
        let isb = crate::inv_sbox_u32();
        let enc_staged = crate::staged_round_keys(enc_rk);
        let dec_staged = crate::staged_round_keys(dec_rk);
        let te_all = full_te_tables();
        let flat = flatten_blocks(input);

        // SAFETY: every pointer refers to a live host allocation of at least the
        // length the C side reads (`padded_blocks * 4` words for the input,
        // `BLOCK_DIM` words for each table, `1024` words for `te_all`).
        let raw = unsafe {
            cuda_aes_create(
                flat.as_ptr(),
                input.len() as u32,
                te0.as_ptr(),
                td0.as_ptr(),
                isb.as_ptr(),
                enc_staged.as_ptr(),
                dec_staged.as_ptr(),
                te_all.as_ptr(),
            )
        };
        assert!(!raw.is_null(), "cuda_aes_create returned null");
        Self { raw, padded_blocks: input.len() }
    }

    /// Mean kernel time in microseconds.
    pub fn bench(&mut self, kind: CudaKernel, warmup: u32, iters: u32) -> f64 {
        // SAFETY: `self.raw` is a valid context for the lifetime of `self`.
        let ms = unsafe { cuda_aes_bench(self.raw, kind as i32, warmup as i32, iters as i32) };
        ms as f64 * 1000.0
    }

    /// Copy the device output buffer back to the host.
    pub fn output(&mut self) -> Vec<U32_4> {
        let mut flat = vec![0u32; self.padded_blocks * 4];
        // SAFETY: `flat` has exactly `padded_blocks * 4` words, matching the
        // device buffer allocated in `cuda_aes_create`.
        unsafe { cuda_aes_copy_out(self.raw, flat.as_mut_ptr()) };
        flat.chunks_exact(4).map(|c| U32_4::new([c[0], c[1], c[2], c[3]])).collect()
    }
}

impl Drop for CudaAes {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was produced by `cuda_aes_create` and is dropped once.
        unsafe { cuda_aes_destroy(self.raw) };
    }
}

fn pad_to_block_dim(t: &[u32; 256]) -> Vec<u32> {
    let mut v = vec![0u32; BLOCK_DIM as usize];
    let n = t.len().min(v.len());
    v[..n].copy_from_slice(&t[..n]);
    v
}

fn flatten_blocks(blocks: &[U32_4]) -> Vec<u32> {
    let mut v = Vec::with_capacity(blocks.len() * 4);
    for b in blocks {
        for j in 0..4 {
            v.push(b[j]);
        }
    }
    v
}

/// `[TE0 | TE1 | TE2 | TE3]` for the classic `__constant__`-memory kernel.
fn full_te_tables() -> Vec<u32> {
    let mut v = Vec::with_capacity(1024);
    v.extend_from_slice(&tables::TE0);
    v.extend(tables::TE0.iter().map(|x| x.rotate_right(8)));
    v.extend(tables::TE0.iter().map(|x| x.rotate_right(16)));
    v.extend(tables::TE0.iter().map(|x| x.rotate_right(24)));
    v
}
