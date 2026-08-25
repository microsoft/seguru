//! Utility kernel: zero a `u32` buffer on the device.
//!
//! `pass_hist` is written non-atomically by the upsweep for the `thread_blocks`
//! live slots only, but `radix_scan` reads all `padded_thread_blocks` slots, so
//! the padding has to start at zero on every pass. Doing that with a kernel keeps
//! the host out of the inner loop.

use crunchy::unroll;
use gpu::prelude::*;

pub const CLEAR_THREADS: u32 = 256;
pub const CLEAR_PER_THREAD: u32 = 4;

#[gpu::cuda_kernel]
pub fn clear_u32(buf: &mut [u32]) {
    assert!(Config::BDIM_X == CLEAR_THREADS);
    let mut c = chunk_mut(
        buf,
        reshape_map!([CLEAR_PER_THREAD] | [CLEAR_THREADS, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
    );
    unroll! {
        for k in 0..4 {
            c[k as u32] = 0u32;
        }
    }
}

/// Grid size needed to clear `len` elements. `len` must be a multiple of
/// `CLEAR_THREADS * CLEAR_PER_THREAD`.
pub fn clear_grid(len: usize) -> u32 {
    let unit = CLEAR_THREADS * CLEAR_PER_THREAD;
    (len as u32).div_ceil(unit)
}

/// Round `len` up to the granularity `clear_u32` requires.
pub fn clear_padded_len(len: usize) -> usize {
    let unit = (CLEAR_THREADS * CLEAR_PER_THREAD) as usize;
    len.div_ceil(unit) * unit
}
