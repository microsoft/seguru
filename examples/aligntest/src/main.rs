//! Regression example for 64-bit global memory access width.
//!
//! Both kernels move `u64` values through global memory. Each access must
//! lower to a single `ld.global.u64`/`st.global.u64`. When the MLIR module
//! carries no data layout, translation falls back to LLVM's default (where
//! `i64` has an ABI alignment of 4) and the NVPTX backend splits every access
//! into a pair of 32-bit ones. Inspect the generated PTX to check:
//!
//! ```text
//! grep -oE '(ld|st)\.global[a-z0-9._]*' <crate>gpu.gpu.ptx | sort | uniq -c
//! ```

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use crunchy::unroll;
use gpu::prelude::*;

const THREADS: u32 = 256;
const PER_THREAD: usize = 8;

/// One `u64` per thread, addressed directly.
#[gpu::cuda_kernel]
pub fn scalar_u64(src: &[u64], out: &mut [u64]) {
    let n = grid_dim::<DimX>() * Config::BDIM_X;
    let i = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([1] | [n] => layout: [i0, t0]));
    o[0u32] = src[i as usize] + 1;
}

/// Eight contiguous `u64` per thread, written through a chunk.
#[gpu::cuda_kernel]
pub fn chunked_u64(src: &[u64], out: &mut [u64]) {
    let n = grid_dim::<DimX>() * Config::BDIM_X;
    let tid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut v = [0u64; 8];
    unroll! {
        for j in 0..8 {
            v[j] = src[(tid * 8 + (j as u32)) as usize];
        }
    }
    let mut o = chunk_mut(out, reshape_map!([8] | [n] => layout: [i0, t0]));
    unroll! {
        for j in 0..8 {
            o[j as u32] = v[j] + 1;
        }
    }
}

fn main() {
    let n = THREADS as usize * PER_THREAD;
    let input: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();

    gpu_host::cuda_ctx(0, |ctx, m| {
        let src = ctx.new_tensor_view(input.as_slice()).unwrap();
        let mut out = ctx.new_tensor_view(vec![0u64; n].as_slice()).unwrap();

        let cfg = gpu_host::gpu_config!(1, 1, 1, @const THREADS, 1, 1, 0);
        scalar_u64::launch(cfg, ctx, m, &src, &mut out).unwrap();
        let mut got = vec![0u64; n];
        out.copy_to_host(got.as_mut_slice()).unwrap();
        for i in 0..THREADS as usize {
            assert_eq!(got[i], input[i].wrapping_add(1), "scalar_u64 mismatch at {i}");
        }

        let cfg = gpu_host::gpu_config!(1, 1, 1, @const THREADS, 1, 1, 0);
        chunked_u64::launch(cfg, ctx, m, &src, &mut out).unwrap();
        out.copy_to_host(got.as_mut_slice()).unwrap();
        for i in 0..n {
            assert_eq!(got[i], input[i].wrapping_add(1), "chunked_u64 mismatch at {i}");
        }
    });

    println!("ok: {n} u64 elements round-tripped through both kernels");
}
