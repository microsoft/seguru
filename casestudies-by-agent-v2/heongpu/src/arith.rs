//! Element-wise ciphertext arithmetic modulo `q`.
//!
//! A ciphertext is a flat `[u64]` holding two length-`N` polynomials in RNS
//! form; every operation here is coefficient-wise, so the kernels are plain
//! streaming kernels: `ELEMS_PER_THREAD` independent elements per thread,
//! strided by the grid so every access is coalesced, and the host pads the
//! buffers to a whole number of CTA tiles so no kernel needs a tail predicate.

use crunchy::unroll;
use gpu::*;

use crate::modular::{Modulus, add_mod, mul_mod, mul_mod_shoup, neg_mod, sub_mod};

/// Threads per block for the element-wise kernels.
pub const BLOCK_DIM: u32 = 256;
/// Elements handled by one thread.
pub const ELEMS_PER_THREAD: u32 = 4;
/// Elements handled by one CTA.
pub const ELEMS_PER_CTA: u32 = BLOCK_DIM * ELEMS_PER_THREAD;

#[gpu::cuda_kernel]
pub fn poly_add(a: &[u64], b: &[u64], out: &mut [u64], q: u64) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            o[k as u32] = add_mod(a[i], b[i], q);
        }
    }
}

#[gpu::cuda_kernel]
pub fn poly_sub(a: &[u64], b: &[u64], out: &mut [u64], q: u64) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            o[k as u32] = sub_mod(a[i], b[i], q);
        }
    }
}

#[gpu::cuda_kernel]
pub fn poly_neg(a: &[u64], out: &mut [u64], q: u64) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            o[k as u32] = neg_mod(a[i], q);
        }
    }
}

/// Coefficient-wise modular product of two vectors (Barrett).
#[gpu::cuda_kernel]
pub fn poly_mul(a: &[u64], b: &[u64], out: &mut [u64], q: u64, mu: u64, bit: u32) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            o[k as u32] = mul_mod(a[i], b[i], q, mu, bit);
        }
    }
}

/// Ciphertext times plaintext in the NTT domain: every length-`n` polynomial of
/// the ciphertext is multiplied coefficient-wise by the same plaintext.
///
/// `n` must be a power of two; the wrap-around is a mask, not a modulo.
#[gpu::cuda_kernel]
pub fn cipher_plain_mul(
    c: &[u64],
    p: &[u64],
    out: &mut [u64],
    n_mask: u32,
    q: u64,
    mu: u64,
    bit: u32,
) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = gid + (k as u32) * nthreads;
            o[k as u32] = mul_mod(c[i as usize], p[(i & n_mask) as usize], q, mu, bit);
        }
    }
}

/// Multiply a vector by a compile-time-known scalar using Shoup's method.
#[gpu::cuda_kernel]
pub fn poly_mul_scalar(a: &[u64], out: &mut [u64], w: u64, w_shoup: u64, q: u64) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([ELEMS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
    unroll! {
        for k in 0..4 {
            let i = (gid + (k as u32) * nthreads) as usize;
            o[k as u32] = mul_mod_shoup(a[i], w, w_shoup, q);
        }
    }
}

// ---------------------------------------------------------------------------
// Host drivers
// ---------------------------------------------------------------------------

/// Length rounded up to a whole number of CTA tiles.
pub fn padded_len(len: usize) -> usize {
    len.div_ceil(ELEMS_PER_CTA as usize) * ELEMS_PER_CTA as usize
}

/// Grid size (CTAs) covering `len` elements.
pub fn grid_for(len: usize) -> u32 {
    len.div_ceil(ELEMS_PER_CTA as usize).max(1) as u32
}

fn padded(v: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; padded_len(v.len()).max(ELEMS_PER_CTA as usize)];
    out[..v.len()].copy_from_slice(v);
    out
}

/// Which binary element-wise operation a host driver should run.
#[derive(Clone, Copy, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

/// Run one binary element-wise op on the GPU and return the result.
pub fn run_binary(a: &[u64], b: &[u64], op: BinOp, m: &Modulus) -> Vec<u64> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let ha = padded(a);
    let hb = padded(b);
    let grid = grid_for(len);
    let mut host_out = vec![0u64; ha.len()];

    gpu_host::cuda_ctx(0, |ctx, md| {
        let d_a = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let d_b = ctx.new_tensor_view(hb.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(host_out.as_slice()).unwrap();
        match op {
            BinOp::Add => {
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                poly_add::launch(cfg, ctx, md, &d_a, &d_b, &mut d_out, m.q).unwrap();
            }
            BinOp::Sub => {
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                poly_sub::launch(cfg, ctx, md, &d_a, &d_b, &mut d_out, m.q).unwrap();
            }
            BinOp::Mul => {
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                poly_mul::launch(cfg, ctx, md, &d_a, &d_b, &mut d_out, m.q, m.mu, m.bit).unwrap();
            }
        }
        d_out.copy_to_host(&mut host_out).unwrap();
    });
    host_out.truncate(len);
    host_out
}

/// Negate every coefficient on the GPU.
pub fn run_neg(a: &[u64], m: &Modulus) -> Vec<u64> {
    let len = a.len();
    let ha = padded(a);
    let grid = grid_for(len);
    let mut host_out = vec![0u64; ha.len()];
    gpu_host::cuda_ctx(0, |ctx, md| {
        let d_a = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(host_out.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
        poly_neg::launch(cfg, ctx, md, &d_a, &mut d_out, m.q).unwrap();
        d_out.copy_to_host(&mut host_out).unwrap();
    });
    host_out.truncate(len);
    host_out
}

/// Multiply every coefficient by `s` on the GPU (Shoup).
pub fn run_scalar_mul(a: &[u64], s: u64, m: &Modulus) -> Vec<u64> {
    let len = a.len();
    let ha = padded(a);
    let grid = grid_for(len);
    let s_shoup = m.shoup(s);
    let mut host_out = vec![0u64; ha.len()];
    gpu_host::cuda_ctx(0, |ctx, md| {
        let d_a = ctx.new_tensor_view(ha.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(host_out.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
        poly_mul_scalar::launch(cfg, ctx, md, &d_a, &mut d_out, s, s_shoup, m.q).unwrap();
        d_out.copy_to_host(&mut host_out).unwrap();
    });
    host_out.truncate(len);
    host_out
}

/// Ciphertext (`2 * n` coefficients) times a length-`n` plaintext.
pub fn run_cipher_plain_mul(c: &[u64], p: &[u64], m: &Modulus) -> Vec<u64> {
    let n = p.len();
    assert!(n.is_power_of_two(), "ring size must be a power of two");
    assert_eq!(c.len() % n, 0);
    let len = c.len();
    let hc = padded(c);
    let hp = padded(p);
    let grid = grid_for(len);
    let mut host_out = vec![0u64; hc.len()];
    gpu_host::cuda_ctx(0, |ctx, md| {
        let d_c = ctx.new_tensor_view(hc.as_slice()).unwrap();
        let d_p = ctx.new_tensor_view(hp.as_slice()).unwrap();
        let mut d_out = ctx.new_tensor_view(host_out.as_slice()).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
        cipher_plain_mul::launch(
            cfg,
            ctx,
            md,
            &d_c,
            &d_p,
            &mut d_out,
            (n - 1) as u32,
            m.q,
            m.mu,
            m.bit,
        )
        .unwrap();
        d_out.copy_to_host(&mut host_out).unwrap();
    });
    host_out.truncate(len);
    host_out
}
