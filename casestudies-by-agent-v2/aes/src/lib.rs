//! AES-128 ECB on the GPU, written against the SeGuRu `gpu` crate.
//!
//! Design notes (differences from a naive port of the CUDA reference):
//!
//! * **Vectorised I/O.** Plaintext and ciphertext are typed as `[U32_4]`, so a
//!   whole 16-byte AES block moves in a single 128-bit global load/store
//!   instead of four 32-bit accesses.
//! * **Four blocks per thread.** Each thread encrypts `BLOCKS_PER_THREAD`
//!   independent AES blocks. The round computations of the four blocks are
//!   interleaved by the unroller, which hides shared-memory latency without
//!   needing extra occupancy.
//! * **One T-table instead of four.** `TE1..TE3` are byte rotations of `TE0`,
//!   so only 1 KiB of shared memory is staged (versus 4 KiB) and the
//!   prologue costs one load per thread rather than four. The rotations are a
//!   shift/or pair each, which is free next to the shared-memory traffic.
//! * **Round keys in shared memory.** All 44 words are staged once per block
//!   rather than being re-read from global memory on every round.
//! * **Exact, branch-free mapping.** The host pads the device buffers to a
//!   whole number of `grid * block * BLOCKS_PER_THREAD` AES blocks, so the
//!   kernel needs no tail predicate and the `reshape_map!` chunk proves the
//!   writes are disjoint per thread.
//!
//! No `unsafe` appears anywhere in the crate outside of the optional
//! `cuda_ffi` module that binds the CUDA C++ reference used for benchmarking.

pub mod cpu;
pub mod tables;

#[cfg(feature = "bench")]
pub mod cuda_ffi;

use crunchy::unroll;
use gpu::*;

// Re-exported so downstream host code (tests, benchmarks) can name the block
// type without depending on the `gpu` crate directly.
pub use gpu::U32_4;

/// Threads per block. Fixed so the shared-memory staging is a single
/// conflict-free load per thread.
pub const BLOCK_DIM: u32 = 256;

/// AES blocks processed by each thread.
pub const BLOCKS_PER_THREAD: u32 = 4;

/// AES blocks handled by one thread block.
pub const BLOCKS_PER_CTA: u32 = BLOCK_DIM * BLOCKS_PER_THREAD;

#[gpu::device]
#[inline(always)]
fn rotr8(x: u32) -> u32 {
    (x >> 8) | (x << 24)
}

#[gpu::device]
#[inline(always)]
fn rotr16(x: u32) -> u32 {
    (x >> 16) | (x << 16)
}

#[gpu::device]
#[inline(always)]
fn rotr24(x: u32) -> u32 {
    (x >> 24) | (x << 8)
}

/// One AES round column using a single T-table plus rotations.
///
/// `a`, `b`, `c`, `d` are the four state bytes feeding this column, already
/// selected according to `ShiftRows`.
#[gpu::device]
#[inline(always)]
fn round_col(t: &[u32; BLOCK_DIM as usize], a: u32, b: u32, c: u32, d: u32, k: u32) -> u32 {
    t[a as usize] ^ rotr8(t[b as usize]) ^ rotr16(t[c as usize]) ^ rotr24(t[d as usize]) ^ k
}

/// `S(x)` recovered from `TE0`: the second big-endian byte of `TE0[x]`.
#[gpu::device]
#[inline(always)]
fn sbox_from_te(t: &[u32; BLOCK_DIM as usize], x: u32) -> u32 {
    (t[x as usize] >> 16) & 0xff
}

/// AES-128 ECB encryption.
///
/// * `input` / `output`: padded to `grid_dim.x * BLOCK_DIM * BLOCKS_PER_THREAD`
///   AES blocks.
/// * `round_keys`: the 44 encryption round-key words.
/// * `te0`: [`tables::TE0`].
#[gpu::cuda_kernel]
pub fn aes128_encrypt(
    input: &[U32_4],
    output: &mut [U32_4],
    round_keys: &[u32],
    te0: &[u32],
) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    assert!(Config::BDIM_Y == 1);
    assert!(Config::BDIM_Z == 1);

    let tid = thread_id::<DimX>();

    // `round_keys` is staged by the host to BLOCK_DIM words (44 live, rest
    // zero) so every thread stages exactly one word with no divergence.
    let mut te_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
    let mut rk_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
    {
        let mut te_chunk = te_smem.chunk_mut(MapContinuousLinear::new(1));
        te_chunk[0] = te0[tid as usize];
        let mut rk_chunk = rk_smem.chunk_mut(MapContinuousLinear::new(1));
        rk_chunk[0] = round_keys[tid as usize];
    }
    sync_threads();
    let te = &*te_smem;
    let rk = &*rk_smem;

    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + tid;

    // Each thread owns BLOCKS_PER_THREAD slots strided by `nthreads`, which
    // keeps every load and store fully coalesced across the warp.
    let mut out =
        chunk_mut(output, reshape_map!([BLOCKS_PER_THREAD] | [nthreads] => layout: [t0, i0]));

    let mut st = [[0u32; 4]; BLOCKS_PER_THREAD as usize];
    unroll! {
        for k in 0..4 {
            let v = input[(gid + (k as u32) * nthreads) as usize];
            st[k][0] = v[0] ^ rk[0];
            st[k][1] = v[1] ^ rk[1];
            st[k][2] = v[2] ^ rk[2];
            st[k][3] = v[3] ^ rk[3];
        }
    }

    unroll! {
        for r in 0..9 {
            let ko = (r + 1) * 4;
            unroll! {
                for k in 0..4 {
                    let s = st[k];
                    let b0 = s[0] >> 24;
                    let b1 = s[1] >> 24;
                    let b2 = s[2] >> 24;
                    let b3 = s[3] >> 24;
                    let c0 = (s[0] >> 16) & 0xff;
                    let c1 = (s[1] >> 16) & 0xff;
                    let c2 = (s[2] >> 16) & 0xff;
                    let c3 = (s[3] >> 16) & 0xff;
                    let d0 = (s[0] >> 8) & 0xff;
                    let d1 = (s[1] >> 8) & 0xff;
                    let d2 = (s[2] >> 8) & 0xff;
                    let d3 = (s[3] >> 8) & 0xff;
                    let e0 = s[0] & 0xff;
                    let e1 = s[1] & 0xff;
                    let e2 = s[2] & 0xff;
                    let e3 = s[3] & 0xff;
                    st[k] = [
                        round_col(te, b0, c1, d2, e3, rk[ko]),
                        round_col(te, b1, c2, d3, e0, rk[ko + 1]),
                        round_col(te, b2, c3, d0, e1, rk[ko + 2]),
                        round_col(te, b3, c0, d1, e2, rk[ko + 3]),
                    ];
                }
            }
        }
    }

    // Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns).
    unroll! {
        for k in 0..4 {
            let s = st[k];
            let w0 = (sbox_from_te(te, s[0] >> 24) << 24)
                | (sbox_from_te(te, (s[1] >> 16) & 0xff) << 16)
                | (sbox_from_te(te, (s[2] >> 8) & 0xff) << 8)
                | sbox_from_te(te, s[3] & 0xff);
            let w1 = (sbox_from_te(te, s[1] >> 24) << 24)
                | (sbox_from_te(te, (s[2] >> 16) & 0xff) << 16)
                | (sbox_from_te(te, (s[3] >> 8) & 0xff) << 8)
                | sbox_from_te(te, s[0] & 0xff);
            let w2 = (sbox_from_te(te, s[2] >> 24) << 24)
                | (sbox_from_te(te, (s[3] >> 16) & 0xff) << 16)
                | (sbox_from_te(te, (s[0] >> 8) & 0xff) << 8)
                | sbox_from_te(te, s[1] & 0xff);
            let w3 = (sbox_from_te(te, s[3] >> 24) << 24)
                | (sbox_from_te(te, (s[0] >> 16) & 0xff) << 16)
                | (sbox_from_te(te, (s[1] >> 8) & 0xff) << 8)
                | sbox_from_te(te, s[2] & 0xff);
            out[k as u32] =
                U32_4::new([w0 ^ rk[40], w1 ^ rk[41], w2 ^ rk[42], w3 ^ rk[43]]);
        }
    }
}

/// AES-128 ECB decryption (equivalent inverse cipher).
///
/// * `inv_round_keys`: [`tables::inv_round_keys`] applied to the encryption schedule.
/// * `td0`: [`tables::TD0`].
/// * `inv_sbox`: [`tables::INV_SBOX`] widened to `u32` so the final round is a
///   plain shared-memory lookup instead of byte extraction from packed words.
#[gpu::cuda_kernel]
pub fn aes128_decrypt(
    input: &[U32_4],
    output: &mut [U32_4],
    inv_round_keys: &[u32],
    td0: &[u32],
    inv_sbox: &[u32],
) {
    assert!(Config::BDIM_X == BLOCK_DIM);
    assert!(Config::BDIM_Y == 1);
    assert!(Config::BDIM_Z == 1);

    let tid = thread_id::<DimX>();

    let mut td_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
    let mut isb_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
    let mut rk_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
    {
        let mut td_chunk = td_smem.chunk_mut(MapContinuousLinear::new(1));
        td_chunk[0] = td0[tid as usize];
        let mut isb_chunk = isb_smem.chunk_mut(MapContinuousLinear::new(1));
        isb_chunk[0] = inv_sbox[tid as usize];
        let mut rk_chunk = rk_smem.chunk_mut(MapContinuousLinear::new(1));
        rk_chunk[0] = inv_round_keys[tid as usize];
    }
    sync_threads();
    let td = &*td_smem;
    let isb = &*isb_smem;
    let rk = &*rk_smem;

    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let gid = block_id::<DimX>() * Config::BDIM_X + tid;

    let mut out =
        chunk_mut(output, reshape_map!([BLOCKS_PER_THREAD] | [nthreads] => layout: [t0, i0]));

    let mut st = [[0u32; 4]; BLOCKS_PER_THREAD as usize];
    unroll! {
        for k in 0..4 {
            let v = input[(gid + (k as u32) * nthreads) as usize];
            st[k][0] = v[0] ^ rk[40];
            st[k][1] = v[1] ^ rk[41];
            st[k][2] = v[2] ^ rk[42];
            st[k][3] = v[3] ^ rk[43];
        }
    }

    unroll! {
        for r in 0..9 {
            let ko = (9 - r) * 4;
            unroll! {
                for k in 0..4 {
                    let s = st[k];
                    let b0 = s[0] >> 24;
                    let b1 = s[1] >> 24;
                    let b2 = s[2] >> 24;
                    let b3 = s[3] >> 24;
                    let c0 = (s[0] >> 16) & 0xff;
                    let c1 = (s[1] >> 16) & 0xff;
                    let c2 = (s[2] >> 16) & 0xff;
                    let c3 = (s[3] >> 16) & 0xff;
                    let d0 = (s[0] >> 8) & 0xff;
                    let d1 = (s[1] >> 8) & 0xff;
                    let d2 = (s[2] >> 8) & 0xff;
                    let d3 = (s[3] >> 8) & 0xff;
                    let e0 = s[0] & 0xff;
                    let e1 = s[1] & 0xff;
                    let e2 = s[2] & 0xff;
                    let e3 = s[3] & 0xff;
                    // InvShiftRows rotates columns the other way.
                    st[k] = [
                        round_col(td, b0, c3, d2, e1, rk[ko]),
                        round_col(td, b1, c0, d3, e2, rk[ko + 1]),
                        round_col(td, b2, c1, d0, e3, rk[ko + 2]),
                        round_col(td, b3, c2, d1, e0, rk[ko + 3]),
                    ];
                }
            }
        }
    }

    unroll! {
        for k in 0..4 {
            let s = st[k];
            let w0 = (isb[(s[0] >> 24) as usize] << 24)
                | (isb[((s[3] >> 16) & 0xff) as usize] << 16)
                | (isb[((s[2] >> 8) & 0xff) as usize] << 8)
                | isb[(s[1] & 0xff) as usize];
            let w1 = (isb[(s[1] >> 24) as usize] << 24)
                | (isb[((s[0] >> 16) & 0xff) as usize] << 16)
                | (isb[((s[3] >> 8) & 0xff) as usize] << 8)
                | isb[(s[2] & 0xff) as usize];
            let w2 = (isb[(s[2] >> 24) as usize] << 24)
                | (isb[((s[1] >> 16) & 0xff) as usize] << 16)
                | (isb[((s[0] >> 8) & 0xff) as usize] << 8)
                | isb[(s[3] & 0xff) as usize];
            let w3 = (isb[(s[3] >> 24) as usize] << 24)
                | (isb[((s[2] >> 16) & 0xff) as usize] << 16)
                | (isb[((s[1] >> 8) & 0xff) as usize] << 8)
                | isb[(s[0] & 0xff) as usize];
            out[k as u32] = U32_4::new([w0 ^ rk[0], w1 ^ rk[1], w2 ^ rk[2], w3 ^ rk[3]]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/// Number of AES blocks the device buffers must be padded to for `n_blocks`
/// real blocks and a given grid size.
pub fn padded_blocks(n_blocks: usize) -> usize {
    n_blocks.div_ceil(BLOCKS_PER_CTA as usize) * BLOCKS_PER_CTA as usize
}

/// Grid size (in CTAs) used for `n_blocks` AES blocks.
pub fn grid_dim_for(n_blocks: usize) -> u32 {
    n_blocks.div_ceil(BLOCKS_PER_CTA as usize).max(1) as u32
}

/// Convert a byte buffer into padded `U32_4` AES blocks (big-endian words).
pub fn bytes_to_blocks(data: &[u8]) -> Vec<U32_4> {
    assert!(data.len() % 16 == 0, "ECB input must be a multiple of 16 bytes");
    let n = data.len() / 16;
    let mut out = vec![U32_4::default(); padded_blocks(n).max(BLOCKS_PER_CTA as usize)];
    for (i, chunk) in data.chunks_exact(16).enumerate() {
        let mut w = [0u32; 4];
        for (j, word) in chunk.chunks_exact(4).enumerate() {
            w[j] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        out[i] = U32_4::new(w);
    }
    out
}

/// Inverse of [`bytes_to_blocks`], truncated to `n_bytes`.
pub fn blocks_to_bytes(blocks: &[U32_4], n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    for b in blocks.iter() {
        for j in 0..4 {
            out.extend_from_slice(&b[j].to_be_bytes());
        }
        if out.len() >= n_bytes {
            break;
        }
    }
    out.truncate(n_bytes);
    out
}

/// [`tables::INV_SBOX`] widened to `u32` for direct shared-memory indexing.
pub fn inv_sbox_u32() -> Vec<u32> {
    tables::INV_SBOX.iter().map(|&b| b as u32).collect()
}

/// Round keys padded to `BLOCK_DIM` words.
///
/// The kernel stages the schedule with exactly one word per thread; padding the
/// host buffer keeps that load uniform across the CTA instead of predicating
/// the 44 live lanes.
pub fn staged_round_keys(rk: &[u32; 44]) -> Vec<u32> {
    let mut v = vec![0u32; BLOCK_DIM as usize];
    v[..44].copy_from_slice(rk);
    v
}

#[cfg(test)]
mod tests;
