//! Negacyclic number-theoretic transform over Z_q for power-of-two ring sizes.
//!
//! # Algorithm
//!
//! * **Forward** — Cooley-Tukey decimation-in-time (`psi_rev` twiddles,
//!   natural-order input, bit-reversed-order output).
//! * **Inverse** — Gentleman-Sande decimation-in-frequency (bit-reversed input,
//!   natural output, `n^-1` folded into the last stage).
//!
//! The pair composes to the identity with no explicit bit-reversal permutation,
//! which is the standard Longa-Naehrig formulation used by HE libraries.
//!
//! # Mapping onto the GPU
//!
//! The transform is split so that as much as possible is **shared-memory
//! resident**:
//!
//! * [`ntt_forward_tile`] / [`ntt_inverse_tile`] hold a [`TILE`] = 4096
//!   coefficient chunk (32 KiB) in `GpuShared` and run **12 butterfly stages**
//!   with only three block-wide barriers in between. 512 threads each own 8
//!   coefficients, so every group of three stages is a radix-8 sub-transform
//!   done entirely in registers, and shared memory is touched once per three
//!   stages rather than once per stage.
//! * The tile kernel's per-round gather patterns are exactly the reshapes
//!   `pos = t_lo + j*LOW + t_hi*8*LOW` with `LOW` = 512, 64, 8, 1, which is
//!   what `reshape_map!` expresses natively, so every shared-memory write is
//!   proven disjoint at compile time. Reads are ordinary slice reads.
//! * Only the remaining `log2(N) - 12` stages — at most four, for N up to
//!   65536 — have butterfly distances larger than a tile. Those are run as
//!   separate global passes ([`ntt_stage_forward`] / [`ntt_stage_inverse`]),
//!   four butterflies per thread.
//!
//! For N = 4096 the entire transform is a single kernel with one global read
//! and one global write.
//!
//! All butterflies use Shoup multiplication: the twiddle tables carry both `w`
//! and `floor(w * 2^64 / q)`, so a butterfly is two 64-bit multiplies, a
//! multiply-high and a conditional subtraction.

use crunchy::unroll;
use gpu::*;

use crate::modular::{Modulus, add_mod, mul_mod_shoup, sub_mod};

/// Coefficients held in shared memory by one CTA.
pub const TILE: u32 = 4096;
/// Threads per CTA in the NTT kernels.
pub const NTT_BDIM: u32 = 512;
/// Coefficients owned by one thread (`TILE / NTT_BDIM`).
pub const EPT: u32 = 8;
/// Stages performed by one tile kernel (`log2(TILE)`).
pub const TILE_STAGES: u32 = 12;

/// Length the device twiddle tables are padded to.
///
/// Every twiddle index this crate generates is `< N <= 65536` (see the maps in
/// the tile kernels), so masking with `WMASK` is a *no-op on the data*. Its
/// purpose is to hand LLVM a compile-time-provable range fact: combined with
/// the constant-length subslice `&w[..WMAX]` taken at the top of each kernel,
/// `idx & WMASK < WMAX == w.len()` folds every per-access bounds check away.
/// Same trick as `polybench/src/bin/boundscheck.rs`, and still safe Rust.
pub const WMAX: usize = 65536;
const WMASK: u32 = (WMAX - 1) as u32;

// ---------------------------------------------------------------------------
// Register-resident radix-8 sub-transforms
// ---------------------------------------------------------------------------

/// Fetch the twiddle pair `(w, w_shoup)` at index `$i` in one 128-bit load.
///
/// This is a `macro_rules!` and not a `#[gpu::device] fn` on purpose. SeGuRu
/// only emits `ld.global.v4.u32` for a `[U32_4]` index expression that appears
/// *textually in the kernel body*; the identical expression inside an
/// `#[inline(always)] #[gpu::device]` helper is lowered as four scalar
/// `ld.global.u32`. See `README.md`, experiment C. A macro expands before
/// codegen, so the access is a kernel-body access and does vectorise.
///
/// `& WMASK` is a no-op on the data (every twiddle index this crate generates
/// is `< N <= WMAX`) that lets LLVM discharge the bounds check; see [`WMAX`].
macro_rules! twiddle {
    ($wp:expr, $i:expr) => {{
        let p = $wp[(($i) & WMASK) as usize];
        ((p[0] as u64) | ((p[1] as u64) << 32), (p[2] as u64) | ((p[3] as u64) << 32))
    }};
}

/// Three Cooley-Tukey stages on eight register-resident coefficients.
///
/// `$h0` is the twiddle index of the first stage; the second and third stages
/// use `2*h0 + ..` and `4*h0 + ..`, which is the block index doubling as the
/// transform splits.
macro_rules! fwd_radix8 {
    ($v:expr, $wp:expr, $q:expr, $h0:expr) => {{
        let mut v = $v;
        let q = $q;
        let h0 = $h0;
        let (s, sp) = twiddle!($wp, h0);
        unroll! {
            for j in 0..4 {
                let u = v[j];
                let t = mul_mod_shoup(v[j + 4], s, sp, q);
                v[j] = add_mod(u, t, q);
                v[j + 4] = sub_mod(u, t, q);
            }
        }
        unroll! {
            for g in 0..2 {
                let (s, sp) = twiddle!($wp, 2 * h0 + g as u32);
                unroll! {
                    for jj in 0..2 {
                        let a = g * 4 + jj;
                        let u = v[a];
                        let t = mul_mod_shoup(v[a + 2], s, sp, q);
                        v[a] = add_mod(u, t, q);
                        v[a + 2] = sub_mod(u, t, q);
                    }
                }
            }
        }
        unroll! {
            for g in 0..4 {
                let (s, sp) = twiddle!($wp, 4 * h0 + g as u32);
                let a = g * 2;
                let u = v[a];
                let t = mul_mod_shoup(v[a + 1], s, sp, q);
                v[a] = add_mod(u, t, q);
                v[a + 1] = sub_mod(u, t, q);
            }
        }
        v
    }};
}

/// Three Gentleman-Sande stages on eight register-resident coefficients.
///
/// `$a0` is the twiddle index base of the first stage; the following stages use
/// `a0 >> 1` and `a0 >> 2` as the transform merges.
macro_rules! inv_radix8 {
    ($v:expr, $wp:expr, $q:expr, $a0:expr) => {{
        let mut v = $v;
        let q = $q;
        let a0 = $a0;
        unroll! {
            for g in 0..4 {
                let (s, sp) = twiddle!($wp, a0 + g as u32);
                let a = g * 2;
                let u = v[a];
                let t = v[a + 1];
                v[a] = add_mod(u, t, q);
                v[a + 1] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
            }
        }
        let a1 = a0 >> 1;
        unroll! {
            for g in 0..2 {
                let (s, sp) = twiddle!($wp, a1 + g as u32);
                unroll! {
                    for jj in 0..2 {
                        let a = g * 4 + jj;
                        let u = v[a];
                        let t = v[a + 2];
                        v[a] = add_mod(u, t, q);
                        v[a + 2] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
                    }
                }
            }
        }
        let a2 = a0 >> 2;
        let (s, sp) = twiddle!($wp, a2);
        unroll! {
            for j in 0..4 {
                let u = v[j];
                let t = v[j + 4];
                v[j] = add_mod(u, t, q);
                v[j + 4] = mul_mod_shoup(sub_mod(u, t, q), s, sp, q);
            }
        }
        v
    }};
}

// ---------------------------------------------------------------------------
// Shared-memory-resident tile kernels (12 stages each)
// ---------------------------------------------------------------------------

/// Last 12 forward stages: one 4096-coefficient tile per CTA, shared resident.
///
/// `m0 = N / TILE` is the number of blocks the polynomial has already been
/// split into by the preceding global passes.
#[gpu::cuda_kernel]
pub fn ntt_forward_tile(inp: &[u64], out: &mut [u64], wp: &[U32_4], q: u64, m0: u32) {
    assert!(Config::BDIM_X == NTT_BDIM);
    let wp = &wp[..WMAX];
    let tid = thread_id::<DimX>();
    let blk = block_id::<DimX>();
    let tile_base = blk * TILE;
    let base = m0 + (blk & (m0 - 1));

    let mut smem = GpuShared::<[u64; TILE as usize]>::zero();

    // Round 0: coefficients tid + j*512, twiddle block index `base`.
    let mut v = [0u64; 8];
    unroll! {
        for j in 0..8 {
            v[j] = inp[(tile_base + tid + (j as u32) * 512) as usize];
        }
    }
    v = fwd_radix8!(v, wp, q, base);
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [512] => layout: [t0, i0]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 1: coefficients (tid & 63) + j*64 + (tid >> 6)*512.
    let t_lo = tid & 63;
    let t_hi = tid >> 6;
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(t_lo + (j as u32) * 64 + t_hi * 512) as usize];
            }
        }
    }
    v = fwd_radix8!(v, wp, q, (base << 3) + t_hi);
    sync_threads();
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [64, 8] => layout: [t0, i0, t1]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 2: coefficients (tid & 7) + j*8 + (tid >> 3)*64.
    let t_lo = tid & 7;
    let t_hi = tid >> 3;
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(t_lo + (j as u32) * 8 + t_hi * 64) as usize];
            }
        }
    }
    v = fwd_radix8!(v, wp, q, (base << 6) + t_hi);
    sync_threads();
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [8, 64] => layout: [t0, i0, t1]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 3: eight contiguous coefficients per thread, written straight out.
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(tid * 8 + (j as u32)) as usize];
            }
        }
    }
    v = fwd_radix8!(v, wp, q, (base << 9) + tid);

    let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
    let mut o = chunk_mut(out, reshape_map!([8] | [nthreads] => layout: [i0, t0]));
    unroll! {
        for j in 0..8 {
            o[j as u32] = v[j];
        }
    }
}

/// First 12 inverse stages: one 4096-coefficient tile per CTA, shared resident.
///
/// `log_n` is `log2(N)`; `scale`/`scale_shoup` apply the `n^-1` factor and are
/// set to 1 when a global pass follows this one.
#[gpu::cuda_kernel]
pub fn ntt_inverse_tile(
    inp: &[u64],
    out: &mut [u64],
    wp: &[U32_4],
    q: u64,
    log_n: u32,
    scale: u64,
    scale_shoup: u64,
) {
    assert!(Config::BDIM_X == NTT_BDIM);
    let wp = &wp[..WMAX];
    let tid = thread_id::<DimX>();
    let blk = block_id::<DimX>();
    let n = 1u32 << log_n;
    let m0 = n / TILE;
    let tile_base = blk * TILE;
    let cp = blk & (m0 - 1);

    let mut smem = GpuShared::<[u64; TILE as usize]>::zero();

    // Round 0: eight contiguous coefficients per thread (t_hi = tid, LOW = 1).
    let mut v = [0u64; 8];
    unroll! {
        for j in 0..8 {
            v[j] = inp[(tile_base + tid * 8 + (j as u32)) as usize];
        }
    }
    v = inv_radix8!(v, wp, q, (n >> 1) + (cp << 11) + 4 * tid);
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [512] => layout: [i0, t0]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 1: (tid & 7) + j*8 + (tid >> 3)*64.
    let t_lo = tid & 7;
    let t_hi = tid >> 3;
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(t_lo + (j as u32) * 8 + t_hi * 64) as usize];
            }
        }
    }
    v = inv_radix8!(v, wp, q, (n >> 4) + (cp << 8) + 4 * t_hi);
    sync_threads();
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [8, 64] => layout: [t0, i0, t1]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 2: (tid & 63) + j*64 + (tid >> 6)*512.
    let t_lo = tid & 63;
    let t_hi = tid >> 6;
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(t_lo + (j as u32) * 64 + t_hi * 512) as usize];
            }
        }
    }
    v = inv_radix8!(v, wp, q, (n >> 7) + (cp << 5) + 4 * t_hi);
    sync_threads();
    {
        let mut c = smem.chunk_mut(reshape_map!([8] | [64, 8] => layout: [t0, i0, t1]));
        unroll! {
            for j in 0..8 {
                c[j as u32] = v[j];
            }
        }
    }
    sync_threads();

    // Round 3: tid + j*512, written straight out (coalesced).
    {
        let s = &*smem;
        unroll! {
            for j in 0..8 {
                v[j] = s[(tid + (j as u32) * 512) as usize];
            }
        }
    }
    v = inv_radix8!(v, wp, q, (n >> 10) + (cp << 2));

    // Positions blk*TILE + tid + j*512: [512 threads][8 coefficients][grid].
    let ngrid = grid_dim::<DimX>();
    let mut o = chunk_mut(out, reshape_map!([8] | [512, ngrid] => layout: [t0, i0, t1]));
    unroll! {
        for j in 0..8 {
            o[j as u32] = mul_mod_shoup(v[j], scale, scale_shoup, q);
        }
    }
}

// ---------------------------------------------------------------------------
// Global passes for the stages whose butterfly distance exceeds a tile
// ---------------------------------------------------------------------------

/// One Cooley-Tukey stage with butterfly distance `t = 1 << log_t`.
///
/// The polynomial is viewed as `[batch][m_blocks][2][t]`; each thread owns four
/// adjacent `t`-offsets and both halves of the butterfly, which is exactly the
/// `reshape_map!` below.
#[gpu::cuda_kernel]
pub fn ntt_stage_forward(
    inp: &[u64],
    out: &mut [u64],
    wp: &[U32_4],
    q: u64,
    log_t: u32,
    log_n: u32,
    batch: u32,
) {
    assert!(Config::BDIM_X == NTT_BDIM);
    let wp = &wp[..WMAX];
    let t = 1u32 << log_t;
    let tq = t >> 2;
    let m_blocks = 1u32 << (log_n - 1 - log_t);
    let mut o = chunk_mut(
        out,
        reshape_map!([4, 2] | [tq, m_blocks, batch] => layout: [i0, t0, i1, t1, t2]),
    );

    let lin = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let low = lin & (tq - 1);
    let i = (lin >> (log_t - 2)) & (m_blocks - 1);
    let poly = lin >> (log_n - 3);
    let base = (poly << log_n) + i * (t << 1) + low * 4;

    let (s, sp) = twiddle!(wp, m_blocks + i);
    unroll! {
        for e in 0..4 {
            let u = inp[(base + e as u32) as usize];
            let x = mul_mod_shoup(inp[(base + e as u32 + t) as usize], s, sp, q);
            o[(e as u32, 0)] = add_mod(u, x, q);
            o[(e as u32, 1)] = sub_mod(u, x, q);
        }
    }
}

/// One Gentleman-Sande stage with butterfly distance `t = 1 << log_t`,
/// optionally folding in the `n^-1` scaling.
#[gpu::cuda_kernel]
pub fn ntt_stage_inverse(
    inp: &[u64],
    out: &mut [u64],
    wp: &[U32_4],
    q: u64,
    log_t: u32,
    log_n: u32,
    batch: u32,
    scale: u64,
    scale_shoup: u64,
) {
    assert!(Config::BDIM_X == NTT_BDIM);
    let wp = &wp[..WMAX];
    let t = 1u32 << log_t;
    let tq = t >> 2;
    let m_blocks = 1u32 << (log_n - 1 - log_t);
    let mut o = chunk_mut(
        out,
        reshape_map!([4, 2] | [tq, m_blocks, batch] => layout: [i0, t0, i1, t1, t2]),
    );

    let lin = block_id::<DimX>() * Config::BDIM_X + thread_id::<DimX>();
    let low = lin & (tq - 1);
    let i = (lin >> (log_t - 2)) & (m_blocks - 1);
    let poly = lin >> (log_n - 3);
    let base = (poly << log_n) + i * (t << 1) + low * 4;

    let (s, sp) = twiddle!(wp, m_blocks + i);
    unroll! {
        for e in 0..4 {
            let u = inp[(base + e as u32) as usize];
            let x = inp[(base + e as u32 + t) as usize];
            let lo = add_mod(u, x, q);
            let hi = mul_mod_shoup(sub_mod(u, x, q), s, sp, q);
            o[(e as u32, 0)] = mul_mod_shoup(lo, scale, scale_shoup, q);
            o[(e as u32, 1)] = mul_mod_shoup(hi, scale, scale_shoup, q);
        }
    }
}

// ---------------------------------------------------------------------------
// Host side: twiddle tables and drivers
// ---------------------------------------------------------------------------

/// Twiddle-factor tables for one ring size and modulus.
#[derive(Clone, Debug)]
pub struct NttTables {
    pub n: usize,
    pub log_n: u32,
    pub modulus: Modulus,
    /// `psi^bitrev(i)` for the forward transform.
    pub w_fwd: Vec<u64>,
    pub w_fwd_shoup: Vec<u64>,
    /// `psi^-bitrev(i)` for the inverse transform.
    pub w_inv: Vec<u64>,
    pub w_inv_shoup: Vec<u64>,
    /// `n^-1 mod q`.
    pub n_inv: u64,
    pub n_inv_shoup: u64,
    /// The primitive `2n`-th root of unity used to build the tables.
    pub psi: u64,
}

/// Smallest `x` such that `x^n == -1 mod q`, i.e. a primitive `2n`-th root.
pub fn find_psi(n: usize, m: &Modulus) -> u64 {
    let e = (m.q - 1) / (2 * n as u64);
    assert!((m.q - 1) % (2 * n as u64) == 0, "q - 1 must be divisible by 2n");
    for g in 2..1000u64 {
        let psi = m.pow(g, e);
        if m.pow(psi, n as u64) == m.q - 1 {
            return psi;
        }
    }
    panic!("no primitive 2n-th root of unity found");
}

impl NttTables {
    pub fn new(n: usize, modulus: Modulus) -> Self {
        assert!(n.is_power_of_two() && n >= TILE as usize, "N must be a power of two >= 4096");
        let log_n = n.trailing_zeros();
        let psi = find_psi(n, &modulus);
        let psi_inv = modulus.inv(psi);
        let w_fwd = crate::cpu::psi_table(psi, n, modulus.q);
        let w_inv = crate::cpu::psi_table(psi_inv, n, modulus.q);
        let w_fwd_shoup = w_fwd.iter().map(|&x| modulus.shoup(x)).collect();
        let w_inv_shoup = w_inv.iter().map(|&x| modulus.shoup(x)).collect();
        let n_inv = modulus.inv(n as u64);
        Self {
            n,
            log_n,
            modulus,
            w_fwd,
            w_fwd_shoup,
            w_inv,
            w_inv_shoup,
            n_inv,
            n_inv_shoup: modulus.shoup(n_inv),
            psi,
        }
    }

    /// Number of global passes preceding (forward) or following (inverse) the
    /// shared-memory tile kernel.
    pub fn global_passes(&self) -> u32 {
        self.log_n - TILE_STAGES
    }
}

/// Device-resident twiddle tables.
///
/// `w` and `w_shoup` are *interleaved* into one `[U32_4]` table, so a butterfly
/// fetches both halves of its Shoup constant in a single 128-bit load. See
/// [`twiddle`]; the layout also exists because SeGuRu can only vectorise
/// 16-byte accesses.
pub struct DeviceTables<'a> {
    pub w_fwd: gpu_host::TensorViewMut<'a, [U32_4]>,
    pub w_inv: gpu_host::TensorViewMut<'a, [U32_4]>,
}

impl<'a> DeviceTables<'a> {
    pub fn upload(
        ctx: &gpu_host::GpuCtxZeroGuard<'_, 'a>,
        tables: &NttTables,
    ) -> DeviceTables<'a> {
        // Padded to a compile-time constant length so the kernels can take a
        // `&wp[..WMAX]` subslice; see `WMAX`. The padding is never read.
        let pack = |w: &Vec<u64>, ws: &Vec<u64>| {
            let mut p: Vec<U32_4> = w
                .iter()
                .zip(ws.iter())
                .map(|(&a, &b)| {
                    U32_4::new([a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32])
                })
                .collect();
            p.resize(WMAX, U32_4::new([0, 0, 0, 0]));
            p
        };
        DeviceTables {
            w_fwd: ctx
                .new_tensor_view::<[U32_4]>(pack(&tables.w_fwd, &tables.w_fwd_shoup).as_slice())
                .unwrap(),
            w_inv: ctx
                .new_tensor_view::<[U32_4]>(pack(&tables.w_inv, &tables.w_inv_shoup).as_slice())
                .unwrap(),
        }
    }
}

/// Grid size (CTAs) for both NTT kernel families.
pub fn grid_for(n: usize, batch: usize) -> u32 {
    (n * batch / TILE as usize) as u32
}

type View<'a> = gpu_host::TensorViewMut<'a, [u64]>;

/// Launch a full forward NTT over `batch` polynomials.
///
/// Returns `(result, scratch)`: the transform ping-pongs between the two
/// buffers, so the caller gets them back in whichever order they ended up.
pub fn launch_forward<'a>(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, 'a>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    tables: &NttTables,
    dev: &DeviceTables<'a>,
    mut src: View<'a>,
    mut dst: View<'a>,
    batch: usize,
) -> (View<'a>, View<'a>) {
    let q = tables.modulus.q;
    let grid = grid_for(tables.n, batch);
    let passes = tables.global_passes();
    for s in 0..passes {
        let log_t = tables.log_n - 1 - s;
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const NTT_BDIM, 1, 1, 0);
        ntt_stage_forward::launch(
            cfg,
            ctx,
            md,
            &src,
            &mut dst,
            &dev.w_fwd,
            q,
            log_t,
            tables.log_n,
            batch as u32,
        )
        .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    let cfg = gpu_host::gpu_config!(grid, 1, 1, @const NTT_BDIM, 1, 1, 0);
    ntt_forward_tile::launch(
        cfg,
        ctx,
        md,
        &src,
        &mut dst,
        &dev.w_fwd,
        q,
        (tables.n / TILE as usize) as u32,
    )
    .unwrap();
    core::mem::swap(&mut src, &mut dst);
    (src, dst)
}

/// Launch a full inverse NTT (including the `n^-1` scaling) over `batch`
/// polynomials.
pub fn launch_inverse<'a>(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, 'a>,
    md: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    tables: &NttTables,
    dev: &DeviceTables<'a>,
    mut src: View<'a>,
    mut dst: View<'a>,
    batch: usize,
) -> (View<'a>, View<'a>) {
    let q = tables.modulus.q;
    let grid = grid_for(tables.n, batch);
    let passes = tables.global_passes();
    let (tile_scale, tile_scale_shoup) = if passes == 0 {
        (tables.n_inv, tables.n_inv_shoup)
    } else {
        (1, tables.modulus.shoup(1))
    };
    let cfg = gpu_host::gpu_config!(grid, 1, 1, @const NTT_BDIM, 1, 1, 0);
    ntt_inverse_tile::launch(
        cfg,
        ctx,
        md,
        &src,
        &mut dst,
        &dev.w_inv,
        q,
        tables.log_n,
        tile_scale,
        tile_scale_shoup,
    )
    .unwrap();
    core::mem::swap(&mut src, &mut dst);

    for s in 0..passes {
        let log_t = TILE_STAGES + s;
        let last = s + 1 == passes;
        let (scale, scale_shoup) = if last {
            (tables.n_inv, tables.n_inv_shoup)
        } else {
            (1, tables.modulus.shoup(1))
        };
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const NTT_BDIM, 1, 1, 0);
        ntt_stage_inverse::launch(
            cfg,
            ctx,
            md,
            &src,
            &mut dst,
            &dev.w_inv,
            q,
            log_t,
            tables.log_n,
            batch as u32,
            scale,
            scale_shoup,
        )
        .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    (src, dst)
}

/// Convenience driver: forward NTT of `batch` polynomials, host to host.
pub fn forward(data: &[u64], tables: &NttTables, batch: usize) -> Vec<u64> {
    run(data, tables, batch, true)
}

/// Convenience driver: inverse NTT of `batch` polynomials, host to host.
pub fn inverse(data: &[u64], tables: &NttTables, batch: usize) -> Vec<u64> {
    run(data, tables, batch, false)
}

fn run(data: &[u64], tables: &NttTables, batch: usize, fwd: bool) -> Vec<u64> {
    assert_eq!(data.len(), tables.n * batch);
    let mut out = vec![0u64; data.len()];
    gpu_host::cuda_ctx(0, |ctx, md| {
        let dev = DeviceTables::upload(ctx, tables);
        let a = ctx.new_tensor_view(data).unwrap();
        let b = ctx.new_tensor_view(out.as_slice()).unwrap();
        let (res, _scratch) = if fwd {
            launch_forward(ctx, md, tables, &dev, a, b, batch)
        } else {
            launch_inverse(ctx, md, tables, &dev, a, b, batch)
        };
        res.copy_to_host(&mut out).unwrap();
    });
    out
}
