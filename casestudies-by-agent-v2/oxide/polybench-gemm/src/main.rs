//! PolyBench/GPU GEMM in cuda-oxide, for comparison against the SeGuRu port in
//! `../../polybench/src/gemm.rs`.
//!
//! `C = alpha * A * B + beta * C`, `A` is `M x K`, `B` is `K x N`, `C` is
//! `M x N`, all row-major.
//!
//! Two kernels, because the point of this benchmark is what the safety model
//! costs:
//!
//! * [`kernels::gemm_safe`] is register-blocked with **no `unsafe` at all**. A
//!   16x16 CTA owns a 64x64 tile of `C` and each thread accumulates a 4x4
//!   micro-tile, exactly the SeGuRu blocking, but the `K` dimension is streamed
//!   straight from global memory because cuda-oxide has no safe shared memory.
//!   The write goes through `RuntimeRowMajorTiles<4, 4>`, which proves the 4x4
//!   rectangle in bounds once and then needs no per-element check.
//! * [`kernels::gemm_smem`] adds the shared-memory `K` staging that the SeGuRu
//!   kernel uses, which is what makes that kernel fast. It is the same
//!   algorithm, and it cannot be written without `unsafe`: shared memory in
//!   cuda-oxide is a `static mut SharedArray`, and `crates/cuda-device/src/
//!   shared.rs:41-46` states that all access requires `unsafe` because the
//!   memory starts uninitialised and the barriers that order it are invisible
//!   to the type system. cuda-oxide's own `gemm_views` example, whose whole
//!   subject is proof-carrying safe views, still writes its shared tiles inside
//!   `unsafe` blocks for the same reason.
//!
//! The SeGuRu kernel this is compared against uses shared-memory staging *and*
//! contains no `unsafe`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{
    DisjointSlice, LocalIndex32, RuntimeRowMajorTiles, SharedArray, Uniform, cuda_module, kernel,
    launch_bounds,
    launch_contract, thread,
};
use std::time::Instant;

/// Threads per CTA in each dimension.
const BDIM: u32 = 16;
/// Outputs per thread in each dimension.
const TT: usize = 4;
/// Rows/columns of `C` owned by one CTA.
const TILE: u32 = BDIM * TT as u32;
/// Depth of one `K` slab staged in shared memory by [`kernels::gemm_smem`].
const KTILE: usize = 16;

#[cuda_module]
mod kernels {
    use super::*;

    /// Register-blocked GEMM with no shared memory and no `unsafe`.
    ///
    /// Each thread holds a 4x4 accumulator in registers and walks `K` one step
    /// at a time, reading its four `A` values and four `B` values from global
    /// memory. The four `B` reads of the 16 threads in a row are contiguous, so
    /// the `B` traffic coalesces and is served by L1/L2; the `A` values are
    /// reused across the row of threads through the cache rather than through
    /// shared memory.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    pub fn gemm_safe(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<4, 4>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        let row0 = coord.row() as usize * TT;
        let col0 = coord.col() as usize * TT;
        let (m, n, k) = (m as usize, n.get() as usize, k as usize);

        if row0 + TT > m || col0 + TT > n {
            return;
        }

        let mut acc = [[0.0f32; TT]; TT];
        let mut kk = 0usize;
        while kk < k {
            let mut af = [0.0f32; TT];
            let mut bf = [0.0f32; TT];
            let mut i = 0usize;
            while i < TT {
                let ai = (row0 + i) * k + kk;
                let bi = kk * n + col0 + i;
                af[i] = if ai < a.len() { a[ai] } else { 0.0 };
                bf[i] = if bi < b.len() { b[bi] } else { 0.0 };
                i += 1;
            }
            let mut i = 0usize;
            while i < TT {
                let mut j = 0usize;
                while j < TT {
                    acc[i][j] += af[i] * bf[j];
                    j += 1;
                }
                i += 1;
            }
            kk += 1;
        }

        // One check for the whole 4x4 rectangle, then unchecked element access
        // inside it. This is the safe write path; no `unsafe` is involved.
        // One check for the whole 4x4 rectangle, then unchecked element access
        // inside it. This is the safe write path; no `unsafe` is involved.
        if let Some(mut tile) = c.tile_2d32_rt(coord) {
            let mut i = 0usize;
            while i < TT {
                let mut j = 0usize;
                while j < TT {
                    if let Some(r) = LocalIndex32::new(i as u32)
                        && let Some(col) = LocalIndex32::new(j as u32)
                    {
                        let mut cell = tile.at(r, col);
                        let previous = cell.read();
                        cell.write(alpha * acc[i][j] + beta * previous);
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }

    /// The same blocking with the `K` slab staged through shared memory, which
    /// is the algorithm the SeGuRu kernel implements.
    ///
    /// `As` and `Bs` hold a `KTILE x TILE` slab each, stored `k`-major. Every
    /// thread loads four elements of each per slab, so the CTA's 256 threads
    /// fill both 16x64 slabs exactly.
    ///
    /// The shared accesses below are the reason this kernel is not safe: a
    /// `SharedArray` lives in a `static mut`, so reading or writing one is an
    /// `unsafe` operation no matter how the surrounding indices were derived.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
        requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
    pub fn gemm_smem(
        m: u32,
        n: Uniform<u32>,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, RuntimeRowMajorTiles<4, 4>>,
    ) {
        static mut AS: SharedArray<f32, { KTILE * TILE as usize }> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, { KTILE * TILE as usize }> = SharedArray::UNINIT;

        let coord = thread::coord_2d_u32(lc);
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let brow = thread::blockIdx_y() as usize * TILE as usize;
        let bcol = thread::blockIdx_x() as usize * TILE as usize;
        let (m, n, k) = (m as usize, n.get() as usize, k as usize);

        // The host pads to whole tiles, so a CTA is either fully inside the
        // matrix or entirely outside it; there is no partial-tile path.
        if brow + TILE as usize > m || bcol + TILE as usize > n || k % KTILE != 0 {
            return;
        }

        let mut acc = [[0.0f32; TT]; TT];

        let mut slab = 0usize;
        while slab < k {
            thread::sync_threads();
            // Stage one KTILE-deep slab. `As[kk][r]` and `Bs[kk][c]`, both
            // k-major, so the inner product below reads them contiguously.
            let mut u = 0usize;
            while u < TT {
                let r = ty + BDIM as usize * u;
                let ai = (brow + r) * k + slab + tx;
                let bi = (slab + ty) * n + bcol + tx + BDIM as usize * u;
                let av = if ai < a.len() { a[ai] } else { 0.0 };
                let bv = if bi < b.len() { b[bi] } else { 0.0 };
                // SAFETY: `tx * TILE + r` and `ty * TILE + tx + 16u` are below
                // KTILE * TILE for tx, ty < 16 and u < 4, and each (tx, ty, u)
                // maps to a distinct slot, so no two threads write the same
                // element. cuda-oxide requires `unsafe` here regardless: shared
                // memory is a `static mut`.
                unsafe {
                    AS[tx * TILE as usize + r] = av;
                    BS[ty * TILE as usize + tx + BDIM as usize * u] = bv;
                }
                u += 1;
            }
            thread::sync_threads();

            let mut kl = 0usize;
            while kl < KTILE {
                let base = kl * TILE as usize;
                let mut af = [0.0f32; TT];
                let mut bf = [0.0f32; TT];
                let mut i = 0usize;
                while i < TT {
                    // SAFETY: both indices are below KTILE * TILE, and the
                    // `sync_threads` above ordered every write of this slab
                    // before these reads.
                    unsafe {
                        af[i] = AS[base + ty * TT + i];
                        bf[i] = BS[base + tx * TT + i];
                    }
                    i += 1;
                }
                let mut i = 0usize;
                while i < TT {
                    let mut j = 0usize;
                    while j < TT {
                        acc[i][j] += af[i] * bf[j];
                        j += 1;
                    }
                    i += 1;
                }
                kl += 1;
            }
            slab += KTILE;
        }

        if let Some(mut tile) = c.tile_2d32_rt(coord) {
            let mut i = 0usize;
            while i < TT {
                let mut j = 0usize;
                while j < TT {
                    if let Some(r) = LocalIndex32::new(i as u32)
                        && let Some(col) = LocalIndex32::new(j as u32)
                    {
                        let mut cell = tile.at(r, col);
                        let previous = cell.read();
                        cell.write(alpha * acc[i][j] + beta * previous);
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }
}

/// Deterministic pseudo-random values in `[-1, 1)`, identical to the generator
/// in `../../polybench/src/common.rs`.
fn seq(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

fn gemm_cpu(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
) {
    for i in 0..ni {
        for j in 0..nj {
            let mut acc = 0.0f32;
            for kk in 0..nk {
                acc += a[i * nk + kk] * b[kk * nj + j];
            }
            c[i * nj + j] = alpha * acc + beta * c[i * nj + j];
        }
    }
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    let mut scale = 1.0f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
        }
        if w.abs() > scale {
            scale = w.abs();
        }
    }
    worst / scale
}

/// Matches `iters_for_flops` in the SeGuRu benchmark.
fn iters_for_flops(flops: f64) -> u32 {
    (2.0e10 / flops).clamp(10.0, 300.0) as u32
}

const WARMUP: u32 = 3;

/// One `(time, error)` pair per kernel, in the order `(safe, smem)`.
fn bench(n: usize, verify: bool) -> [(f64, f32); 2] {
    assert!(n % TILE as usize == 0 && n % KTILE == 0);
    let (alpha, beta) = (0.5f32, 0.5f32);
    let a = seq(n * n, 1);
    let b = seq(n * n, 2);
    let c = seq(n * n, 3);
    let iters = iters_for_flops(2.0 * (n as f64).powi(3));

    let want = verify.then(|| {
        let mut w = c.clone();
        gemm_cpu(&a, &b, &mut w, n, n, n, alpha, beta);
        w
    });

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let d_a = DeviceBuffer::from_host(&stream, &a).unwrap();
    let d_b = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut d_c = DeviceBuffer::from_host(&stream, &c).unwrap();

    // SAFETY: cuda-oxide makes module loading unsafe unconditionally.
    let module = unsafe { kernels::load(&ctx) }.unwrap();

    let grid = (n as u32).div_ceil(TILE);
    let cfg = LaunchConfig2D::new((grid, grid), (BDIM, BDIM), 0);
    let prep_safe = module.prepare_gemm_safe(cfg).unwrap();
    let prep_smem = module.prepare_gemm_smem(cfg).unwrap();

    let nu = n as u32;

    macro_rules! measure {
        ($prep:expr, $method:ident) => {{
            macro_rules! once {
                () => {
                    module
                        .$method(
                            &stream,
                            &$prep,
                            nu,
                            nu,
                            nu,
                            alpha,
                            &d_a,
                            &d_b,
                            beta,
                            cuda_host::RowWidth::new(&mut d_c, nu),
                        )
                        .unwrap()
                };
            }

            for _ in 0..WARMUP {
                once!();
            }
            stream.synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..iters {
                once!();
            }
            stream.synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // A clean run from the original `C` for verification, since the
            // timed loop let `C` evolve in place.
            let err = match &want {
                Some(w) => {
                    d_c.copy_from_host(&stream, &c).unwrap();
                    once!();
                    stream.synchronize().unwrap();
                    max_rel(&d_c.to_host_vec(&stream).unwrap(), w)
                }
                None => 0.0,
            };
            d_c.copy_from_host(&stream, &c).unwrap();
            (us, err)
        }};
    }

    let safe = measure!(prep_safe, gemm_safe);
    let smem = measure!(prep_smem, gemm_smem);
    [safe, smem]
}

fn main() {
    println!("kernel,impl,size,us,rel_err");
    for (i, &n) in [512usize, 1024, 2048, 4096].iter().enumerate() {
        let [safe, smem] = bench(n, i == 0);
        println!("gemm,oxide-safe,{n}^3,{:.3},{:.3e}", safe.0, safe.1);
        println!("gemm,oxide-smem,{n}^3,{:.3},{:.3e}", smem.0, smem.1);
    }
}
