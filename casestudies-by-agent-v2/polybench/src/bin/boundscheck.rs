//! Measurement-only experiment: how much of a SeGuRu kernel's runtime is spent
//! on slice bounds checks that the backend fails to elide?
//!
//! **This binary is now historical.** It established the size of the
//! bounds-check tax; the shipped `conv3d` and `mvt` kernels have since been
//! rewritten so that the checks are elided for *arbitrary* sizes, not just
//! powers of two (see `README.md`, optimisation log experiment A). Re-running
//! it therefore no longer reports 29-50%: `conv3d` now comes out *negative*
//! (the shipped kernel is faster than this masked variant, because the masked
//! variant still uses the old `reshape_map!` output map — experiment B), and
//! the mvt column pass still shows ~20%, which is analysed in the README as a
//! `IMAD.WIDE.U32` addressing artefact rather than a bounds check.
//!
//! `rustc_codegen_gpu` emits, for every `&[f32]` index in a kernel, a
//! `cvt.u64.u32` + `setp.gt.u64` + two `selp.b64` (a branchless "address or
//! null" select). The instruction-count evidence for that is in `README.md`.
//! This binary turns the inference into a measurement by building a variant of
//! the two worst-affected kernels in which the checks *are* elided, and timing
//! the pair back to back.
//!
//! The mechanism is a compile-time power-of-two mask. `idx & (N - 1)` is
//! provably `< N` to LLVM, and the kernel body is guarded by a single
//! `a.len() >= N` test, so `idx < a.len()` becomes provable and the per-access
//! check folds away. Because every index the stock kernel produces is already
//! `< N`, the mask is a no-op on the data: **the variant is numerically
//! identical to the stock kernel**, which this binary asserts before reporting
//! any time. The mask itself costs one `and.b32` per access, so the measured
//! saving is a slight *under*-estimate of the true bounds-check tax.
//!
//! The mask requires the problem size to be a power of two, so the sizes here
//! are 128^3 / 256^3 for conv3d (instead of the 384^3 in the main sweep) and
//! 2048 / 8192 for mvt.
//!
//! No `unsafe`: this is still safe Rust, it just hands LLVM a range fact it
//! could not derive on its own.

use std::time::Instant;

use crunchy::unroll;
use gpu::*;
use polybench_gpu::common::seq;
use polybench_gpu::conv3d::{BX, BY};
use polybench_gpu::mvt::COL_BDIM;
use polybench_gpu::{conv3d, mvt};

// ---------------------------------------------------------------------------
// conv3d, bounds-check-free variants
// ---------------------------------------------------------------------------

const C11: f32 = 2.0;
const C21: f32 = 5.0;
const C31: f32 = -8.0;
const C12: f32 = -3.0;
const C22: f32 = 6.0;
const C32: f32 = -9.0;
const C13: f32 = 4.0;
const C23: f32 = 7.0;
const C33: f32 = 10.0;

const N128: u32 = 128 * 128 * 128;
const N256: u32 = 256 * 256 * 256;

#[gpu::device]
#[inline(always)]
fn at128(a: &[f32], plane: u32, nk: u32, i: u32, j: u32, k: u32) -> f32 {
    a[((i * plane + j * nk + k) & (N128 - 1)) as usize]
}

#[gpu::device]
#[inline(always)]
fn at256(a: &[f32], plane: u32, nk: u32, i: u32, j: u32, k: u32) -> f32 {
    a[((i * plane + j * nk + k) & (N256 - 1)) as usize]
}

macro_rules! conv3d_body {
    ($at:ident, $a:ident, $b:ident, $ni:ident, $nj:ident, $nk:ident, $n:expr) => {{
        let gx = grid_dim::<DimX>();
        let gy = grid_dim::<DimY>();
        let gz = grid_dim::<DimZ>();

        let k = block_id::<DimX>() * BX + thread_id::<DimX>();
        let j = block_id::<DimY>() * BY + thread_id::<DimY>();
        let i = block_id::<DimZ>();

        let mut out = chunk_mut(
            $b,
            reshape_map!([1] | [32, gx, 8, gy, 1, gz] => layout: [i0, t0, t1, t2, t3, t4, t5]),
        );

        if $a.len() >= $n as usize {
            let interior =
                i > 0 && i + 1 < $ni && j > 0 && j + 1 < $nj && k > 0 && k + 1 < $nk;

            let im = i.max(1) - 1;
            let ip = (i + 1).min($ni - 1);
            let jm = j.max(1) - 1;
            let jp = (j + 1).min($nj - 1);
            let km = k.max(1) - 1;
            let kp = (k + 1).min($nk - 1);

            let plane = $nj * $nk;

            let v = C11 * $at($a, plane, $nk, im, jm, km)
                + C13 * $at($a, plane, $nk, ip, jm, km)
                + C21 * $at($a, plane, $nk, im, jm, km)
                + C23 * $at($a, plane, $nk, ip, jm, km)
                + C31 * $at($a, plane, $nk, im, jm, km)
                + C33 * $at($a, plane, $nk, ip, jm, km)
                + C12 * $at($a, plane, $nk, i, jm, k)
                + C22 * $at($a, plane, $nk, i, j, k)
                + C32 * $at($a, plane, $nk, i, jp, k)
                + C11 * $at($a, plane, $nk, im, jm, kp)
                + C13 * $at($a, plane, $nk, ip, jm, kp)
                + C21 * $at($a, plane, $nk, im, j, kp)
                + C23 * $at($a, plane, $nk, ip, j, kp)
                + C31 * $at($a, plane, $nk, im, jp, kp)
                + C33 * $at($a, plane, $nk, ip, jp, kp);

            out[0] = if interior { v } else { 0.0 };
        }
    }};
}

#[gpu::cuda_kernel]
pub fn conv3d_nobc_128(a: &[f32], b: &mut [f32], ni: u32, nj: u32, nk: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    assert!(Config::BDIM_Z == 1);
    conv3d_body!(at128, a, b, ni, nj, nk, N128)
}

#[gpu::cuda_kernel]
pub fn conv3d_nobc_256(a: &[f32], b: &mut [f32], ni: u32, nj: u32, nk: u32) {
    assert!(Config::BDIM_X == BX);
    assert!(Config::BDIM_Y == BY);
    assert!(Config::BDIM_Z == 1);
    conv3d_body!(at256, a, b, ni, nj, nk, N256)
}

// ---------------------------------------------------------------------------
// mvt column pass, bounds-check-free variants
// ---------------------------------------------------------------------------

const MV2048: u32 = 2048;
const MV8192: u32 = 8192;

macro_rules! mvt_x2_body {
    ($a:ident, $y2:ident, $x2:ident, $n:ident, $side:expr) => {{
        let i = block_id::<DimX>() * COL_BDIM + thread_id::<DimX>();

        let mut out = chunk_mut($x2, MapContinuousLinear::new(1));

        if $a.len() >= ($side as usize) * ($side as usize) && $y2.len() >= $side as usize {
            let mut acc = [0.0f32; 4];
            let mut j = 0u32;
            while j < $n {
                unroll! {
                    for u in 0..4 {
                        let jj = j + u as u32;
                        acc[u] += $a[((jj * $n + i) & ($side * $side - 1)) as usize]
                            * $y2[(jj & ($side - 1)) as usize];
                    }
                }
                j += 4;
            }
            out[0] = out[0] + (acc[0] + acc[1]) + (acc[2] + acc[3]);
        }
    }};
}

#[gpu::cuda_kernel]
pub fn mvt_x2_nobc_2048(a: &[f32], y2: &[f32], x2: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == COL_BDIM);
    mvt_x2_body!(a, y2, x2, n, MV2048)
}

#[gpu::cuda_kernel]
pub fn mvt_x2_nobc_8192(a: &[f32], y2: &[f32], x2: &mut [f32], n: u32) {
    assert!(Config::BDIM_X == COL_BDIM);
    mvt_x2_body!(a, y2, x2, n, MV8192)
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Row {
    kernel: &'static str,
    size: String,
    stock: f64,
    nobc: f64,
    err: f32,
}

/// Relative infinity-norm error, as in the main benchmark driver.
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

fn main() {
    let mut rows: Vec<Row> = Vec::new();

    for &n in &[128usize, 256usize] {
        rows.push(bench_conv3d(n));
    }
    for &n in &[2048usize, 8192usize] {
        rows.push(bench_mvt_x2(n));
    }

    std::println!("\nBounds-check tax: stock SeGuRu kernel vs a variant in which the");
    std::println!("per-access slice bounds check is provably elided (see file header).");
    std::println!("Each variant is verified against the stock kernel before its time is printed.\n");
    std::println!("| Kernel | Size | stock (us) | no-bounds-check (us) | tax | max rel err |");
    std::println!("|---|---|---|---|---|---|");
    for r in &rows {
        let tax = 100.0 * (r.stock - r.nobc) / r.stock;
        if r.err > 1e-5 {
            std::println!(
                "| {} | {} | {:.1} | MISMATCH ({:.1e}) | - | - |",
                r.kernel, r.size, r.stock, r.err
            );
            continue;
        }
        std::println!(
            "| {} | {} | {:.1} | {:.1} | {:.1}% | {:.1e} |",
            r.kernel, r.size, r.stock, r.nobc, tax, r.err
        );
    }
    std::println!();
}

fn bench_conv3d(n: usize) -> Row {
    let a = seq(n * n * n, 61);
    let zb = vec![0.0f32; n * n * n];
    let (iters, warmup) = if n <= 128 { (200, 20) } else { (50, 5) };

    let (t_stock, t_nobc, err) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let mut db = ctx.new_tensor_view(zb.as_slice()).unwrap();
        macro_rules! cfg {
            () => {
                gpu_host::gpu_config!(
                    (n / BX as usize) as u32, (n / BY as usize) as u32, n as u32,
                    @const BX, @const BY, 1, 0)
            };
        }
        let (nu, nu2, nu3) = (n as u32, n as u32, n as u32);

        // --- stock ---
        for _ in 0..warmup {
            conv3d::conv3d_kernel::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
        }
        ctx.sync().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            conv3d::conv3d_kernel::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
        }
        ctx.sync().unwrap();
        let t_stock = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let mut ref_out = vec![0.0f32; n * n * n];
        db.copy_to_host(&mut ref_out).unwrap();

        // --- no bounds check ---
        db.copy_from_host(zb.as_slice()).unwrap();
        for _ in 0..warmup {
            if n == 128 {
                conv3d_nobc_128::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
            } else {
                conv3d_nobc_256::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
            }
        }
        ctx.sync().unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            if n == 128 {
                conv3d_nobc_128::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
            } else {
                conv3d_nobc_256::launch(cfg!(), ctx, m, &da, &mut db, nu, nu2, nu3).unwrap();
            }
        }
        ctx.sync().unwrap();
        let t_nobc = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let mut got = vec![0.0f32; n * n * n];
        db.copy_to_host(&mut got).unwrap();

        let err = max_rel(&got, &ref_out);
        (t_stock, t_nobc, err)
    });

    Row {
        kernel: "conv3d",
        size: format!("{n}^3"),
        stock: t_stock,
        nobc: t_nobc,
        err,
    }
}

fn bench_mvt_x2(n: usize) -> Row {
    let a = seq(n * n, 31);
    let y2 = seq(n, 34);
    let x2 = seq(n, 36);
    let iters = if n <= 2048 { 200 } else { 20 };
    let warmup = 10;

    let (t_stock, t_nobc, err) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let dy = ctx.new_tensor_view(y2.as_slice()).unwrap();
        let mut dx = ctx.new_tensor_view(x2.as_slice()).unwrap();
        macro_rules! cfg {
            () => {
                gpu_host::gpu_config!(
                    (n / COL_BDIM as usize) as u32, 1, 1, @const COL_BDIM, 1, 1, 0)
            };
        }
        let nu = n as u32;

        for _ in 0..warmup {
            mvt::mvt_x2::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
        }
        ctx.sync().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            mvt::mvt_x2::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
        }
        ctx.sync().unwrap();
        let t_stock = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // single clean pass for verification
        dx.copy_from_host(x2.as_slice()).unwrap();
        mvt::mvt_x2::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
        let mut ref_out = vec![0.0f32; n];
        dx.copy_to_host(&mut ref_out).unwrap();

        dx.copy_from_host(x2.as_slice()).unwrap();
        for _ in 0..warmup {
            if n == 2048 {
                mvt_x2_nobc_2048::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            } else {
                mvt_x2_nobc_8192::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            }
        }
        ctx.sync().unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            if n == 2048 {
                mvt_x2_nobc_2048::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            } else {
                mvt_x2_nobc_8192::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            }
        }
        ctx.sync().unwrap();
        let t_nobc = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        dx.copy_from_host(x2.as_slice()).unwrap();
        if n == 2048 {
                mvt_x2_nobc_2048::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            } else {
                mvt_x2_nobc_8192::launch(cfg!(), ctx, m, &da, &dy, &mut dx, nu).unwrap();
            }
        let mut got = vec![0.0f32; n];
        dx.copy_to_host(&mut got).unwrap();

        let err = max_rel(&got, &ref_out);
        (t_stock, t_nobc, err)
    });

    Row {
        kernel: "mvt (column pass)",
        size: format!("{n}^2"),
        stock: t_stock,
        nobc: t_nobc,
        err,
    }
}
