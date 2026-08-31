//! PolyBench/GPU benchmark: SeGuRu vs hand-written CUDA vs single-core CPU.
//!
//! Methodology:
//!
//! * All GPU timings are **kernel only**. Host allocation and H2D/D2H copies
//!   happen once, outside the timed region, on both sides.
//! * The SeGuRu side is timed with `Instant` around `warmup` untimed launches
//!   followed by `iters` timed launches and a single `ctx.sync()`; the CUDA
//!   side is timed the same way with CUDA events. Both therefore include
//!   launch overhead amortised over the iteration count.
//! * The timed loop deliberately lets in-place state evolve (that is free and
//!   does not change the amount of work). Verification uses a separate clean
//!   run from the original inputs on both sides.
//! * A kernel's time is only printed if the SeGuRu output matches the CUDA
//!   output element-wise within a relative tolerance. Otherwise the row reads
//!   `MISMATCH`.
//! * The CPU column is the crate's own scalar reference, run once at the
//!   smallest size only (it is 3-5 orders of magnitude slower).

use std::time::Instant;

use polybench_gpu::common::{seq, to_float4};
use polybench_gpu::cuda_ffi as cu;
use polybench_gpu::*;
use gpu_host::gpu_config;

const WARMUP: u32 = 3;

// Block-dim constants must be plain paths for `gpu_config!(@const ..)`.
const GEMM_BDIM: u32 = gemm::BDIM;
const SYRK_BDIM: u32 = syrk::BDIM;
const SYR2K_BDIM: u32 = syr2k::BDIM;
const MV_BX: u32 = atax::MV_BX;
const MV_BY: u32 = atax::MV_BY;
const COL_BDIM: u32 = atax::COL_BDIM;
const S_BX: u32 = conv2d::BX;
const S_BY: u32 = conv2d::BY;
const J1_BDIM: u32 = jacobi1d::BDIM;

/// Time `body` on the SeGuRu side: `warmup` untimed launches, then `iters`
/// timed launches, wall clock, divided by `iters`. Result in microseconds.
macro_rules! time_sg {
    ($ctx:expr, $warmup:expr, $iters:expr, $body:block) => {{
        for _ in 0..$warmup {
            $body
        }
        $ctx.sync().unwrap();
        let __t = Instant::now();
        for _ in 0..$iters {
            $body
        }
        $ctx.sync().unwrap();
        __t.elapsed().as_secs_f64() * 1e6 / ($iters as f64)
    }};
}

struct Row {
    kernel: &'static str,
    size: String,
    sg: f64,
    cu: f64,
    extra: Option<(&'static str, f64)>,
    cpu: Option<f64>,
    err: f32,
    tol: f32,
}

impl Row {
    fn ok(&self) -> bool {
        self.err <= self.tol
    }
}

/// Relative infinity-norm error, `max|got - want| / max(max|want|, 1)`.
///
/// The per-element form used by `common::assert_close` is not usable here: the
/// values produced at benchmark sizes are `O(1e3)`-`O(1e4)`, so an element that
/// happens to sit near zero after massive cancellation gets a denominator of 1
/// and reports a huge "relative" error for what is ordinary f32 noise. The
/// vector-norm form is the standard BLAS verification metric and is what is
/// gated on here. Non-finite values are only tolerated if both sides agree.
fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    let mut scale = 1.0f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if !g.is_finite() || !w.is_finite() {
            if g.is_finite() != w.is_finite() {
                return f32::INFINITY;
            }
            continue;
        }
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

fn iters_for_flops(flops: f64) -> u32 {
    (2.0e10 / flops).clamp(10.0, 300.0) as u32
}

fn iters_for_bytes(bytes: f64) -> u32 {
    (1.0e10 / bytes).clamp(10.0, 300.0) as u32
}

// ---------------------------------------------------------------------------
// GEMM family
// ---------------------------------------------------------------------------

fn bench_gemm(n: usize, cpu: bool) -> Row {
    let (ni, nj, nk) = (n, n, n);
    let (alpha, beta) = (0.5f32, 0.5f32);
    let a = seq(ni * nk, 1);
    let b = seq(nk * nj, 2);
    let c = seq(ni * nj, 3);
    let iters = iters_for_flops(2.0 * (n as f64).powi(3));

    let gx = (nj / gemm::TILE as usize) as u32;
    let gy = (ni / gemm::TILE as usize) as u32;
    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(c.as_slice()).unwrap();
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(gx, gy, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, &da, &db, &mut dc, nk as u32, alpha, beta)
                .unwrap();
        });
        dc.copy_from_host(c.as_slice()).unwrap();
        let cfg = gpu_config!(gx, gy, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
        gemm::gemm_kernel::launch(cfg, ctx, m, &da, &db, &mut dc, nk as u32, alpha, beta).unwrap();
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ni * nj];
        dc.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::gemm(&a, &b, &c, ni, nj, nk, alpha, beta, WARMUP, iters);
    let (bl_us, _) = cu::gemm_cublas(&a, &b, &c, ni, nj, nk, alpha, beta, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let mut want = c.clone();
        let t = Instant::now();
        gemm::gemm_cpu(&a, &b, &mut want, ni, nj, nk, alpha, beta);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "gemm",
        size: format!("{n}^3"),
        sg: sg_us,
        cu: cu_us,
        extra: Some(("cuBLAS", bl_us)),
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_twomm(n: usize, cpu: bool) -> Row {
    let (ni, nj, nk, nl) = (n, n, n, n);
    let (alpha, beta) = (0.5f32, 0.5f32);
    let a = seq(ni * nk, 121);
    let b = seq(nk * nj, 122);
    let c = seq(nj * nl, 123);
    let d = seq(ni * nl, 124);
    let iters = iters_for_flops(4.0 * (n as f64).powi(3));

    let t = gemm::TILE as usize;
    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let dc = ctx.new_tensor_view(c.as_slice()).unwrap();
        let mut dd = ctx.new_tensor_view(d.as_slice()).unwrap();
        let zt = vec![0.0f32; ni * nj];
        let mut dt = ctx.new_tensor_view(zt.as_slice()).unwrap();
        let g1x = (nj / t) as u32;
        let g1y = (ni / t) as u32;
        let g2x = (nl / t) as u32;
        let go = |ctx: &_, m: &_, dt: &mut _, dd: &mut _| {
            let cfg = gpu_config!(g1x, g1y, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, &da, &db, dt, nk as u32, alpha, 0.0).unwrap();
            let cfg = gpu_config!(g2x, g1y, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, dt, &dc, dd, nj as u32, 1.0, beta).unwrap();
        };
        let us = time_sg!(ctx, WARMUP, iters, {
            go(ctx, m, &mut dt, &mut dd);
        });
        dd.copy_from_host(d.as_slice()).unwrap();
        go(ctx, m, &mut dt, &mut dd);
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ni * nl];
        dd.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::twomm(&a, &b, &c, &d, ni, nj, nk, nl, alpha, beta, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let mut want = d.clone();
        let t = Instant::now();
        twomm::twomm_cpu(&a, &b, &c, &mut want, ni, nj, nk, nl, alpha, beta);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "twomm",
        size: format!("{n}^3 x2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_threemm(n: usize, cpu: bool) -> Row {
    let (ni, nj, nk, nl, nm) = (n, n, n, n, n);
    let a = seq(ni * nk, 131);
    let b = seq(nk * nj, 132);
    let c = seq(nj * nm, 133);
    let d = seq(nm * nl, 134);
    let iters = iters_for_flops(6.0 * (n as f64).powi(3));

    let t = gemm::TILE as usize;
    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let dc = ctx.new_tensor_view(c.as_slice()).unwrap();
        let dd = ctx.new_tensor_view(d.as_slice()).unwrap();
        let ze = vec![0.0f32; ni * nj];
        let zf = vec![0.0f32; nj * nl];
        let zg = vec![0.0f32; ni * nl];
        let mut de = ctx.new_tensor_view(ze.as_slice()).unwrap();
        let mut df = ctx.new_tensor_view(zf.as_slice()).unwrap();
        let mut dg = ctx.new_tensor_view(zg.as_slice()).unwrap();
        let gi = (ni / t) as u32;
        let gj = (nj / t) as u32;
        let gl = (nl / t) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(gj, gi, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, &da, &db, &mut de, nk as u32, 1.0, 0.0).unwrap();
            let cfg = gpu_config!(gl, gj, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, &dc, &dd, &mut df, nm as u32, 1.0, 0.0).unwrap();
            let cfg = gpu_config!(gl, gi, 1, @const GEMM_BDIM, @const GEMM_BDIM, 1, 0);
            gemm::gemm_kernel::launch(cfg, ctx, m, &de, &df, &mut dg, nj as u32, 1.0, 0.0).unwrap();
        });
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ni * nl];
        dg.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::threemm(&a, &b, &c, &d, ni, nj, nk, nl, nm, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = threemm::threemm_cpu(&a, &b, &c, &d, ni, nj, nk, nl, nm);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "threemm",
        size: format!("{n}^3 x3"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_syrk(n: usize, cpu: bool) -> Row {
    let m = n;
    let (alpha, beta) = (0.75f32, 0.5f32);
    let a = seq(n * m, 101);
    let c = seq(n * n, 102);
    let iters = iters_for_flops(2.0 * (n as f64) * (n as f64) * (m as f64));

    let g = (n / syrk::TILE as usize) as u32;
    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, mo| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(c.as_slice()).unwrap();
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(g, g, 1, @const SYRK_BDIM, @const SYRK_BDIM, 1, 0);
            syrk::syrk_kernel::launch(cfg, ctx, mo, &da, &mut dc, m as u32, alpha, beta).unwrap();
        });
        dc.copy_from_host(c.as_slice()).unwrap();
        let cfg = gpu_config!(g, g, 1, @const SYRK_BDIM, @const SYRK_BDIM, 1, 0);
        syrk::syrk_kernel::launch(cfg, ctx, mo, &da, &mut dc, m as u32, alpha, beta).unwrap();
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; n * n];
        dc.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::syrk(&a, &c, n, m, alpha, beta, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let mut want = c.clone();
        let t = Instant::now();
        syrk::syrk_cpu(&a, &mut want, n, m, alpha, beta);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "syrk",
        size: format!("{n}^3"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_syr2k(n: usize, cpu: bool) -> Row {
    let m = n;
    let (alpha, beta) = (0.5f32, 0.5f32);
    let a = seq(n * m, 111);
    let b = seq(n * m, 112);
    let c = seq(n * n, 113);
    let iters = iters_for_flops(4.0 * (n as f64) * (n as f64) * (m as f64));

    let g = (n / syr2k::TILE as usize) as u32;
    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, mo| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let mut dc = ctx.new_tensor_view(c.as_slice()).unwrap();
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(g, g, 1, @const SYR2K_BDIM, @const SYR2K_BDIM, 1, 0);
            syr2k::syr2k_kernel::launch(cfg, ctx, mo, &da, &db, &mut dc, m as u32, alpha, beta)
                .unwrap();
        });
        dc.copy_from_host(c.as_slice()).unwrap();
        let cfg = gpu_config!(g, g, 1, @const SYR2K_BDIM, @const SYR2K_BDIM, 1, 0);
        syr2k::syr2k_kernel::launch(cfg, ctx, mo, &da, &db, &mut dc, m as u32, alpha, beta)
            .unwrap();
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; n * n];
        dc.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::syr2k(&a, &b, &c, n, m, alpha, beta, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let mut want = c.clone();
        let t = Instant::now();
        syr2k::syr2k_cpu(&a, &b, &mut want, n, m, alpha, beta);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "syr2k",
        size: format!("{n}^3"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

// ---------------------------------------------------------------------------
// Mat-vec family
// ---------------------------------------------------------------------------

fn bench_atax(n: usize, cpu: bool) -> Row {
    let (nx, ny) = (n, n);
    let a = seq(nx * ny, 11);
    let x = seq(ny, 12);
    let a4 = to_float4(&a);
    let x4 = to_float4(&x);
    let iters = iters_for_bytes(2.0 * (nx * ny * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(a4.as_slice()).unwrap();
        let dx4 = ctx.new_tensor_view(x4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let zt = vec![0.0f32; nx];
        let zy = vec![0.0f32; ny];
        let mut dt = ctx.new_tensor_view(zt.as_slice()).unwrap();
        let mut dy = ctx.new_tensor_view(zy.as_slice()).unwrap();
        let g1 = (nx / atax::MV_BY as usize) as u32;
        let g2 = (ny / atax::COL_BDIM as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(1, g1, 1, @const MV_BX, @const MV_BY, 1, 0);
            atax::atax_tmp::launch(cfg, ctx, m, &da4, &dx4, &mut dt, (ny / 4) as u32).unwrap();
            let cfg = gpu_config!(g2, 1, 1, @const COL_BDIM, 1, 1, 0);
            atax::atax_y::launch(cfg, ctx, m, &da, &dt, &mut dy, nx as u32, ny as u32).unwrap();
        });
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ny];
        dy.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::atax(&a, &x, nx, ny, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = atax::atax_cpu(&a, &x, nx, ny);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "atax",
        size: format!("{n}^2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_bicg(n: usize, cpu: bool) -> Row {
    let (nx, ny) = (n, n);
    let a = seq(nx * ny, 21);
    let p = seq(ny, 22);
    let r = seq(nx, 23);
    let a4 = to_float4(&a);
    let p4 = to_float4(&p);
    let iters = iters_for_bytes(2.0 * (nx * ny * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(a4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let dp4 = ctx.new_tensor_view(p4.as_slice()).unwrap();
        let dr = ctx.new_tensor_view(r.as_slice()).unwrap();
        let zq = vec![0.0f32; nx];
        let zs = vec![0.0f32; ny];
        let mut dq = ctx.new_tensor_view(zq.as_slice()).unwrap();
        let mut ds = ctx.new_tensor_view(zs.as_slice()).unwrap();
        let g1 = (nx / bicg::MV_BY as usize) as u32;
        let g2 = (ny / bicg::COL_BDIM as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(1, g1, 1, @const MV_BX, @const MV_BY, 1, 0);
            bicg::bicg_q::launch(cfg, ctx, m, &da4, &dp4, &mut dq, (ny / 4) as u32).unwrap();
            let cfg = gpu_config!(g2, 1, 1, @const COL_BDIM, 1, 1, 0);
            bicg::bicg_s::launch(cfg, ctx, m, &da, &dr, &mut ds, nx as u32, ny as u32).unwrap();
        });
        ctx.sync().unwrap();
        let mut hs = vec![0.0f32; ny];
        let mut hq = vec![0.0f32; nx];
        ds.copy_to_host(&mut hs).unwrap();
        dq.copy_to_host(&mut hq).unwrap();
        (us, (hs, hq))
    });

    let (cu_us, (cs, cq)) = cu::bicg(&a, &p, &r, nx, ny, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = bicg::bicg_cpu(&a, &p, &r, nx, ny);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "bicg",
        size: format!("{n}^2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out.0, &cs).max(max_rel(&sg_out.1, &cq)),
        tol: 1e-5,
    }
}

fn bench_gesummv(n: usize, cpu: bool) -> Row {
    let (alpha, beta) = (1.5f32, -0.75f32);
    let a = seq(n * n, 41);
    let b = seq(n * n, 42);
    let x = seq(n, 43);
    let a4 = to_float4(&a);
    let b4 = to_float4(&b);
    let x4 = to_float4(&x);
    let iters = iters_for_bytes(2.0 * (n * n * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a4.as_slice()).unwrap();
        let db = ctx.new_tensor_view(b4.as_slice()).unwrap();
        let dx = ctx.new_tensor_view(x4.as_slice()).unwrap();
        let zy = vec![0.0f32; n];
        let mut dy = ctx.new_tensor_view(zy.as_slice()).unwrap();
        let g = (n / gesummv::MV_BY as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(1, g, 1, @const MV_BX, @const MV_BY, 1, 0);
            gesummv::gesummv_kernel::launch(
                cfg,
                ctx,
                m,
                &da,
                &db,
                &dx,
                &mut dy,
                (n / 4) as u32,
                alpha,
                beta,
            )
            .unwrap();
        });
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; n];
        dy.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::gesummv(&a, &b, &x, n, alpha, beta, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = gesummv::gesummv_cpu(&a, &b, &x, n, alpha, beta);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "gesummv",
        size: format!("{n}^2 x2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_mvt(n: usize, cpu: bool) -> Row {
    let a = seq(n * n, 31);
    let x1 = seq(n, 32);
    let x2 = seq(n, 33);
    let y1 = seq(n, 34);
    let y2 = seq(n, 35);
    let a4 = to_float4(&a);
    let y14 = to_float4(&y1);
    let iters = iters_for_bytes(2.0 * (n * n * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da4 = ctx.new_tensor_view(a4.as_slice()).unwrap();
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let dy1 = ctx.new_tensor_view(y14.as_slice()).unwrap();
        let dy2 = ctx.new_tensor_view(y2.as_slice()).unwrap();
        let mut dx1 = ctx.new_tensor_view(x1.as_slice()).unwrap();
        let mut dx2 = ctx.new_tensor_view(x2.as_slice()).unwrap();
        let g1 = (n / mvt::MV_BY as usize) as u32;
        let g2 = (n / mvt::COL_BDIM as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(1, g1, 1, @const MV_BX, @const MV_BY, 1, 0);
            mvt::mvt_x1::launch(cfg, ctx, m, &da4, &dy1, &mut dx1, (n / 4) as u32).unwrap();
            let cfg = gpu_config!(g2, 1, 1, @const COL_BDIM, 1, 1, 0);
            mvt::mvt_x2::launch(cfg, ctx, m, &da, &dy2, &mut dx2, n as u32).unwrap();
        });
        dx1.copy_from_host(x1.as_slice()).unwrap();
        dx2.copy_from_host(x2.as_slice()).unwrap();
        let cfg = gpu_config!(1, g1, 1, @const MV_BX, @const MV_BY, 1, 0);
        mvt::mvt_x1::launch(cfg, ctx, m, &da4, &dy1, &mut dx1, (n / 4) as u32).unwrap();
        let cfg = gpu_config!(g2, 1, 1, @const COL_BDIM, 1, 1, 0);
        mvt::mvt_x2::launch(cfg, ctx, m, &da, &dy2, &mut dx2, n as u32).unwrap();
        ctx.sync().unwrap();
        let mut h1 = vec![0.0f32; n];
        let mut h2 = vec![0.0f32; n];
        dx1.copy_to_host(&mut h1).unwrap();
        dx2.copy_to_host(&mut h2).unwrap();
        (us, (h1, h2))
    });

    let (cu_us, (c1, c2)) = cu::mvt(&a, &x1, &x2, &y1, &y2, n, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = mvt::mvt_cpu(&a, &x1, &x2, &y1, &y2, n);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "mvt",
        size: format!("{n}^2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out.0, &c1).max(max_rel(&sg_out.1, &c2)),
        tol: 1e-5,
    }
}

// ---------------------------------------------------------------------------
// Stencils
// ---------------------------------------------------------------------------

fn bench_conv2d(n: usize, cpu: bool) -> Row {
    let (ni, nj) = (n, n);
    let a = seq(ni * nj, 51);
    let iters = iters_for_bytes(2.0 * (ni * nj * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let zb = vec![0.0f32; ni * nj];
        let mut db = ctx.new_tensor_view(zb.as_slice()).unwrap();
        let gx = (nj / conv2d::BX as usize) as u32;
        let gy = (ni / conv2d::CTA_ROWS as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            conv2d::conv2d_kernel::launch(cfg, ctx, m, &da, &mut db, ni as u32, nj as u32).unwrap();
        });
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ni * nj];
        db.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::conv2d(&a, ni, nj, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = conv2d::conv2d_cpu(&a, ni, nj);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "conv2d",
        size: format!("{n}^2"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-6,
    }
}

fn bench_conv3d(n: usize, cpu: bool) -> Row {
    let (ni, nj, nk) = (n, n, n);
    let a = seq(ni * nj * nk, 61);
    let iters = iters_for_bytes(2.0 * (ni * nj * nk * 4) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let zb = vec![0.0f32; ni * nj * nk];
        let mut db = ctx.new_tensor_view(zb.as_slice()).unwrap();
        let gx = (nk / conv3d::BX as usize) as u32;
        let gy = (nj / conv3d::BY as usize) as u32;
        let gz = ni as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            let cfg = gpu_config!(gx, gy, gz, @const S_BX, @const S_BY, 1, 0);
            conv3d::conv3d_kernel::launch(
                cfg, ctx, m, &da, &mut db, ni as u32, nj as u32, nk as u32,
            )
            .unwrap();
        });
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; ni * nj * nk];
        db.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, cu_out) = cu::conv3d(&a, ni, nj, nk, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = conv3d::conv3d_cpu(&a, ni, nj, nk);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "conv3d",
        size: format!("{n}^3"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &cu_out),
        tol: 1e-5,
    }
}

fn bench_jacobi1d(n: usize, tsteps: usize, cpu: bool) -> Row {
    let a = seq(n, 71);
    let b = seq(n, 72);
    let iters = iters_for_bytes(4.0 * (n * 4 * tsteps) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let mut da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let mut db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let g = (n / jacobi1d::BDIM as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            for _ in 0..tsteps {
                let cfg = gpu_config!(g, 1, 1, @const J1_BDIM, 1, 1, 0);
                jacobi1d::jacobi1d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
                let cfg = gpu_config!(g, 1, 1, @const J1_BDIM, 1, 1, 0);
                jacobi1d::jacobi1d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
            }
        });
        da.copy_from_host(a.as_slice()).unwrap();
        db.copy_from_host(b.as_slice()).unwrap();
        for _ in 0..tsteps {
            let cfg = gpu_config!(g, 1, 1, @const J1_BDIM, 1, 1, 0);
            jacobi1d::jacobi1d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
            let cfg = gpu_config!(g, 1, 1, @const J1_BDIM, 1, 1, 0);
            jacobi1d::jacobi1d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
        }
        ctx.sync().unwrap();
        let mut ha = vec![0.0f32; n];
        da.copy_to_host(&mut ha).unwrap();
        (us, ha)
    });

    let (cu_us, (ca, _)) = cu::jacobi1d(&a, &b, n, tsteps, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = jacobi1d::jacobi1d_cpu(&a, &b, n, tsteps);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "jacobi1d",
        size: format!("{n} x t{tsteps}"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &ca),
        tol: 1e-5,
    }
}

fn bench_jacobi2d(n: usize, tsteps: usize, cpu: bool) -> Row {
    let a = seq(n * n, 81);
    let b = seq(n * n, 82);
    let iters = iters_for_bytes(4.0 * (n * n * 4 * tsteps) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let mut da = ctx.new_tensor_view(a.as_slice()).unwrap();
        let mut db = ctx.new_tensor_view(b.as_slice()).unwrap();
        let gx = (n / jacobi2d::BX as usize) as u32;
        let gy = (n / jacobi2d::BY as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            for _ in 0..tsteps {
                let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
                jacobi2d::jacobi2d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
                let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
                jacobi2d::jacobi2d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
            }
        });
        da.copy_from_host(a.as_slice()).unwrap();
        db.copy_from_host(b.as_slice()).unwrap();
        for _ in 0..tsteps {
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            jacobi2d::jacobi2d_step::launch(cfg, ctx, m, &da, &mut db, n as u32).unwrap();
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            jacobi2d::jacobi2d_copy::launch(cfg, ctx, m, &db, &mut da, n as u32).unwrap();
        }
        ctx.sync().unwrap();
        let mut ha = vec![0.0f32; n * n];
        da.copy_to_host(&mut ha).unwrap();
        (us, ha)
    });

    let (cu_us, (ca, _)) = cu::jacobi2d(&a, &b, n, tsteps, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = jacobi2d::jacobi2d_cpu(&a, &b, n, tsteps);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "jacobi2d",
        size: format!("{n}^2 x t{tsteps}"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &ca),
        tol: 1e-5,
    }
}

fn bench_fdtd2d(n: usize, tmax: usize, cpu: bool) -> Row {
    let (nx, ny) = (n, n);
    let ex = seq(nx * ny, 91);
    let ey = seq(nx * ny, 92);
    let hz = seq(nx * ny, 93);
    let fict = seq(tmax, 94);
    let iters = iters_for_bytes(6.0 * (nx * ny * 4 * tmax) as f64);

    let (sg_us, sg_out) = gpu_host::cuda_ctx(0, |ctx, m| {
        let mut dex = ctx.new_tensor_view(ex.as_slice()).unwrap();
        let mut dey = ctx.new_tensor_view(ey.as_slice()).unwrap();
        let mut dhz = ctx.new_tensor_view(hz.as_slice()).unwrap();
        let gx = (ny / fdtd2d::BX as usize) as u32;
        let gy = (nx / fdtd2d::BY as usize) as u32;
        let us = time_sg!(ctx, WARMUP, iters, {
            for &f in fict.iter() {
                let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
                fdtd2d::fdtd_ey::launch(cfg, ctx, m, &mut dey, &dhz, ny as u32, f).unwrap();
                let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
                fdtd2d::fdtd_ex::launch(cfg, ctx, m, &mut dex, &dhz, ny as u32).unwrap();
                let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
                fdtd2d::fdtd_hz::launch(cfg, ctx, m, &mut dhz, &dex, &dey, nx as u32, ny as u32)
                    .unwrap();
            }
        });
        dex.copy_from_host(ex.as_slice()).unwrap();
        dey.copy_from_host(ey.as_slice()).unwrap();
        dhz.copy_from_host(hz.as_slice()).unwrap();
        for &f in fict.iter() {
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            fdtd2d::fdtd_ey::launch(cfg, ctx, m, &mut dey, &dhz, ny as u32, f).unwrap();
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            fdtd2d::fdtd_ex::launch(cfg, ctx, m, &mut dex, &dhz, ny as u32).unwrap();
            let cfg = gpu_config!(gx, gy, 1, @const S_BX, @const S_BY, 1, 0);
            fdtd2d::fdtd_hz::launch(cfg, ctx, m, &mut dhz, &dex, &dey, nx as u32, ny as u32)
                .unwrap();
        }
        ctx.sync().unwrap();
        let mut h = vec![0.0f32; nx * ny];
        dhz.copy_to_host(&mut h).unwrap();
        (us, h)
    });

    let (cu_us, (_, _, chz)) = cu::fdtd2d(&ex, &ey, &hz, &fict, nx, ny, WARMUP, iters);

    let cpu_us = cpu.then(|| {
        let t = Instant::now();
        let _ = fdtd2d::fdtd2d_cpu(&ex, &ey, &hz, &fict, nx, ny);
        t.elapsed().as_secs_f64() * 1e6
    });

    Row {
        kernel: "fdtd2d",
        size: format!("{n}^2 x t{tmax}"),
        sg: sg_us,
        cu: cu_us,
        extra: None,
        cpu: cpu_us,
        err: max_rel(&sg_out, &chz),
        tol: 1e-5,
    }
}

// ---------------------------------------------------------------------------

fn main() {
    let filter: Option<String> = std::env::args().nth(1);
    let want = |k: &str| filter.as_deref().is_none_or(|f| f == k);

    let mut rows: Vec<Row> = Vec::new();

    if want("gemm") {
        for (i, &n) in [512usize, 1024, 2048, 4096].iter().enumerate() {
            rows.push(bench_gemm(n, i == 0));
        }
    }
    if want("twomm") {
        for (i, &n) in [512usize, 1024, 2048].iter().enumerate() {
            rows.push(bench_twomm(n, i == 0));
        }
    }
    if want("threemm") {
        for (i, &n) in [512usize, 1024, 2048].iter().enumerate() {
            rows.push(bench_threemm(n, i == 0));
        }
    }
    if want("syrk") {
        for (i, &n) in [512usize, 1024, 2048, 4096].iter().enumerate() {
            rows.push(bench_syrk(n, i == 0));
        }
    }
    if want("syr2k") {
        for (i, &n) in [512usize, 1024, 2048].iter().enumerate() {
            rows.push(bench_syr2k(n, i == 0));
        }
    }
    if want("atax") {
        for (i, &n) in [2048usize, 4096, 8192].iter().enumerate() {
            rows.push(bench_atax(n, i == 0));
        }
    }
    if want("bicg") {
        for (i, &n) in [2048usize, 4096, 8192].iter().enumerate() {
            rows.push(bench_bicg(n, i == 0));
        }
    }
    if want("gesummv") {
        for (i, &n) in [2048usize, 4096, 8192].iter().enumerate() {
            rows.push(bench_gesummv(n, i == 0));
        }
    }
    if want("mvt") {
        for (i, &n) in [2048usize, 4096, 8192].iter().enumerate() {
            rows.push(bench_mvt(n, i == 0));
        }
    }
    if want("conv2d") {
        for (i, &n) in [2048usize, 4096, 8192].iter().enumerate() {
            rows.push(bench_conv2d(n, i == 0));
        }
    }
    if want("conv3d") {
        for (i, &n) in [128usize, 256, 384].iter().enumerate() {
            rows.push(bench_conv3d(n, i == 0));
        }
    }
    if want("jacobi1d") {
        for (i, &n) in [1usize << 20, 1 << 22, 1 << 24].iter().enumerate() {
            rows.push(bench_jacobi1d(n, 20, i == 0));
        }
    }
    if want("jacobi2d") {
        for (i, &n) in [1024usize, 2048, 4096].iter().enumerate() {
            rows.push(bench_jacobi2d(n, 10, i == 0));
        }
    }
    if want("fdtd2d") {
        for (i, &n) in [1024usize, 2048, 4096].iter().enumerate() {
            rows.push(bench_fdtd2d(n, 10, i == 0));
        }
    }

    println!("\nPolyBench/GPU: SeGuRu (safe Rust) vs hand-written CUDA, A100 80GB PCIe");
    println!("Times are kernel-only microseconds per iteration.\n");
    println!(
        "| Kernel | Size | SeGuRu (us) | CUDA (us) | SeGuRu/CUDA | CPU (us) | GPU vs CPU | max rel err |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for r in &rows {
        if !r.ok() {
            println!(
                "| {} | {} | MISMATCH | MISMATCH | - | - | - | {:.3e} (tol {:.0e}) |",
                r.kernel, r.size, r.err, r.tol
            );
            continue;
        }
        let cpu = r.cpu.map(|c| format!("{c:.0}")).unwrap_or_else(|| "-".into());
        let sp = r
            .cpu
            .map(|c| format!("{:.0}x", c / r.sg))
            .unwrap_or_else(|| "-".into());
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {} | {} | {:.1e} |",
            r.kernel,
            r.size,
            r.sg,
            r.cu,
            r.sg / r.cu,
            cpu,
            sp,
            r.err
        );

        let param = r.size.replace(' ', "");
        csv_row("polybench", r.kernel, &param, "seguru", "time", r.sg, "us");
        csv_row("polybench", r.kernel, &param, "cuda", "time", r.cu, "us");
        if let Some(c) = r.cpu {
            csv_row("polybench", r.kernel, &param, "cpu", "time", c, "us");
        }
    }

    let extras: Vec<&Row> = rows.iter().filter(|r| r.extra.is_some()).collect();
    if !extras.is_empty() {
        println!("\nVendor-library reference (not the mirrored baseline)\n");
        println!("| Kernel | Size | SeGuRu (us) | CUDA mirror (us) | cuBLAS (us) | SeGuRu/cuBLAS |");
        println!("|---|---|---|---|---|---|");
        for r in extras {
            let (_, t) = r.extra.unwrap();
            println!(
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.2}x |",
                r.kernel,
                r.size,
                r.sg,
                r.cu,
                t,
                r.sg / t
            );

            if r.ok() {
                let param = r.size.replace(' ', "");
                csv_row("polybench", r.kernel, &param, "cublas", "time", t, "us");
            }
        }
    }

    let bad = rows.iter().filter(|r| !r.ok()).count();
    if bad > 0 {
        eprintln!("\n{bad} kernel/size combinations MISMATCHED; times suppressed for those rows");
        std::process::exit(1);
    }
}

/// Appends one measurement row to the CSV file named by `BENCH_CSV`, if set.
/// No-op (and creates no file) when the environment variable is unset.
fn csv_row(suite: &str, workload: &str, parameter: &str, implementation: &str, metric: &str, value: f64, units: &str) {
    use std::io::Write;
    let Ok(path) = std::env::var("BENCH_CSV") else { return };
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) else { return };
    let _ = writeln!(f, "{suite},{workload},{parameter},{implementation},{metric},{value:.6},{units}");
}
