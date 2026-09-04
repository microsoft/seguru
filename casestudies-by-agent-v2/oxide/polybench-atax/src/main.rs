//! PolyBench/GPU ATAX in cuda-oxide, for comparison against the SeGuRu port in
//! `../../polybench/src/atax.rs`.
//!
//! `y = A^T (A x)` with `A` of shape `NX x NY`. Both kernels keep the SeGuRu
//! decomposition so the two ports measure the toolchain rather than two
//! different algorithms:
//!
//! * `atax_tmp` reduces along a row with one warp per row and a shuffle
//!   reduction; lane 0 stores the row's result.
//! * `atax_y` reduces along a column with one thread per output column, which
//!   is already coalesced and needs no cross-thread reduction.
//!
//! Neither kernel needs shared memory, so both are expressible in cuda-oxide
//! with **no `unsafe` at all**: `WarpIndex` is the index space of `tmp`, so
//! lane 0's store goes through the checked `get_mut`, and `y` is written
//! through the ordinary 1-D thread index. The only `unsafe` in the file is
//! `kernels::load`, which cuda-oxide makes unsafe unconditionally.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread, warp};
use std::time::Instant;

/// Threads per CTA in `atax_tmp`; one warp per row, so eight rows per CTA.
const MV_THREADS: u32 = 256;
/// Rows per CTA in `atax_tmp`.
const MV_ROWS: u32 = MV_THREADS / 32;
/// Threads per CTA in `atax_y`.
const COL_THREADS: u32 = 256;

/// Four floats in one 128-bit element, matching the SeGuRu port's `Float4`, so
/// the row reduction issues `ld.global.v4.f32`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct F32x4([f32; 4]);

// SAFETY: a plain POD aggregate of four `f32` with no pointers or padding, so
// it is sound to memcpy between host and device.
unsafe impl cuda_core::DeviceCopy for F32x4 {}

#[cuda_module]
mod kernels {
    use super::*;

    /// `tmp[i] = sum_j A[i][j] * x[j]`, one warp per row.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn atax_tmp(
        a: &[super::F32x4],
        x: &[super::F32x4],
        mut tmp: DisjointSlice<f32, thread::WarpIndex>,
        ny4: u32,
    ) {
        let gid = thread::index_1d().get();
        let lane = warp::lane_id() as usize;
        let row = gid / 32;

        let ny4 = ny4 as usize;
        let base = row * ny4;

        let mut acc = 0.0f32;
        let mut j = lane;
        while j < ny4 {
            let ai = base + j;
            if ai < a.len() && j < x.len() {
                let av = a[ai].0;
                let xv = x[j].0;
                acc += av[0] * xv[0] + av[1] * xv[1] + av[2] * xv[2] + av[3] * xv[3];
            }
            j += 32;
        }
        let total = warp::reduce_sum_f32(acc);

        // `warp_index` mints the witness only for lane 0, so the row's slot has
        // exactly one writer and the store stays checked and safe.
        if let Some(slot) = thread::warp_index()
            && let Some(out) = tmp.get_mut(slot)
        {
            *out = total;
        }
    }

    /// `y[j] = sum_i A[i][j] * tmp[i]`, one thread per column.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn atax_y(a: &[f32], tmp: &[f32], mut y: DisjointSlice<f32>, nx: u32, ny: u32) {
        let idx = thread::index_1d();
        let j = idx.get();
        let (nx, ny) = (nx as usize, ny as usize);

        // Four independent accumulators, matching the 4-way unroll of the
        // SeGuRu port, so the FMA chain is not serialised on one register.
        let mut acc = [0.0f32; 4];
        let mut i = 0usize;
        while i + 4 <= nx {
            let mut u = 0usize;
            while u < 4 {
                let ii = i + u;
                let ai = ii * ny + j;
                if ai < a.len() && ii < tmp.len() {
                    acc[u] += a[ai] * tmp[ii];
                }
                u += 1;
            }
            i += 4;
        }
        while i < nx {
            let ai = i * ny + j;
            if ai < a.len() && i < tmp.len() {
                acc[0] += a[ai] * tmp[i];
            }
            i += 1;
        }

        if let Some(out) = y.get_mut(idx) {
            *out = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        }
    }
}

/// Deterministic pseudo-random values in `[-1, 1)`.
///
/// Byte-for-byte the generator in `../../polybench/src/common.rs`, so both
/// ports are verified and timed on identical inputs.
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

fn atax_cpu(a: &[f32], x: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; nx];
    for i in 0..nx {
        let mut s = 0.0f32;
        for j in 0..ny {
            s += a[i * ny + j] * x[j];
        }
        tmp[i] = s;
    }
    let mut y = vec![0.0f32; ny];
    for j in 0..ny {
        let mut s = 0.0f32;
        for i in 0..nx {
            s += a[i * ny + j] * tmp[i];
        }
        y[j] = s;
    }
    y
}

/// Relative infinity-norm error, the metric used by the SeGuRu benchmark.
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

/// Matches `iters_for_bytes` in the SeGuRu benchmark.
fn iters_for_bytes(bytes: f64) -> u32 {
    (1.0e10 / bytes).clamp(10.0, 300.0) as u32
}

const WARMUP: u32 = 3;

fn bench(n: usize) -> (f64, f32) {
    let (nx, ny) = (n, n);
    let a = seq(nx * ny, 11);
    let x = seq(ny, 12);
    let iters = iters_for_bytes(2.0 * (nx * ny * 4) as f64);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // `atax_tmp` reads A as 128-bit quads and `atax_y` reads it as scalars.
    // `cast_chunks` re-views the *same* allocation, so A is stored once, as in
    // the SeGuRu port where both views alias one tensor. Uploading a second
    // quad copy instead would double the working set and thrash L2.
    let mut d_a = DeviceBuffer::from_host(&stream, &a).unwrap();
    let x4: Vec<F32x4> = x
        .chunks_exact(4)
        .map(|c| F32x4([c[0], c[1], c[2], c[3]]))
        .collect();
    let d_x4 = DeviceBuffer::from_host(&stream, &x4).unwrap();
    let mut d_tmp = DeviceBuffer::<f32>::zeroed(&stream, nx).unwrap();
    let mut d_y = DeviceBuffer::<f32>::zeroed(&stream, ny).unwrap();

    // SAFETY: cuda-oxide makes module loading unsafe unconditionally; there is
    // no safe alternative. Nothing else in this file needs `unsafe`.
    let module = unsafe { kernels::load(&ctx) }.unwrap();

    let cfg_tmp = LaunchConfig1D::new((nx as u32).div_ceil(MV_ROWS), MV_THREADS, 0);
    let cfg_y = LaunchConfig1D::new((ny as u32).div_ceil(COL_THREADS), COL_THREADS, 0);
    let prep_tmp = module.prepare_atax_tmp(cfg_tmp).unwrap();
    let prep_y = module.prepare_atax_y(cfg_y).unwrap();

    // A macro rather than a closure: every launch needs a fresh mutable borrow
    // of `d_tmp`, which a closure capturing it would hold for its whole life.
    macro_rules! atax_once {
        () => {{
            let d_a4 = match d_a.cast_chunks::<F32x4>() {
                Ok(v) => v,
                Err(_) => unreachable!("A is a power-of-two element count"),
            };
            module
                .atax_tmp(&stream, &prep_tmp, &d_a4, &d_x4, &mut d_tmp, (ny / 4) as u32)
                .unwrap();
            d_a = match d_a4.cast_chunks::<f32>() {
                Ok(v) => v,
                Err(_) => unreachable!("quads always split back into scalars"),
            };
            module
                .atax_y(&stream, &prep_y, &d_a, &d_tmp, &mut d_y, nx as u32, ny as u32)
                .unwrap();
        }};
    }

    for _ in 0..WARMUP {
        atax_once!();
    }
    stream.synchronize().unwrap();

    let t = Instant::now();
    for _ in 0..iters {
        atax_once!();
    }
    stream.synchronize().unwrap();
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    if std::env::var_os("ATAX_SPLIT").is_some() {
        let t = Instant::now();
        for _ in 0..iters {
            let d_a4 = d_a.cast_chunks::<F32x4>().ok().unwrap();
            module
                .atax_tmp(&stream, &prep_tmp, &d_a4, &d_x4, &mut d_tmp, (ny / 4) as u32)
                .unwrap();
            d_a = d_a4.cast_chunks::<f32>().ok().unwrap();
        }
        stream.synchronize().unwrap();
        let us_tmp = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            module
                .atax_y(&stream, &prep_y, &d_a, &d_tmp, &mut d_y, nx as u32, ny as u32)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let us_y = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!("split {n}: atax_tmp={us_tmp:.1}us atax_y={us_y:.1}us");
    }

    let got = d_y.to_host_vec(&stream).unwrap();
    let want = atax_cpu(&a, &x, nx, ny);
    (us, max_rel(&got, &want))
}

fn main() {
    println!("kernel,impl,size,us,rel_err");
    for &n in &[2048usize, 4096, 8192] {
        let (us, err) = bench(n);
        println!("atax,oxide,{n}^2,{us:.3},{err:.3e}");
    }
}
