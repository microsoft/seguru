//! KernelBench Phase B — LLM-generated SeGuRu kernels.
//!
//! Each problem lives in its own module (one file). A sub-agent writes each
//! module from the PyTorch source + the skill doc.
//!
//! Interface each problem module MUST expose:
//!
//!   pub fn run(
//!       ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
//!       md:  &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
//!       in_dir:  &std::path::Path,
//!       out_dir: &std::path::Path,
//!       iters:   usize,
//!       shape:   &[usize],
//!   ) -> (f64, f64);   // (kernel_us_per_iter, warmup_us_per_iter)
//!
//! It reads its inputs from `in_dir/*.bin` (raw little-endian f32), writes its
//! output to `out_dir/<name>.bin`, times `iters` launches bracketed by
//! `ctx.sync()`, and returns per-iter kernel time in microseconds.

#![allow(non_snake_case)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use gpu::prelude::*;

// Problem modules — each is written by the LLM sub-agent.
pub mod leaky_relu;
pub mod tanh;
pub mod rms_norm;

// ===== shared utilities (available to all problem modules) =====

pub fn read_bin(path: &Path, n: usize) -> Vec<f32> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out = vec![0f32; n];
    for (i, c) in buf.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(c.try_into().unwrap());
    }
    out
}

pub fn write_bin(path: &Path, data: &[f32]) {
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path:?}: {e}"));
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data { buf.extend_from_slice(&v.to_le_bytes()); }
    f.write_all(&buf).unwrap();
}

/// Helper: time N kernel launches, returning (per-iter us, warmup_per_iter us).
/// `f` runs one launch. Caller is responsible for ctx.sync() via this helper.
pub fn time_launches<F: FnMut()>(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    mut f: F,
    iters: usize,
    warmup_iters: usize,
) -> (f64, f64) {
    // Warmup
    let wt = Instant::now();
    for _ in 0..warmup_iters { f(); }
    ctx.sync().unwrap();
    let warmup = wt.elapsed().as_micros() as f64 / warmup_iters.max(1) as f64;
    // Timed
    let t = Instant::now();
    for _ in 0..iters { f(); }
    ctx.sync().unwrap();
    let us = t.elapsed().as_micros() as f64 / iters as f64;
    (us, warmup)
}

// ===== CLI =====

#[derive(Default)]
struct CliArgs {
    problem: String,
    in_dir: PathBuf,
    out_dir: PathBuf,
    iters: usize,
    shape: Vec<usize>,
}

fn parse_cli(args: Vec<String>) -> CliArgs {
    let mut a = CliArgs { iters: 100, ..Default::default() };
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        match k.as_str() {
            "--problem" => a.problem = it.next().unwrap(),
            "--in-dir"  => a.in_dir  = it.next().unwrap().into(),
            "--out-dir" => a.out_dir = it.next().unwrap().into(),
            "--iters"   => a.iters   = it.next().unwrap().parse().unwrap(),
            "--shape"   => a.shape   = it.next().unwrap().split(',').map(|t| t.parse().unwrap()).collect(),
            _ => panic!("unknown arg: {k}"),
        }
    }
    a
}

fn main() {
    let a = parse_cli(std::env::args().skip(1).collect());
    gpu_host::cuda_ctx(0, |ctx, md| {
        let (us, warmup) = match a.problem.as_str() {
            "leaky_relu" => leaky_relu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "tanh"       => tanh::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "rms_norm"   => rms_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            other => panic!("unknown problem: {other}"),
        };
        println!(
            "{{\"problem\":\"{}\",\"shape\":{:?},\"iters\":{},\"kernel_us\":{:.3},\"warmup_us\":{:.3}}}",
            a.problem, a.shape, a.iters, us, warmup
        );
    });
}
