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
pub mod relu;
pub mod sigmoid;
pub mod gelu;
pub mod softmax;
pub mod layer_norm;
pub mod sum_dim;
pub mod l2_norm;
pub mod empty;
pub mod log_softmax;
pub mod swish;
pub mod softplus;
pub mod l1_norm;
pub mod max_pool1d;
pub mod avg_pool1d;
pub mod mean_dim;
pub mod max_dim;
pub mod min_dim;
pub mod argmax_dim;
pub mod cumsum;
pub mod mse_loss;

// Parallel set of SeGuRu kernels translated from raw CUDA (not PyTorch).
pub mod from_cuda;

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

pub fn write_bin_i64(path: &Path, data: &[i64]) {
    std::fs::create_dir_all(path.parent().unwrap()).ok();
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("create {path:?}: {e}"));
    let mut buf = Vec::with_capacity(data.len() * 8);
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
            "relu"       => relu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "sigmoid"    => sigmoid::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gelu"       => gelu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "softmax"    => softmax::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "layer_norm" => layer_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "sum_dim"    => sum_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "l2_norm"    => l2_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "empty"      => empty::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "log_softmax" => log_softmax::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "swish"       => swish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "softplus"    => softplus::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "l1_norm"     => l1_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "max_pool1d"  => max_pool1d::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "avg_pool1d"  => avg_pool1d::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "mean_dim"    => mean_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "max_dim"     => max_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "min_dim"     => min_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "argmax_dim"  => argmax_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "cumsum"      => cumsum::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "mse_loss"    => mse_loss::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "log_softmax_fc" => from_cuda::log_softmax::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "swish_fc"       => from_cuda::swish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "softplus_fc"    => from_cuda::softplus::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "l1_norm_fc"     => from_cuda::l1_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "max_pool1d_fc"  => from_cuda::max_pool1d::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "avg_pool1d_fc"  => from_cuda::avg_pool1d::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "mean_dim_fc"    => from_cuda::mean_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "max_dim_fc"     => from_cuda::max_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "min_dim_fc"     => from_cuda::min_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "argmax_dim_fc"  => from_cuda::argmax_dim::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "cumsum_fc"      => from_cuda::cumsum::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "mse_loss_fc"    => from_cuda::mse_loss::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),

            // from_cuda variants (SeGuRu translated from the raw-CUDA kernel).
            "leaky_relu_fc" => from_cuda::leaky_relu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "tanh_fc"       => from_cuda::tanh::run      (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "rms_norm_fc"   => from_cuda::rms_norm::run  (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "relu_fc"       => from_cuda::relu::run      (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "sigmoid_fc"    => from_cuda::sigmoid::run   (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gelu_fc"       => from_cuda::gelu::run      (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "softmax_fc"    => from_cuda::softmax::run   (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "layer_norm_fc" => from_cuda::layer_norm::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "sum_dim_fc"    => from_cuda::sum_dim::run   (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "l2_norm_fc"    => from_cuda::l2_norm::run   (ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            other => panic!("unknown problem: {other}"),
        };
        println!(
            "{{\"problem\":\"{}\",\"shape\":{:?},\"iters\":{},\"kernel_us\":{:.3},\"warmup_us\":{:.3}}}",
            a.problem, a.shape, a.iters, us, warmup
        );
    });
}
