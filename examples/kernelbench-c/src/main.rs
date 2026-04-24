//! KernelBench Phase C — fused operators (Level 2).
//!
//! Same interface and helper plumbing as kernelbench-b. Each problem module
//! exposes:
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
//! Driver writes all inputs (including Linear weights/biases or Conv weights)
//! into in_dir as raw little-endian f32 .bin files — each problem's `run`
//! documents which file names it expects.

#![allow(non_snake_case)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use gpu::prelude::*;

// Pilot problem modules.
pub mod gemm_mul_lrelu;       // 12_Gemm_Multiply_LeakyReLU
pub mod conv_relu_hardswish;  // 57_Conv2d_ReLU_HardSwish
pub mod matmul_mish_mish;       // 29
pub mod matmul_scale_resadd;    // 40
pub mod gemm_scale_htanh_gelu;  // 53
pub mod matmul_sigmoid_sum;     // 56
pub mod gemm_relu_div;          // 63
pub mod conv_relu_biasadd;      // 1

pub mod from_cuda;

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

pub fn time_launches<F: FnMut()>(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    mut f: F,
    iters: usize,
    warmup_iters: usize,
) -> (f64, f64) {
    let wt = Instant::now();
    for _ in 0..warmup_iters { f(); }
    ctx.sync().unwrap();
    let warmup = wt.elapsed().as_micros() as f64 / warmup_iters.max(1) as f64;
    let t = Instant::now();
    for _ in 0..iters { f(); }
    ctx.sync().unwrap();
    let us = t.elapsed().as_micros() as f64 / iters as f64;
    (us, warmup)
}

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
            "gemm_mul_lrelu"      => gemm_mul_lrelu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "conv_relu_hardswish" => conv_relu_hardswish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_mish_mish"       => matmul_mish_mish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_scale_resadd"    => matmul_scale_resadd::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gemm_scale_htanh_gelu"  => gemm_scale_htanh_gelu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_sigmoid_sum"     => matmul_sigmoid_sum::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gemm_relu_div"          => gemm_relu_div::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "conv_relu_biasadd"      => conv_relu_biasadd::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gemm_mul_lrelu_fc"      => from_cuda::gemm_mul_lrelu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "conv_relu_hardswish_fc" => from_cuda::conv_relu_hardswish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_mish_mish_fc"       => from_cuda::matmul_mish_mish::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_scale_resadd_fc"    => from_cuda::matmul_scale_resadd::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gemm_scale_htanh_gelu_fc"  => from_cuda::gemm_scale_htanh_gelu::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "matmul_sigmoid_sum_fc"     => from_cuda::matmul_sigmoid_sum::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "gemm_relu_div_fc"          => from_cuda::gemm_relu_div::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            "conv_relu_biasadd_fc"      => from_cuda::conv_relu_biasadd::run(ctx, md, &a.in_dir, &a.out_dir, a.iters, &a.shape),
            other => panic!("unknown problem: {other}"),
        };
        println!(
            "{{\"problem\":\"{}\",\"shape\":{:?},\"iters\":{},\"kernel_us\":{:.3},\"warmup_us\":{:.3}}}",
            a.problem, a.shape, a.iters, us, warmup
        );
    });
}
