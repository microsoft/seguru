//! Benchmark binary: times 43 SeGuRu GPU kernels against CUDA C reference implementations.
//! Outputs CSV to stdout and a summary table to stderr.

#[path = "../cuda_ffi.rs"]
#[allow(dead_code)]
mod cuda_ffi;

use gpu_host::cuda_ctx;
use std::collections::BTreeMap;
use std::time::Instant;

use kernelbench::argreduce::*;
use kernelbench::cumulative::*;
use kernelbench::elementwise::*;
use kernelbench::gelu_variants::*;
use kernelbench::loss::*;
use kernelbench::matmul::*;
use kernelbench::matvec::*;
use kernelbench::norm::*;
use kernelbench::reduction::*;
use kernelbench::softmax::*;

const WARMUP: i32 = 3;
const ITERS: i32 = 10;

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn gen_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.1).collect()
}

fn gen_targets(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 11) as f32 * 0.1).collect()
}

struct BenchResult {
    kernel: String,
    category: String,
    size_label: String,
    n_elements: usize,
    seguru_us: f64,
    cuda_us: f64,
}

impl BenchResult {
    fn ratio(&self) -> f64 {
        self.seguru_us / self.cuda_us
    }
    fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.2},{:.2},{:.4}",
            self.kernel,
            self.category,
            self.size_label,
            self.n_elements,
            self.seguru_us,
            self.cuda_us,
            self.ratio()
        )
    }
}

fn main() {
    let mut results: Vec<BenchResult> = Vec::new();

    cuda_ctx(0, |ctx, m| {
        // ── Elementwise + GELU + Scalar ─────────────────────────────────
        for &(label, n) in &[("small", 4096usize), ("large", 1_048_576usize)] {
            let input = gen_input(n);
            let mut output = vec![0.0f32; n];
            let block = 256u32;
            let grid = ((n as u32) + block - 1) / block;

            // Macro for simple elementwise kernels: (name, category, launch_expr, cuda_expr)
            macro_rules! bench_elt {
                ($name:expr, $cat:expr, $launch:ident) => {{
                    let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                    let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                    // warmup
                    for _ in 0..WARMUP {
                        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, n as u32).unwrap();
                        ctx.sync().unwrap();
                    }
                    let mut times = Vec::new();
                    for _ in 0..ITERS {
                        let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                        let start = Instant::now();
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, n as u32).unwrap();
                        ctx.sync().unwrap();
                        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                    }
                    median(&mut times)
                }};
            }

            // relu_forward
            let seguru_us = bench_elt!("relu_forward", "elementwise", relu_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_relu_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "relu_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // sigmoid_forward
            let seguru_us = bench_elt!("sigmoid_forward", "elementwise", sigmoid_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_sigmoid_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "sigmoid_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // tanh_forward
            let seguru_us = bench_elt!("tanh_forward", "elementwise", tanh_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_tanh_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "tanh_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // swish_forward
            let seguru_us = bench_elt!("swish_forward", "elementwise", swish_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_swish_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "swish_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // selu_forward
            let seguru_us = bench_elt!("selu_forward", "elementwise", selu_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_selu_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "selu_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // hard_sigmoid_forward
            let seguru_us = bench_elt!("hard_sigmoid_forward", "elementwise", hard_sigmoid_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_hard_sigmoid_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "hard_sigmoid_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // softplus_forward
            let seguru_us = bench_elt!("softplus_forward", "elementwise", softplus_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_softplus_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "softplus_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // softsign_forward
            let seguru_us = bench_elt!("softsign_forward", "elementwise", softsign_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_softsign_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "softsign_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // leaky_relu_forward (extra param: alpha)
            let alpha = 0.01f32;
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    leaky_relu_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, alpha).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    leaky_relu_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, alpha).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_leaky_relu_forward(
                        input.as_ptr(), output.as_mut_ptr(), n as i32, alpha,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "leaky_relu_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // elu_forward (extra param: alpha)
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    elu_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, alpha).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    elu_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, alpha).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_elu_forward(
                        input.as_ptr(), output.as_mut_ptr(), n as i32, alpha,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "elu_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // hard_tanh_forward (extra: min_val, max_val)
            {
                let min_val = -1.0f32;
                let max_val = 1.0f32;
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    hard_tanh_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, min_val, max_val).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    hard_tanh_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32, min_val, max_val).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_hard_tanh_forward(
                        input.as_ptr(), output.as_mut_ptr(), n as i32, min_val, max_val,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "hard_tanh_forward".into(), category: "elementwise".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // GELU variants
            let seguru_us = bench_elt!("gelu_forward", "gelu", gelu_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_gelu_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "gelu_forward".into(), category: "gelu".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            let seguru_us = bench_elt!("mingpt_new_gelu_forward", "gelu", mingpt_new_gelu_forward);
            let cuda_us = unsafe {
                cuda_ffi::bench_mingpt_new_gelu_forward(
                    input.as_ptr(), output.as_mut_ptr(), n as i32,
                    grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "mingpt_new_gelu_forward".into(), category: "gelu".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });

            // scalar_multiply
            {
                let s = 2.5f32;
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    scalar_multiply::launch(config, ctx, m, &d_in, &mut d_out, s, n as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    scalar_multiply::launch(config, ctx, m, &d_in, &mut d_out, s, n as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_scalar_multiply(
                        input.as_ptr(), output.as_mut_ptr(), s, n as i32,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "scalar_multiply".into(), category: "scalar".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }
        }

        // ── Matmul 2D ───────────────────────────────────────────────────
        for &(label, mm, nn, kk) in &[("small", 64usize, 64, 64), ("large", 1024usize, 1024, 1024)] {
            let block_dim = 16u32;
            let grid_x = ((nn as u32) + block_dim - 1) / block_dim;
            let grid_y = ((mm as u32) + block_dim - 1) / block_dim;
            let n_out = mm * nn;

            // matmul_forward: A(M×K), B(K×N) → C(M×N)
            {
                let a = gen_input(mm * kk);
                let b = gen_input(kk * nn);
                let mut c = vec![0.0f32; n_out];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    matmul_forward::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    let start = Instant::now();
                    matmul_forward::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_matmul_forward(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        mm as i32, nn as i32, kk as i32,
                        grid_x as i32, grid_y as i32, block_dim as i32, block_dim as i32,
                        WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "matmul_forward".into(), category: "matmul_2d".into(), size_label: label.into(), n_elements: n_out, seguru_us, cuda_us });
            }

            // matmul_transposed_a: A stored as K×M, B as K×N
            {
                let a = gen_input(kk * mm);
                let b = gen_input(kk * nn);
                let mut c = vec![0.0f32; n_out];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    matmul_transposed_a::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    let start = Instant::now();
                    matmul_transposed_a::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_matmul_transposed_a(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        mm as i32, nn as i32, kk as i32,
                        grid_x as i32, grid_y as i32, block_dim as i32, block_dim as i32,
                        WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "matmul_transposed_a".into(), category: "matmul_2d".into(), size_label: label.into(), n_elements: n_out, seguru_us, cuda_us });
            }

            // matmul_transposed_b: A as M×K, B stored as N×K
            {
                let a = gen_input(mm * kk);
                let b = gen_input(nn * kk);
                let mut c = vec![0.0f32; n_out];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    matmul_transposed_b::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    let start = Instant::now();
                    matmul_transposed_b::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_matmul_transposed_b(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        mm as i32, nn as i32, kk as i32,
                        grid_x as i32, grid_y as i32, block_dim as i32, block_dim as i32,
                        WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "matmul_transposed_b".into(), category: "matmul_2d".into(), size_label: label.into(), n_elements: n_out, seguru_us, cuda_us });
            }

            // matmul_transposed_both: A stored as K×M, B stored as N×K
            {
                let a = gen_input(kk * mm);
                let b = gen_input(nn * kk);
                let mut c = vec![0.0f32; n_out];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    matmul_transposed_both::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid_x, grid_y, 1, block_dim, block_dim, 1, 0);
                    let start = Instant::now();
                    matmul_transposed_both::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_matmul_transposed_both(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        mm as i32, nn as i32, kk as i32,
                        grid_x as i32, grid_y as i32, block_dim as i32, block_dim as i32,
                        WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "matmul_transposed_both".into(), category: "matmul_2d".into(), size_label: label.into(), n_elements: n_out, seguru_us, cuda_us });
            }
        }

        // ── Batched matmul + tensor3d_matmul ────────────────────────────
        for &(label, batch, mm, nn, kk) in &[("small", 4usize, 32, 32, 32), ("large", 16usize, 256, 256, 256)] {
            let total = batch * mm * nn;
            let block = 256u32;
            let grid = ((total as u32) + block - 1) / block;

            // matmul_batched
            {
                let a = gen_input(batch * mm * kk);
                let b = gen_input(batch * kk * nn);
                let mut c = vec![0.0f32; total];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    matmul_batched::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32, batch as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    matmul_batched::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32, batch as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_matmul_batched(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        batch as i32, mm as i32, nn as i32, kk as i32,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "matmul_batched".into(), category: "batched_matmul".into(), size_label: label.into(), n_elements: total, seguru_us, cuda_us });
            }

            // tensor3d_matmul
            {
                let a = gen_input(batch * mm * kk);
                let b = gen_input(batch * kk * nn);
                let mut c = vec![0.0f32; total];
                let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
                let d_b = ctx.new_tensor_view(b.as_slice()).unwrap();
                let mut d_c = ctx.new_tensor_view(c.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    tensor3d_matmul::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32, batch as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    let start = Instant::now();
                    tensor3d_matmul::launch(config, ctx, m, &d_a, &d_b, &mut d_c, mm as u32, nn as u32, kk as u32, batch as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_tensor3d_matmul(
                        a.as_ptr(), b.as_ptr(), c.as_mut_ptr(),
                        batch as i32, mm as i32, nn as i32, kk as i32,
                        grid as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "tensor3d_matmul".into(), category: "batched_matmul".into(), size_label: label.into(), n_elements: total, seguru_us, cuda_us });
            }
        }

        // ── Matvec ──────────────────────────────────────────────────────
        for &(label, mm, nn) in &[("small", 64usize, 64), ("large", 4096usize, 4096)] {
            let block = 256u32;
            let grid = ((mm as u32) + block - 1) / block;
            let a = gen_input(mm * nn);
            let x = gen_input(nn);
            let mut y = vec![0.0f32; mm];

            let d_a = ctx.new_tensor_view(a.as_slice()).unwrap();
            let d_x = ctx.new_tensor_view(x.as_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(y.as_mut_slice()).unwrap();
            for _ in 0..WARMUP {
                let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                matvec_forward::launch(config, ctx, m, &d_a, &d_x, &mut d_y, mm as u32, nn as u32).unwrap();
                ctx.sync().unwrap();
            }
            let mut times = Vec::new();
            for _ in 0..ITERS {
                let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                let start = Instant::now();
                matvec_forward::launch(config, ctx, m, &d_a, &d_x, &mut d_y, mm as u32, nn as u32).unwrap();
                ctx.sync().unwrap();
                times.push(start.elapsed().as_nanos() as f64 / 1000.0);
            }
            let seguru_us = median(&mut times);
            let cuda_us = unsafe {
                cuda_ffi::bench_matvec_forward(
                    a.as_ptr(), x.as_ptr(), y.as_mut_ptr(),
                    mm as i32, nn as i32, grid as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "matvec_forward".into(), category: "matvec".into(), size_label: label.into(), n_elements: mm, seguru_us, cuda_us });
        }

        // ── Reduction (sum, mean, max, min) ─────────────────────────────
        for &(label, batch, dim) in &[("small", 64usize, 256usize), ("large", 1024usize, 4096usize)] {
            let n = batch * dim;
            let input = gen_input(n);
            let mut output = vec![0.0f32; batch];
            let block = 128u32;
            let smem = block * 4;

            macro_rules! bench_reduce {
                ($name:expr, $launch:ident, $ffi:ident) => {{
                    let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                    let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                    for _ in 0..WARMUP {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                    }
                    let mut times = Vec::new();
                    for _ in 0..ITERS {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        let start = Instant::now();
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                    }
                    let seguru_us = median(&mut times);
                    let cuda_us = unsafe {
                        cuda_ffi::$ffi(
                            input.as_ptr(), output.as_mut_ptr(),
                            batch as i32, dim as i32, block as i32, WARMUP, ITERS,
                        ) as f64
                    };
                    results.push(BenchResult { kernel: $name.into(), category: "reduction".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
                }};
            }

            bench_reduce!("sum_reduce", sum_reduce, bench_sum_reduce);
            bench_reduce!("mean_reduce", mean_reduce, bench_mean_reduce);
            bench_reduce!("max_reduce", max_reduce, bench_max_reduce);
            bench_reduce!("min_reduce", min_reduce, bench_min_reduce);
        }

        // ── Argreduce (argmax, argmin) — output is u32 ──────────────────
        for &(label, batch, dim) in &[("small", 64usize, 256usize), ("large", 1024usize, 4096usize)] {
            let n = batch * dim;
            let input = gen_input(n);
            let mut output_u32 = vec![0u32; batch];
            let block = 64u32;
            let smem = block * 8;

            macro_rules! bench_argreduce {
                ($name:expr, $launch:ident, $ffi:ident) => {{
                    let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                    let mut d_out = ctx.new_tensor_view(output_u32.as_mut_slice()).unwrap();
                    for _ in 0..WARMUP {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                    }
                    let mut times = Vec::new();
                    for _ in 0..ITERS {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        let start = Instant::now();
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                    }
                    let seguru_us = median(&mut times);
                    let cuda_us = unsafe {
                        cuda_ffi::$ffi(
                            input.as_ptr(), output_u32.as_mut_ptr(),
                            batch as i32, dim as i32, block as i32, WARMUP, ITERS,
                        ) as f64
                    };
                    results.push(BenchResult { kernel: $name.into(), category: "argreduce".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
                }};
            }

            bench_argreduce!("argmax_reduce", argmax_reduce, bench_argmax_reduce);
            bench_argreduce!("argmin_reduce", argmin_reduce, bench_argmin_reduce);
        }

        // ── Softmax ─────────────────────────────────────────────────────
        for &(label, batch, dim) in &[("small", 64usize, 256usize), ("large", 1024usize, 4096usize)] {
            let n = batch * dim;
            let input = gen_input(n);
            let mut output = vec![0.0f32; n];
            let block = 64u32;
            let smem = block * 4;

            macro_rules! bench_softmax {
                ($name:expr, $launch:ident, $ffi:ident) => {{
                    let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                    let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                    for _ in 0..WARMUP {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                    }
                    let mut times = Vec::new();
                    for _ in 0..ITERS {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                        let start = Instant::now();
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                    }
                    let seguru_us = median(&mut times);
                    let cuda_us = unsafe {
                        cuda_ffi::$ffi(
                            input.as_ptr(), output.as_mut_ptr(),
                            batch as i32, dim as i32, block as i32, WARMUP, ITERS,
                        ) as f64
                    };
                    results.push(BenchResult { kernel: $name.into(), category: "softmax".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
                }};
            }

            bench_softmax!("softmax_forward", softmax_forward, bench_softmax_forward);
            bench_softmax!("log_softmax_forward", log_softmax_forward, bench_log_softmax_forward);
        }

        // ── Norm (row-wise: rms, l1, l2, layer) ────────────────────────
        for &(label, batch, dim) in &[("small", 64usize, 256usize), ("large", 1024usize, 4096usize)] {
            let n = batch * dim;
            let input = gen_input(n);
            let mut output = vec![0.0f32; n];
            let block = 128u32;
            let smem = block * 4;
            let eps = 1e-5f32;

            // rms_norm_forward
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    rms_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32, eps).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    rms_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32, eps).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_rms_norm_forward(
                        input.as_ptr(), output.as_mut_ptr(),
                        batch as i32, dim as i32, eps, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "rms_norm_forward".into(), category: "norm".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // l1_norm_forward
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    l1_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    l1_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_l1_norm_forward(
                        input.as_ptr(), output.as_mut_ptr(),
                        batch as i32, dim as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "l1_norm_forward".into(), category: "norm".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // l2_norm_forward
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    l2_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    l2_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_l2_norm_forward(
                        input.as_ptr(), output.as_mut_ptr(),
                        batch as i32, dim as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "l2_norm_forward".into(), category: "norm".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // layer_norm_forward
            {
                let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    layer_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32, eps).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(batch as u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    layer_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, dim as u32, eps).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_layer_norm_forward(
                        input.as_ptr(), output.as_mut_ptr(),
                        batch as i32, dim as i32, eps, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "layer_norm_forward".into(), category: "norm".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }
        }

        // ── Frobenius norm (global reduction) ───────────────────────────
        for &(label, n) in &[("small", 4096usize), ("large", 1_048_576usize)] {
            let input = gen_input(n);
            let mut output = vec![0.0f32; 1];
            let block = 256u32;
            let smem = block * 4;

            let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
            let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
            for _ in 0..WARMUP {
                let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                frobenius_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32).unwrap();
                ctx.sync().unwrap();
            }
            let mut times = Vec::new();
            for _ in 0..ITERS {
                let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                let start = Instant::now();
                frobenius_norm_forward::launch(config, ctx, m, &d_in, &mut d_out, n as u32).unwrap();
                ctx.sync().unwrap();
                times.push(start.elapsed().as_nanos() as f64 / 1000.0);
            }
            let seguru_us = median(&mut times);
            let cuda_us = unsafe {
                cuda_ffi::bench_frobenius_norm_forward(
                    input.as_ptr(), output.as_mut_ptr(),
                    n as i32, block as i32, WARMUP, ITERS,
                ) as f64
            };
            results.push(BenchResult { kernel: "frobenius_norm_forward".into(), category: "norm".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
        }

        // ── Loss (mse, huber, kl_div, hinge) ────────────────────────────
        for &(label, n) in &[("small", 4096usize), ("large", 1_048_576usize)] {
            let predictions = gen_input(n);
            let targets = gen_targets(n);
            let mut output = vec![0.0f32; 1];
            let block = 256u32;
            let smem = block * 4;

            // mse_loss_forward
            {
                let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
                let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    mse_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    mse_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_mse_loss_forward(
                        predictions.as_ptr(), targets.as_ptr(), output.as_mut_ptr(),
                        n as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "mse_loss_forward".into(), category: "loss".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // huber_loss_forward
            {
                let delta = 1.0f32;
                let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
                let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    huber_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32, delta).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    huber_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32, delta).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_huber_loss_forward(
                        predictions.as_ptr(), targets.as_ptr(), output.as_mut_ptr(),
                        n as i32, delta, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "huber_loss_forward".into(), category: "loss".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // kl_div_loss_forward
            {
                let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
                let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    kl_div_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    kl_div_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_kl_div_loss_forward(
                        predictions.as_ptr(), targets.as_ptr(), output.as_mut_ptr(),
                        n as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "kl_div_loss_forward".into(), category: "loss".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }

            // hinge_loss_forward
            {
                let d_pred = ctx.new_tensor_view(predictions.as_slice()).unwrap();
                let d_tgt = ctx.new_tensor_view(targets.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                for _ in 0..WARMUP {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    hinge_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                }
                let mut times = Vec::new();
                for _ in 0..ITERS {
                    let config = gpu_host::gpu_config!(1u32, 1, 1, block, 1, 1, smem);
                    let start = Instant::now();
                    hinge_loss_forward::launch(config, ctx, m, &d_pred, &d_tgt, &mut d_out, n as u32).unwrap();
                    ctx.sync().unwrap();
                    times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                }
                let seguru_us = median(&mut times);
                let cuda_us = unsafe {
                    cuda_ffi::bench_hinge_loss_forward(
                        predictions.as_ptr(), targets.as_ptr(), output.as_mut_ptr(),
                        n as i32, block as i32, WARMUP, ITERS,
                    ) as f64
                };
                results.push(BenchResult { kernel: "hinge_loss_forward".into(), category: "loss".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
            }
        }

        // ── Cumulative (cumsum, cumprod, cumsum_reverse, cumsum_exclusive) ─
        for &(label, batch, dim) in &[("small", 64usize, 256usize), ("large", 1024usize, 4096usize)] {
            let n = batch * dim;
            let input = gen_input(n);
            let mut output = vec![0.0f32; n];

            macro_rules! bench_cumulative {
                ($name:expr, $launch:ident, $ffi:ident) => {{
                    let d_in = ctx.new_tensor_view(input.as_slice()).unwrap();
                    let mut d_out = ctx.new_tensor_view(output.as_mut_slice()).unwrap();
                    for _ in 0..WARMUP {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, 1u32, 1, 1, 0);
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                    }
                    let mut times = Vec::new();
                    for _ in 0..ITERS {
                        let config = gpu_host::gpu_config!(batch as u32, 1, 1, 1u32, 1, 1, 0);
                        let start = Instant::now();
                        $launch::launch(config, ctx, m, &d_in, &mut d_out, dim as u32).unwrap();
                        ctx.sync().unwrap();
                        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
                    }
                    let seguru_us = median(&mut times);
                    let cuda_us = unsafe {
                        cuda_ffi::$ffi(
                            input.as_ptr(), output.as_mut_ptr(),
                            batch as i32, dim as i32, WARMUP, ITERS,
                        ) as f64
                    };
                    results.push(BenchResult { kernel: $name.into(), category: "cumulative".into(), size_label: label.into(), n_elements: n, seguru_us, cuda_us });
                }};
            }

            bench_cumulative!("cumsum_forward", cumsum_forward, bench_cumsum_forward);
            bench_cumulative!("cumprod_forward", cumprod_forward, bench_cumprod_forward);
            bench_cumulative!("cumsum_reverse_forward", cumsum_reverse_forward, bench_cumsum_reverse_forward);
            bench_cumulative!("cumsum_exclusive_forward", cumsum_exclusive_forward, bench_cumsum_exclusive_forward);
        }
    });

    // ── Output CSV to stdout ────────────────────────────────────────
    println!("kernel,category,size_label,n_elements,seguru_us,cuda_us,ratio");
    for r in &results {
        println!("{}", r.csv_row());
    }

    // ── Summary table to stderr ─────────────────────────────────────
    eprintln!("\n{:=<80}", "");
    eprintln!("  BENCHMARK SUMMARY ({} measurements)", results.len());
    eprintln!("{:=<80}", "");

    // Per-category averages
    let mut cat_ratios: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &results {
        cat_ratios.entry(r.category.clone()).or_default().push(r.ratio());
    }
    eprintln!("\n{:<25} {:>10} {:>10}", "Category", "Avg Ratio", "Count");
    eprintln!("{:-<50}", "");
    for (cat, ratios) in &cat_ratios {
        let avg: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
        eprintln!("{:<25} {:>10.4} {:>10}", cat, avg, ratios.len());
    }

    // Overall
    let all_ratios: Vec<f64> = results.iter().map(|r| r.ratio()).collect();
    let overall_avg = all_ratios.iter().sum::<f64>() / all_ratios.len() as f64;
    eprintln!("\n  Overall average ratio (SeGuRu/CUDA): {:.4}", overall_avg);

    // Best/worst
    let best = results.iter().min_by(|a, b| a.ratio().partial_cmp(&b.ratio()).unwrap()).unwrap();
    let worst = results.iter().max_by(|a, b| a.ratio().partial_cmp(&b.ratio()).unwrap()).unwrap();
    eprintln!(
        "  Best:  {} ({}, {}) — ratio {:.4}",
        best.kernel, best.size_label, best.category, best.ratio()
    );
    eprintln!(
        "  Worst: {} ({}, {}) — ratio {:.4}",
        worst.kernel, worst.size_label, worst.category, worst.ratio()
    );
    eprintln!("{:=<80}\n", "");
}
