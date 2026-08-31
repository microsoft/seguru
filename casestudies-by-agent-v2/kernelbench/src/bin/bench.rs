//! Kernel-only benchmark for every operator in the crate.
//!
//! Timing follows `aes/src/bin/bench.rs`: buffers are staged on the device once,
//! the kernel is launched `WARMUP` times, `ctx.sync()` drains the queue, and
//! then `ITERS` launches are timed as a block and averaged. No host/device
//! transfer happens inside the timed region.
//!
//! "GB/s" is the algorithmic minimum traffic (what the operator *must* read and
//! write) divided by the measured time, so it can be compared against the
//! A100's ~1.5-1.9 TB/s of achievable HBM bandwidth. Kernels that revisit a row
//! (softmax, layer norm) move more than that through the cache hierarchy.

use std::time::Instant;

use gpu::Float4;
use gpu_host::gpu_config;
use kernelbench_gpu::activation::*;
use kernelbench_gpu::loss::*;
use kernelbench_gpu::norm::*;
use kernelbench_gpu::pool::*;
use kernelbench_gpu::reduce::*;
use kernelbench_gpu::testkit::sample;
use kernelbench_gpu::util::*;

const WARMUP: u32 = 20;
const ITERS: u32 = 200;

struct Row {
    op: &'static str,
    shape: String,
    us: f64,
    bytes: usize,
}

fn main() {
    let shapes: &[(usize, usize)] = &[(1024, 1024), (4096, 1024)];
    let mut rows: Vec<Row> = Vec::new();

    for &(nrows, ncols) in shapes {
        let n = nrows * ncols;
        let label = format!("{nrows}x{ncols}");
        let x = sample(n, 1);
        let x2 = sample(n, 2);
        let weight = sample(ncols, 3);
        let bias = sample(ncols, 4);

        let plan = RowPlan::new(nrows, ncols);
        let padded_rows = pad_rows(&x, nrows, ncols, plan.stride, 0.0);
        let hx_row = to_float4_padded(&padded_rows, padded_rows.len());
        let hw = to_float4_padded(&weight, plan.stride);
        let hb = to_float4_padded(&bias, plan.stride);

        let ew_grid = n.div_ceil(ELEMS_PER_CTA) as u32;
        let ew_padded = ew_grid as usize * ELEMS_PER_CTA;
        let hx_ew = to_float4_padded(&x, ew_padded);
        let hx2_ew = to_float4_padded(&x2, ew_padded);

        gpu_host::cuda_ctx(0, |ctx, m| {
            let time = |f: &mut dyn FnMut()| -> f64 {
                for _ in 0..WARMUP {
                    f();
                }
                ctx.sync().unwrap();
                let t = Instant::now();
                for _ in 0..ITERS {
                    f();
                }
                ctx.sync().unwrap();
                t.elapsed().as_secs_f64() * 1e6 / ITERS as f64
            };

            // ---------------- elementwise ----------------
            let d_ew = ctx.new_tensor_view::<[Float4]>(&hx_ew).unwrap();
            let d_ew2 = ctx.new_tensor_view::<[Float4]>(&hx2_ew).unwrap();
            let zeros_ew = vec![Float4::default(); hx_ew.len()];
            let mut d_ew_out = ctx.new_tensor_view::<[Float4]>(&zeros_ew).unwrap();

            macro_rules! bench_ew {
                ($op:literal, $kernel:ident) => {{
                    let us = time(&mut || {
                        let cfg = gpu_config!(ew_grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
                        $kernel::launch(cfg, ctx, m, &d_ew, &mut d_ew_out).unwrap();
                    });
                    rows.push(Row { op: $op, shape: label.clone(), us, bytes: 2 * n * 4 });
                }};
            }

            bench_ew!("relu", relu_kernel);
            bench_ew!("gelu", gelu_kernel);
            bench_ew!("sigmoid", sigmoid_kernel);
            bench_ew!("tanh", tanh_kernel);
            bench_ew!("swish", swish_kernel);
            bench_ew!("softplus", softplus_kernel);
            {
                let us = time(&mut || {
                    let cfg = gpu_config!(ew_grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
                    leaky_relu_kernel::launch(cfg, ctx, m, &d_ew, &mut d_ew_out, 0.01).unwrap();
                });
                rows.push(Row { op: "leaky_relu", shape: label.clone(), us, bytes: 2 * n * 4 });
            }

            // ---------------- row -> row ----------------
            let d_row = ctx.new_tensor_view::<[Float4]>(&hx_row).unwrap();
            let d_w = ctx.new_tensor_view::<[Float4]>(&hw).unwrap();
            let d_b = ctx.new_tensor_view::<[Float4]>(&hb).unwrap();
            let zeros_row = vec![Float4::default(); hx_row.len()];
            let mut d_row_out = ctx.new_tensor_view::<[Float4]>(&zeros_row).unwrap();
            let grid = nrows as u32;

            macro_rules! bench_row {
                ($op:literal, $kernel:ident) => {{
                    let us = time(&mut || {
                        let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                        $kernel::launch(cfg, ctx, m, &d_row, &mut d_row_out, plan.cols4, plan.items)
                            .unwrap();
                    });
                    rows.push(Row { op: $op, shape: label.clone(), us, bytes: 2 * n * 4 });
                }};
            }

            bench_row!("softmax", softmax_kernel);
            bench_row!("log_softmax", log_softmax_kernel);
            bench_row!("cumsum", cumsum_kernel);

            macro_rules! bench_scale {
                ($op:literal, $kernel:ident, $inv:expr) => {{
                    let us = time(&mut || {
                        let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                        $kernel::launch(
                            cfg, ctx, m, &d_row, &mut d_row_out, plan.cols4, plan.items, $inv, 1e-5,
                        )
                        .unwrap();
                    });
                    rows.push(Row { op: $op, shape: label.clone(), us, bytes: 2 * n * 4 });
                }};
            }

            bench_scale!("rms_norm", rms_norm_kernel, 1.0 / ncols as f32);
            bench_scale!("l1_norm", l1_norm_kernel, 1.0);
            bench_scale!("l2_norm", l2_norm_kernel, 1.0);

            {
                let inv_n = 1.0 / ncols as f32;
                let us = time(&mut || {
                    let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    layer_norm_kernel::launch(
                        cfg,
                        ctx,
                        m,
                        &d_row,
                        &d_w,
                        &d_b,
                        &mut d_row_out,
                        plan.cols4,
                        plan.items,
                        inv_n,
                        1e-5,
                    )
                    .unwrap();
                });
                rows.push(Row { op: "layer_norm", shape: label.clone(), us, bytes: 2 * n * 4 });
            }

            // ---------------- row -> scalar ----------------
            let scalars = vec![0.0f32; nrows];
            let mut d_scalar = ctx.new_tensor_view::<[f32]>(&scalars).unwrap();
            let idx = vec![0i32; nrows];
            let mut d_idx = ctx.new_tensor_view::<[i32]>(&idx).unwrap();
            let red_bytes = n * 4 + nrows * 4;

            {
                let us = time(&mut || {
                    let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    sum_dim_kernel::launch(
                        cfg, ctx, m, &d_row, &mut d_scalar, plan.cols4, plan.items, 1.0,
                    )
                    .unwrap();
                });
                rows.push(Row { op: "sum_dim", shape: label.clone(), us, bytes: red_bytes });
            }
            {
                let scale = 1.0 / ncols as f32;
                let us = time(&mut || {
                    let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    sum_dim_kernel::launch(
                        cfg, ctx, m, &d_row, &mut d_scalar, plan.cols4, plan.items, scale,
                    )
                    .unwrap();
                });
                rows.push(Row { op: "mean_dim", shape: label.clone(), us, bytes: red_bytes });
            }
            {
                let us = time(&mut || {
                    let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    max_dim_kernel::launch(
                        cfg, ctx, m, &d_row, &mut d_scalar, plan.cols4, plan.items,
                    )
                    .unwrap();
                });
                rows.push(Row { op: "max_dim", shape: label.clone(), us, bytes: red_bytes });
            }
            {
                let cols = ncols as u32;
                let us = time(&mut || {
                    let cfg = gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    argmax_dim_kernel::launch(
                        cfg, ctx, m, &d_row, &mut d_idx, plan.cols4, plan.items, cols,
                    )
                    .unwrap();
                });
                rows.push(Row { op: "argmax_dim", shape: label.clone(), us, bytes: red_bytes });
            }

            // ---------------- loss ----------------
            {
                let partials = vec![0.0f32; ew_grid as usize];
                let mut d_p = ctx.new_tensor_view::<[f32]>(&partials).unwrap();
                let out0 = vec![0.0f32; 1];
                let mut d_out = ctx.new_tensor_view::<[f32]>(&out0).unwrap();
                let inv_n = 1.0 / n as f32;
                let us = time(&mut || {
                    let cfg = gpu_config!(ew_grid, 1, 1, @const EW_BLOCK, 1, 1, 0);
                    mse_partial_kernel::launch(cfg, ctx, m, &d_ew, &d_ew2, &mut d_p).unwrap();
                    let cfg = gpu_config!(@const 1, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                    sum_partials_kernel::launch(cfg, ctx, m, &d_p, &mut d_out, ew_grid, inv_n)
                        .unwrap();
                });
                rows.push(Row { op: "mse_loss", shape: label.clone(), us, bytes: 2 * n * 4 });
            }

            // ---------------- pooling ----------------
            {
                let l_in = ncols;
                let (k, s) = (4usize, 4usize);
                let l_out = out_len(l_in, k, s);
                let n_out = nrows * l_out;
                let pool_grid = n_out.div_ceil(OUT_PER_CTA) as u32;
                let d_pool_in = ctx.new_tensor_view::<[f32]>(&x).unwrap();
                let pool_out = vec![0.0f32; pool_grid as usize * OUT_PER_CTA];
                let mut d_pool_out = ctx.new_tensor_view::<[f32]>(&pool_out).unwrap();
                let us = time(&mut || {
                    let cfg = gpu_config!(pool_grid, 1, 1, @const POOL_BLOCK, 1, 1, 0);
                    max_pool1d_kernel::launch(
                        cfg,
                        ctx,
                        m,
                        &d_pool_in,
                        &mut d_pool_out,
                        k as u32,
                        s as u32,
                        l_in as u32,
                        l_out as u32,
                        n_out as u32,
                    )
                    .unwrap();
                });
                rows.push(Row {
                    op: "max_pool1d k=4 s=4",
                    shape: label.clone(),
                    us,
                    bytes: (n + n_out) * 4,
                });
            }
        });
        eprintln!("done {label}");
    }

    println!("\n| Operator | Shape | Time (us) | GB/s |");
    println!("|---|---|---|---|");
    for r in &rows {
        let gbps = r.bytes as f64 / (r.us * 1e-6) / 1e9;
        println!("| {} | {} | {:.1} | {:.0} |", r.op, r.shape, r.us, gbps);

        // Some `op` labels embed extra config (e.g. "max_pool1d k=4 s=4");
        // split that off into the parameter token, keeping the workload name bare.
        let mut parts = r.op.split_whitespace();
        let workload = parts.next().unwrap_or(r.op);
        let extra: Vec<&str> = parts.collect();
        let parameter = if extra.is_empty() {
            r.shape.clone()
        } else {
            format!("{}/{}", r.shape, extra.join("/"))
        };
        csv_row("kernelbench", workload, &parameter, "seguru", "time", r.us, "us");
        csv_row("kernelbench", workload, &parameter, "seguru", "throughput", gbps, "GB/s");
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
