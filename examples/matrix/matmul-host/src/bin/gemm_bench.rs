//! GEMM benchmark: `C = alpha * A * B + beta * C` at the same 1024^3 shape and
//! 16x16 tiling that cuda-oxide's `gemm` and `tiled_gemm` examples use, so the
//! two safe-GPU-Rust designs can be compared against a common cuBLAS reference.

use std::ffi::c_void;
use std::time::Instant;

use gpu_host::cuda_ctx;
use matmul_gpu::{TILE, sgemm_naive_kernel, sgemm_tiled_kernel};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

const WARMUP: usize = 10;
const ITERS: usize = 100;
const REPS: usize = 5;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
}

const CUBLAS_OP_N: i32 = 0;

fn gflops(ms: f64) -> f64 {
    (2.0 * M as f64 * N as f64 * K as f64) / (ms / 1e3) / 1e9
}

/// Mean and half-width of the 95% confidence interval over `xs`.
fn mean_ci(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    // t_{0.975} for 4 degrees of freedom; REPS is fixed at 5.
    (mean, 2.776 * (var / n).sqrt())
}

fn reference(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for p in 0..K {
                sum += a[i * K + p] * b[p * N + j];
            }
            c[i * N + j] = ALPHA * sum;
        }
    }
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs() / w.abs().max(1.0);
        if d > worst {
            worst = d;
        }
    }
    worst
}

fn main() {
    // cuda-oxide's examples/gemm and examples/tiled_gemm use these fill patterns.
    let mut a = vec![0.0f32; M * K];
    let mut b = vec![0.0f32; K * N];
    for i in 0..M {
        for j in 0..K {
            a[i * K + j] = ((i + j) % 10) as f32 * 0.1;
        }
    }
    for i in 0..K {
        for j in 0..N {
            b[i * N + j] = ((i * j) % 10) as f32 * 0.1;
        }
    }

    println!("computing CPU reference for {M}x{K} * {K}x{N} ...");
    let mut want = vec![0.0f32; M * N];
    reference(&a, &b, &mut want);

    let mut naive = vec![0.0f32; M * N];
    let mut tiled = vec![0.0f32; M * N];
    let mut cublas = vec![0.0f32; M * N];
    let mut naive_ms = Vec::new();
    let mut tiled_ms = Vec::new();
    let mut cublas_ms = Vec::new();

    cuda_ctx(0, |ctx, module| {
        let d_a = ctx.new_tensor_view::<[f32]>(&a).expect("alloc a");
        let d_b = ctx.new_tensor_view::<[f32]>(&b).expect("alloc b");
        let zeros = vec![0.0f32; M * N];
        let mut d_c = ctx.new_tensor_view::<[f32]>(&zeros).expect("alloc c");

        let grid = (N.div_ceil(TILE)) as u32;
        let block = TILE as u32;
        let cfg = || gpu_host::gpu_config!(grid, grid, 1, block, block, 1, 0);

        for _ in 0..WARMUP {
            sgemm_naive_kernel::launch(
                cfg(), ctx, module, M, N, K, ALPHA, &d_a, &d_b, BETA, &mut d_c,
            )
            .expect("naive launch");
        }
        ctx.sync().expect("sync");
        for _ in 0..REPS {
            let start = Instant::now();
            for _ in 0..ITERS {
                sgemm_naive_kernel::launch(
                    cfg(), ctx, module, M, N, K, ALPHA, &d_a, &d_b, BETA, &mut d_c,
                )
                .expect("naive launch");
            }
            ctx.sync().expect("sync");
            naive_ms.push(start.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        d_c.copy_to_host(&mut naive).expect("copy naive");

        for _ in 0..WARMUP {
            sgemm_tiled_kernel::launch(
                cfg(), ctx, module, M, N, K, ALPHA, &d_a, &d_b, BETA, &mut d_c,
            )
            .expect("tiled launch");
        }
        ctx.sync().expect("sync");
        for _ in 0..REPS {
            let start = Instant::now();
            for _ in 0..ITERS {
                sgemm_tiled_kernel::launch(
                    cfg(), ctx, module, M, N, K, ALPHA, &d_a, &d_b, BETA, &mut d_c,
                )
                .expect("tiled launch");
            }
            ctx.sync().expect("sync");
            tiled_ms.push(start.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        d_c.copy_to_host(&mut tiled).expect("copy tiled");

        let mut handle: *mut c_void = std::ptr::null_mut();
        assert_eq!(unsafe { cublasCreate_v2(&mut handle) }, 0, "cublasCreate");
        // cuBLAS is column-major, so computing row-major C = A*B means asking it
        // for C^T = B^T * A^T: swap the operands and pass row widths as leading
        // dimensions.
        let pa = d_a.as_devptr() as *const f32;
        let pb = d_b.as_devptr() as *const f32;
        let pc = d_c.as_devptr() as *mut f32;
        let sgemm = || {
            let ret = unsafe {
                cublasSgemm_v2(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    N as i32,
                    M as i32,
                    K as i32,
                    &ALPHA,
                    pb,
                    N as i32,
                    pa,
                    K as i32,
                    &BETA,
                    pc,
                    N as i32,
                )
            };
            assert_eq!(ret, 0, "cublasSgemm");
        };
        for _ in 0..WARMUP {
            sgemm();
        }
        ctx.sync().expect("sync");
        for _ in 0..REPS {
            let start = Instant::now();
            for _ in 0..ITERS {
                sgemm();
            }
            ctx.sync().expect("sync");
            cublas_ms.push(start.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
        }
        d_c.copy_to_host(&mut cublas).expect("copy cublas");
        assert_eq!(unsafe { cublasDestroy_v2(handle) }, 0, "cublasDestroy");
    });

    println!("\n{:<10} {:>12}  {:>12}", "kernel", "time (ms)", "GFLOP/s");
    let mut means = Vec::new();
    for (name, samples) in
        [("naive", &naive_ms), ("tiled", &tiled_ms), ("cublas", &cublas_ms)]
    {
        let (mean, ci) = mean_ci(samples);
        println!("{name:<10} {mean:8.4} +/- {ci:6.4}  {:12.1}", gflops(mean));
        means.push(mean);
    }

    println!("\nmax relative error vs CPU reference");
    for (name, got) in [("naive", &naive), ("tiled", &tiled), ("cublas", &cublas)] {
        let e = max_rel_err(got, &want);
        println!("  {name:<8} {e:.3e}");
        assert!(e < 1e-4, "{name} is wrong");
    }

    println!("\ntiled speedup over naive:    {:.2}x", means[0] / means[1]);
    println!("tiled as fraction of cuBLAS: {:.1}%", 100.0 * means[2] / means[1]);
}
