//! HEonGPU benchmark: SeGuRu vs hand-written CUDA vs single-core CPU.
//!
//! The size axis is the coefficient count, at a fixed ring degree. Sweeping the
//! ring degree at a fixed coefficient budget instead -- the usual FHE
//! convention -- does not scale the work: the element-wise kernels are entirely
//! ring-agnostic, and the NTT only gains butterfly stages as `log2(N)`, so the
//! whole sweep spans under 4x. Holding the ring fixed and scaling the number of
//! polynomials makes the work linear in the parameter.
//!
//! All GPU timings are kernel-only; host/device transfers happen once, outside
//! the timed loop. The CUDA reference is a mirror of the SeGuRu kernels (same
//! tiling, same butterflies), so the ratio measures code generation rather than
//! algorithm choice.

use std::time::Instant;

use heongpu_gpu::arith::{self, BLOCK_DIM};
use heongpu_gpu::cuda_ffi::{CudaKernel, CudaNtt};
use heongpu_gpu::modular::{DEFAULT_Q, Modulus};
use heongpu_gpu::ntt::{self, DeviceTables, NttTables};
use heongpu_gpu::cpu;
use gpu_host::gpu_config;

/// Coefficient counts, at 1x/10x/100x. Each is a multiple of `RING`.
const COEFFS: [usize; 3] = [1 << 20, 10 << 20, 100 << 20];
/// Ring degree held fixed across the sweep; a representative CKKS/BFV degree.
const RING: usize = 16384;
const WARMUP: u32 = 5;
const ITERS: u32 = 50;

fn sample(n: usize, seed: u64, q: u64) -> Vec<u64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % q
        })
        .collect()
}

struct Row {
    coeffs: usize,
    batch: usize,
    sg_fwd: f64,
    sg_inv: f64,
    cu_fwd: f64,
    cu_inv: f64,
    cpu_fwd: f64,
}

struct EwRow {
    coeffs: usize,
    sg_add: f64,
    cu_add: f64,
    sg_mul: f64,
    cu_mul: f64,
    sg_cpm: f64,
    cu_cpm: f64,
}

fn main() {
    let m = Modulus::new(DEFAULT_Q);
    let mut rows = Vec::new();

    let n = RING;
    let tables = NttTables::new(n, m);
    for &elems in &COEFFS {
        let batch = elems / n;
        let data = sample(elems, 1234 + elems as u64, m.q);
        let aux = sample(elems, 99 + elems as u64, m.q);

        let (sg_fwd, sg_inv, sg_fwd_out) = gpu_host::cuda_ctx(0, |ctx, md| {
                let dev = DeviceTables::upload(ctx, &tables);
                let mut a = Some(ctx.new_tensor_view(data.as_slice()).unwrap());
                let mut b = Some(ctx.new_tensor_view(vec![0u64; elems].as_slice()).unwrap());

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

                let fwd_us = time(&mut || {
                    let (r, s) =
                        ntt::launch_forward(ctx, md, &tables, &dev, a.take().unwrap(), b.take().unwrap(), batch);
                    a = Some(r);
                    b = Some(s);
                });
                let fwd_out = {
                    // Re-run once from the pristine input so the checked output
                    // is a forward transform of `data`.
                    a.as_mut().unwrap().copy_from_host(data.as_slice()).unwrap();
                    let (r, s) =
                        ntt::launch_forward(ctx, md, &tables, &dev, a.take().unwrap(), b.take().unwrap(), batch);
                    a = Some(r);
                    b = Some(s);
                    let mut h = vec![0u64; elems];
                    a.as_ref().unwrap().copy_to_host(&mut h).unwrap();
                    h
                };

                let inv_us = time(&mut || {
                    let (r, s) =
                        ntt::launch_inverse(ctx, md, &tables, &dev, a.take().unwrap(), b.take().unwrap(), batch);
                    a = Some(r);
                    b = Some(s);
                });

                (fwd_us, inv_us, fwd_out)
        });

        // CUDA reference.
        let mut cuda = CudaNtt::new(&tables, &data, &aux);
        let cu_fwd = cuda.bench(CudaKernel::Forward, WARMUP, ITERS);
        cuda.reset(&data);
        let _ = cuda.bench(CudaKernel::Forward, 0, 1);
        assert_eq!(cuda.output(), sg_fwd_out, "SeGuRu and CUDA forward NTT differ at {elems} coeffs");
        cuda.reset(&data);
        let cu_inv = cuda.bench(CudaKernel::Inverse, WARMUP, ITERS);

        // Single-core CPU forward NTT of one polynomial, and a correctness
        // check of the GPU result against it.
        let one = &data[..n];
        let t = Instant::now();
        let cpu_out = cpu::ntt_forward(one, &tables.w_fwd, m.q);
        let cpu_fwd = t.elapsed().as_secs_f64() * 1e6;
        assert_eq!(&sg_fwd_out[..n], cpu_out.as_slice(), "GPU/CPU forward NTT differ at {elems} coeffs");

        rows.push(Row {
            coeffs: elems,
            batch,
            sg_fwd,
            sg_inv,
            cu_fwd,
            cu_inv,
            cpu_fwd,
        });
        println!("done: {elems} coefficients");
    }

    let mcoeff = |us: f64, elems: usize| elems as f64 / us; // coefficients per microsecond
    println!("\nNegacyclic NTT, ring degree {RING}, swept over coefficient count\n");
    println!(
        "| N | batch | SeGuRu fwd (us) | CUDA fwd (us) | SG/CUDA | SeGuRu inv (us) | CUDA inv (us) | SG/CUDA | Mcoeff/s fwd | CPU fwd 1 poly (us) | GPU speedup |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let gpu_per_poly = r.sg_fwd / r.batch as f64;
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {:.1} | {:.1} | {:.2}x | {:.0} | {:.0} | {:.0}x |",
            RING,
            r.batch,
            r.sg_fwd,
            r.cu_fwd,
            r.sg_fwd / r.cu_fwd,
            r.sg_inv,
            r.cu_inv,
            r.sg_inv / r.cu_inv,
            mcoeff(r.sg_fwd, r.coeffs),
            r.cpu_fwd,
            r.cpu_fwd / gpu_per_poly
        );

        let param = format!("coeffs={}", r.coeffs);
        csv_row("heongpu", "ntt_forward", &param, "seguru", "time", r.sg_fwd, "us");
        csv_row("heongpu", "ntt_forward", &param, "cuda", "time", r.cu_fwd, "us");
        csv_row("heongpu", "ntt_forward", &param, "seguru", "throughput", mcoeff(r.sg_fwd, r.coeffs), "Mcoeff/s");
        csv_row("heongpu", "ntt_forward", &param, "cpu", "time", r.cpu_fwd, "us");
        csv_row("heongpu", "ntt_inverse", &param, "seguru", "time", r.sg_inv, "us");
        csv_row("heongpu", "ntt_inverse", &param, "cuda", "time", r.cu_inv, "us");
    }

    // Element-wise sweep. These kernels are ring-agnostic, so the size axis is
    // the coefficient count rather than the ring degree.
    let tables = NttTables::new(RING, m);
    let n_mask = (RING - 1) as u32;
    let mut ew_rows = Vec::new();
    for &coeffs in &COEFFS {
        let data = sample(coeffs, 1234 + coeffs as u64, m.q);
        let aux = sample(coeffs, 99 + coeffs as u64, m.q);
        let egrid = arith::grid_for(coeffs);

        let (sg_add, sg_mul, sg_cpm, sg_add_out, sg_mul_out) = gpu_host::cuda_ctx(0, |ctx, md| {
            let d_a = ctx.new_tensor_view(data.as_slice()).unwrap();
            let d_aux = ctx.new_tensor_view(aux.as_slice()).unwrap();
            let mut d_out = ctx.new_tensor_view(vec![0u64; coeffs].as_slice()).unwrap();

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

            let add_us = time(&mut || {
                let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                arith::poly_add::launch(cfg, ctx, md, &d_a, &d_aux, &mut d_out, m.q).unwrap();
            });
            let mut add_out = vec![0u64; coeffs];
            d_out.copy_to_host(&mut add_out).unwrap();

            let mul_us = time(&mut || {
                let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                arith::poly_mul::launch(cfg, ctx, md, &d_a, &d_aux, &mut d_out, m.q, m.mu, m.bit)
                    .unwrap();
            });
            let mut mul_out = vec![0u64; coeffs];
            d_out.copy_to_host(&mut mul_out).unwrap();

            let cpm_us = time(&mut || {
                let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                arith::cipher_plain_mul::launch(
                    cfg, ctx, md, &d_a, &d_aux, &mut d_out, n_mask, m.q, m.mu, m.bit,
                )
                .unwrap();
            });

            (add_us, mul_us, cpm_us, add_out, mul_out)
        });

        let mut cuda = CudaNtt::new(&tables, &data, &aux);
        let cu_add = cuda.bench(CudaKernel::PolyAdd, WARMUP, ITERS);
        assert_eq!(cuda.output(), sg_add_out, "SeGuRu and CUDA poly_add differ at {coeffs} coeffs");
        let cu_mul = cuda.bench(CudaKernel::PolyMul, WARMUP, ITERS);
        assert_eq!(cuda.output(), sg_mul_out, "SeGuRu and CUDA poly_mul differ at {coeffs} coeffs");
        let cu_cpm = cuda.bench(CudaKernel::CipherPlainMul, WARMUP, ITERS);

        ew_rows.push(EwRow { coeffs, sg_add, cu_add, sg_mul, cu_mul, sg_cpm, cu_cpm });
        println!("done: {coeffs} coefficients");
    }

    println!("\nElement-wise ciphertext operations\n");
    println!(
        "| coefficients | add SeGuRu (us) | add CUDA (us) | mul SeGuRu (us) | mul CUDA (us) | cipher x plain SeGuRu (us) | cipher x plain CUDA (us) | add GB/s |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for r in &ew_rows {
        let gb = 3.0 * (r.coeffs * 8) as f64 / (r.sg_add * 1e-6) / 1e9;
        println!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} |",
            r.coeffs, r.sg_add, r.cu_add, r.sg_mul, r.cu_mul, r.sg_cpm, r.cu_cpm, gb
        );

        let param = format!("coeffs={}", r.coeffs);
        csv_row("heongpu", "poly_add", &param, "seguru", "time", r.sg_add, "us");
        csv_row("heongpu", "poly_add", &param, "cuda", "time", r.cu_add, "us");
        csv_row("heongpu", "poly_add", &param, "seguru", "throughput", gb, "GB/s");
        csv_row("heongpu", "poly_mul", &param, "seguru", "time", r.sg_mul, "us");
        csv_row("heongpu", "poly_mul", &param, "cuda", "time", r.cu_mul, "us");
        csv_row("heongpu", "cipher_plain_mul", &param, "seguru", "time", r.sg_cpm, "us");
        csv_row("heongpu", "cipher_plain_mul", &param, "cuda", "time", r.cu_cpm, "us");
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
