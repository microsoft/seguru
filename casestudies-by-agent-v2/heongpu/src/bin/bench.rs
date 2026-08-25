//! HEonGPU benchmark: SeGuRu vs hand-written CUDA vs single-core CPU.
//!
//! Every size runs a fixed 4 Mi coefficients (32 MiB) split into `batch`
//! polynomials of length `N`, so the ring size varies while the amount of work
//! stays constant. All GPU timings are kernel-only; host/device transfers
//! happen once, outside the timed loop. The CUDA reference is a mirror of the
//! SeGuRu kernels (same tiling, same butterflies), so the ratio measures code
//! generation rather than algorithm choice.

use std::time::Instant;

use heongpu_gpu::arith::{self, BLOCK_DIM};
use heongpu_gpu::cuda_ffi::{CudaKernel, CudaNtt};
use heongpu_gpu::modular::{DEFAULT_Q, Modulus};
use heongpu_gpu::ntt::{self, DeviceTables, NttTables};
use heongpu_gpu::cpu;
use gpu_host::gpu_config;

const TOTAL_COEFFS: usize = 1 << 22;
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
    n: usize,
    batch: usize,
    sg_fwd: f64,
    sg_inv: f64,
    cu_fwd: f64,
    cu_inv: f64,
    sg_add: f64,
    cu_add: f64,
    sg_mul: f64,
    cu_mul: f64,
    sg_cpm: f64,
    cu_cpm: f64,
    cpu_fwd: f64,
}

fn main() {
    let m = Modulus::new(DEFAULT_Q);
    let mut rows = Vec::new();

    for &n in &[4096usize, 8192, 16384, 32768, 65536] {
        let batch = TOTAL_COEFFS / n;
        let tables = NttTables::new(n, m);
        let data = sample(n * batch, 1234 + n as u64, m.q);
        let aux = sample(n * batch, 99 + n as u64, m.q);
        let elems = data.len();
        let egrid = arith::grid_for(elems);
        let n_mask = (n - 1) as u32;

        let (sg_fwd, sg_inv, sg_add, sg_mul, sg_cpm, sg_fwd_out, sg_add_out, sg_mul_out) =
            gpu_host::cuda_ctx(0, |ctx, md| {
                let dev = DeviceTables::upload(ctx, &tables);
                let mut a = Some(ctx.new_tensor_view(data.as_slice()).unwrap());
                let mut b = Some(ctx.new_tensor_view(vec![0u64; elems].as_slice()).unwrap());
                let d_aux = ctx.new_tensor_view(aux.as_slice()).unwrap();
                let mut d_out = ctx.new_tensor_view(vec![0u64; elems].as_slice()).unwrap();

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

                a.as_mut().unwrap().copy_from_host(data.as_slice()).unwrap();
                let d_a = a.take().unwrap();

                let add_us = time(&mut || {
                    let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                    arith::poly_add::launch(cfg, ctx, md, &d_a, &d_aux, &mut d_out, m.q).unwrap();
                });
                let mut add_out = vec![0u64; elems];
                d_out.copy_to_host(&mut add_out).unwrap();

                let mul_us = time(&mut || {
                    let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                    arith::poly_mul::launch(cfg, ctx, md, &d_a, &d_aux, &mut d_out, m.q, m.mu, m.bit)
                        .unwrap();
                });
                let mut mul_out = vec![0u64; elems];
                d_out.copy_to_host(&mut mul_out).unwrap();

                let cpm_us = time(&mut || {
                    let cfg = gpu_config!(egrid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                    arith::cipher_plain_mul::launch(
                        cfg, ctx, md, &d_a, &d_aux, &mut d_out, n_mask, m.q, m.mu, m.bit,
                    )
                    .unwrap();
                });

                (fwd_us, inv_us, add_us, mul_us, cpm_us, fwd_out, add_out, mul_out)
            });

        // CUDA reference.
        let mut cuda = CudaNtt::new(&tables, &data, &aux);
        let cu_fwd = cuda.bench(CudaKernel::Forward, WARMUP, ITERS);
        cuda.reset(&data);
        let _ = cuda.bench(CudaKernel::Forward, 0, 1);
        assert_eq!(cuda.output(), sg_fwd_out, "SeGuRu and CUDA forward NTT differ at N={n}");
        cuda.reset(&data);
        let cu_inv = cuda.bench(CudaKernel::Inverse, WARMUP, ITERS);
        cuda.reset(&data);
        let cu_add = cuda.bench(CudaKernel::PolyAdd, WARMUP, ITERS);
        assert_eq!(cuda.output(), sg_add_out, "SeGuRu and CUDA poly_add differ at N={n}");
        let cu_mul = cuda.bench(CudaKernel::PolyMul, WARMUP, ITERS);
        assert_eq!(cuda.output(), sg_mul_out, "SeGuRu and CUDA poly_mul differ at N={n}");
        let cu_cpm = cuda.bench(CudaKernel::CipherPlainMul, WARMUP, ITERS);

        // Single-core CPU forward NTT of one polynomial, and a correctness
        // check of the GPU result against it.
        let one = &data[..n];
        let t = Instant::now();
        let cpu_out = cpu::ntt_forward(one, &tables.w_fwd, m.q);
        let cpu_fwd = t.elapsed().as_secs_f64() * 1e6;
        assert_eq!(&sg_fwd_out[..n], cpu_out.as_slice(), "GPU/CPU forward NTT differ at N={n}");

        rows.push(Row {
            n,
            batch,
            sg_fwd,
            sg_inv,
            cu_fwd,
            cu_inv,
            sg_add,
            cu_add,
            sg_mul,
            cu_mul,
            sg_cpm,
            cu_cpm,
            cpu_fwd,
        });
        println!("done: N = {n}");
    }

    let mcoeff = |us: f64| TOTAL_COEFFS as f64 / us; // coefficients per microsecond
    println!(
        "\nNegacyclic NTT, {} coefficients per measurement (batch of N-coefficient polynomials)\n",
        TOTAL_COEFFS
    );
    println!(
        "| N | batch | SeGuRu fwd (us) | CUDA fwd (us) | SG/CUDA | SeGuRu inv (us) | CUDA inv (us) | SG/CUDA | Mcoeff/s fwd | CPU fwd 1 poly (us) | GPU speedup |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let gpu_per_poly = r.sg_fwd / r.batch as f64;
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {:.1} | {:.1} | {:.2}x | {:.0} | {:.0} | {:.0}x |",
            r.n,
            r.batch,
            r.sg_fwd,
            r.cu_fwd,
            r.sg_fwd / r.cu_fwd,
            r.sg_inv,
            r.cu_inv,
            r.sg_inv / r.cu_inv,
            mcoeff(r.sg_fwd),
            r.cpu_fwd,
            r.cpu_fwd / gpu_per_poly
        );
    }

    println!("\nElement-wise ciphertext operations (same 4 Mi coefficients)\n");
    println!(
        "| N | add SeGuRu (us) | add CUDA (us) | mul SeGuRu (us) | mul CUDA (us) | cipher x plain SeGuRu (us) | cipher x plain CUDA (us) | add GB/s |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let gb = 3.0 * (TOTAL_COEFFS * 8) as f64 / (r.sg_add * 1e-6) / 1e9;
        println!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.0} |",
            r.n, r.sg_add, r.cu_add, r.sg_mul, r.cu_mul, r.sg_cpm, r.cu_cpm, gb
        );
    }
}
