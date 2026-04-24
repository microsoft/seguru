use std::time::Instant;

use heongpu_gpu::addition::{
    addition_cpu, addition_kernel, multiply_elementwise_cpu, multiply_elementwise_kernel,
};
use heongpu_gpu::decryption::{sk_multiplication_cpu, sk_multiplication_kernel};
use heongpu_gpu::modular::Modulus64;
use heongpu_gpu::multiplication::{cipher_plain_mul_cpu, cipher_plain_mul_kernel};

const PRIMES: [u64; 2] = [1152921504606846883, 1152921504606846819];

/// Run a CPU closure repeatedly, returning microseconds per call.
/// Stops after `timeout_secs` seconds or 1000 iterations, whichever comes first.
fn bench_cpu_with_timeout<F: FnMut()>(mut f: F, timeout_secs: f64) -> f64 {
    let start = Instant::now();
    let mut iters = 0u64;
    loop {
        f();
        iters += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= timeout_secs || iters >= 1000 {
            return elapsed * 1_000_000.0 / iters as f64;
        }
    }
}

fn print_result(label: &str, gpu_us: f64, cpu_us: f64) {
    let speedup = cpu_us / gpu_us;
    println!(
        "  {label:<18} GPU {gpu_us:8.1} µs   CPU {cpu_us:8.1} µs   Speedup: {speedup:.1}×"
    );
}

fn main() {
    println!("HEonGPU BFV Kernel Benchmarks — SeGuRu GPU vs CPU");
    println!("==================================================");
    println!();

    let moduli: Vec<Modulus64> = PRIMES.iter().map(|&p| Modulus64::new(p)).collect();
    let mod_values: Vec<u64> = moduli.iter().map(|m| m.value).collect();
    let mod_bits: Vec<u64> = moduli.iter().map(|m| m.bit).collect();
    let mod_mus: Vec<u64> = moduli.iter().map(|m| m.mu).collect();

    for &n_power in &[12u32, 13, 14] {
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let total = ring_size * rns_count;

        println!("Ring size N={ring_size}, RNS levels={rns_count}, Total elements={total}");
        println!("{:-<70}", "");

        // Deterministic pseudo-random data via LCG
        let mut rng: u64 = 0xdeadbeef;
        let mut rand_val = |p: u64| -> u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng % p
        };

        let in1: Vec<u64> = (0..total)
            .map(|i| rand_val(PRIMES[i >> n_power]))
            .collect();
        let in2: Vec<u64> = (0..total)
            .map(|i| rand_val(PRIMES[i >> n_power]))
            .collect();
        let mut cpu_out = vec![0u64; total];
        let mut gpu_out = vec![0u64; total];

        let block = 256u32;
        let grid = (total as u32 + block - 1) / block;

        // --- Addition ---
        {
            let cpu_us = bench_cpu_with_timeout(
                || addition_cpu(&in1, &in2, &mut cpu_out, &moduli, n_power, rns_count, 1),
                10.0,
            );
            let gpu_us = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_in1 = ctx.new_tensor_view(in1.as_slice()).expect("alloc");
                let d_in2 = ctx.new_tensor_view(in2.as_slice()).expect("alloc");
                let mut d_out = ctx.new_tensor_view(gpu_out.as_mut_slice()).expect("alloc");
                let d_mv = ctx.new_tensor_view(mod_values.as_slice()).expect("alloc");

                for _ in 0..5 {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    addition_kernel::launch(
                        config, ctx, m, &d_in1, &d_in2, &mut d_out, &d_mv, n_power,
                        rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");

                let iters = 100;
                let start = Instant::now();
                for _ in 0..iters {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    addition_kernel::launch(
                        config, ctx, m, &d_in1, &d_in2, &mut d_out, &d_mv, n_power,
                        rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");
                start.elapsed().as_micros() as f64 / iters as f64
            });
            print_result("Addition:", gpu_us, cpu_us);
        }

        // --- Barrett Multiply ---
        {
            let cpu_us = bench_cpu_with_timeout(
                || {
                    multiply_elementwise_cpu(
                        &in1, &in2, &mut cpu_out, &moduli, n_power, rns_count, 1,
                    )
                },
                10.0,
            );
            let gpu_us = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_in1 = ctx.new_tensor_view(in1.as_slice()).expect("alloc");
                let d_in2 = ctx.new_tensor_view(in2.as_slice()).expect("alloc");
                let mut d_out = ctx.new_tensor_view(gpu_out.as_mut_slice()).expect("alloc");
                let d_mv = ctx.new_tensor_view(mod_values.as_slice()).expect("alloc");
                let d_mb = ctx.new_tensor_view(mod_bits.as_slice()).expect("alloc");
                let d_mm = ctx.new_tensor_view(mod_mus.as_slice()).expect("alloc");

                for _ in 0..5 {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    multiply_elementwise_kernel::launch(
                        config, ctx, m, &d_in1, &d_in2, &mut d_out, &d_mv, &d_mb, &d_mm,
                        n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");

                let iters = 100;
                let start = Instant::now();
                for _ in 0..iters {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    multiply_elementwise_kernel::launch(
                        config, ctx, m, &d_in1, &d_in2, &mut d_out, &d_mv, &d_mb, &d_mm,
                        n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");
                start.elapsed().as_micros() as f64 / iters as f64
            });
            print_result("Barrett Mul:", gpu_us, cpu_us);
        }

        // --- SK Multiplication (decrypt) ---
        {
            let cpu_us = bench_cpu_with_timeout(
                || {
                    sk_multiplication_cpu(
                        &in1, &in2, &mut cpu_out, &moduli, n_power, rns_count,
                    )
                },
                10.0,
            );
            let gpu_us = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_ct1 = ctx.new_tensor_view(in1.as_slice()).expect("alloc");
                let d_sk = ctx.new_tensor_view(in2.as_slice()).expect("alloc");
                let mut d_out = ctx.new_tensor_view(gpu_out.as_mut_slice()).expect("alloc");
                let d_mv = ctx.new_tensor_view(mod_values.as_slice()).expect("alloc");
                let d_mb = ctx.new_tensor_view(mod_bits.as_slice()).expect("alloc");
                let d_mm = ctx.new_tensor_view(mod_mus.as_slice()).expect("alloc");

                for _ in 0..5 {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    sk_multiplication_kernel::launch(
                        config, ctx, m, &d_ct1, &d_sk, &mut d_out, &d_mv, &d_mb, &d_mm,
                        n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");

                let iters = 100;
                let start = Instant::now();
                for _ in 0..iters {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    sk_multiplication_kernel::launch(
                        config, ctx, m, &d_ct1, &d_sk, &mut d_out, &d_mv, &d_mb, &d_mm,
                        n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");
                start.elapsed().as_micros() as f64 / iters as f64
            });
            print_result("SK Multiply:", gpu_us, cpu_us);
        }

        // --- Cipher-Plain Multiply ---
        {
            let cpu_us = bench_cpu_with_timeout(
                || {
                    cipher_plain_mul_cpu(
                        &in1, &in2, &mut cpu_out, &moduli, n_power, rns_count, 1,
                    )
                },
                10.0,
            );
            let gpu_us = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_cipher = ctx.new_tensor_view(in1.as_slice()).expect("alloc");
                let d_plain = ctx.new_tensor_view(in2.as_slice()).expect("alloc");
                let mut d_out = ctx.new_tensor_view(gpu_out.as_mut_slice()).expect("alloc");
                let d_mv = ctx.new_tensor_view(mod_values.as_slice()).expect("alloc");
                let d_mb = ctx.new_tensor_view(mod_bits.as_slice()).expect("alloc");
                let d_mm = ctx.new_tensor_view(mod_mus.as_slice()).expect("alloc");

                for _ in 0..5 {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    cipher_plain_mul_kernel::launch(
                        config, ctx, m, &d_cipher, &d_plain, &mut d_out, &d_mv, &d_mb,
                        &d_mm, n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");

                let iters = 100;
                let start = Instant::now();
                for _ in 0..iters {
                    let config = gpu_host::gpu_config!(grid, 1, 1, block, 1, 1, 0);
                    cipher_plain_mul_kernel::launch(
                        config, ctx, m, &d_cipher, &d_plain, &mut d_out, &d_mv, &d_mb,
                        &d_mm, n_power, rns_count as u32,
                    )
                    .expect("launch");
                }
                ctx.sync().expect("sync");
                start.elapsed().as_micros() as f64 / iters as f64
            });
            print_result("Cipher×Plain:", gpu_us, cpu_us);
        }

        println!();
    }
}
