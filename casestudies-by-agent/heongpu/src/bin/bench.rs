use std::time::Instant;

use heongpu_gpu::addition::{
    addition_cpu, addition_kernel, multiply_elementwise_cpu, multiply_elementwise_kernel,
};
use heongpu_gpu::cuda_ffi;
use heongpu_gpu::decryption::{sk_multiplication_cpu, sk_multiplication_kernel};
use heongpu_gpu::modular::Modulus64;
use heongpu_gpu::multiplication::{cipher_plain_mul_cpu, cipher_plain_mul_kernel};

const PRIMES: [u64; 2] = [1152921504606846883, 1152921504606846819];

/// RAII wrapper for CUDA device memory (separate from SeGuRu's allocator).
struct CudaVec {
    ptr: *mut u8,
    _size: usize,
}

impl CudaVec {
    fn from_slice(data: &[u64]) -> Self {
        let size = data.len() * std::mem::size_of::<u64>();
        let mut ptr: *mut u8 = std::ptr::null_mut();
        unsafe {
            cuda_ffi::cuda_malloc(&mut ptr, size);
            cuda_ffi::cuda_memcpy_h2d(ptr, data.as_ptr() as *const u8, size);
        }
        Self { ptr, _size: size }
    }

    fn zeroed(len: usize) -> Self {
        let size = len * std::mem::size_of::<u64>();
        let mut ptr: *mut u8 = std::ptr::null_mut();
        unsafe {
            cuda_ffi::cuda_malloc(&mut ptr, size);
        }
        Self { ptr, _size: size }
    }

    fn as_ptr(&self) -> *const u64 {
        self.ptr as *const u64
    }
    fn as_mut_ptr(&self) -> *mut u64 {
        self.ptr as *mut u64
    }
}

impl Drop for CudaVec {
    fn drop(&mut self) {
        unsafe {
            cuda_ffi::cuda_free(self.ptr);
        }
    }
}

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

fn print_header() {
    println!(
        "  {:<18} {:>10} {:>10} {:>8} {:>10} {:>10}",
        "Operation", "SeGuRu", "CUDA", "Ratio", "CPU", "GPU Spdup"
    );
    println!("  {:-<68}", "");
}

fn print_result(label: &str, seguru_us: f64, cuda_us: f64, cpu_us: f64) {
    let ratio = seguru_us / cuda_us;
    let speedup = cpu_us / seguru_us;
    println!(
        "  {:<18} {:>8.1} µs {:>8.1} µs {:>6.2}× {:>8.1} µs {:>8.1}×",
        label, seguru_us, cuda_us, ratio, cpu_us, speedup,
    );
}

fn main() {
    println!("HEonGPU BFV Kernel Benchmarks — SeGuRu vs CUDA vs CPU");
    println!("======================================================");
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
        println!("{:-<74}", "");
        print_header();

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

        // Allocate CUDA device memory once for this ring size
        let d_in1_cuda = CudaVec::from_slice(&in1);
        let d_in2_cuda = CudaVec::from_slice(&in2);
        let d_out_cuda = CudaVec::zeroed(total);
        let d_mod_vals_cuda = CudaVec::from_slice(&mod_values);
        let d_mod_bits_cuda = CudaVec::from_slice(&mod_bits);
        let d_mod_mus_cuda = CudaVec::from_slice(&mod_mus);

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
            let cuda_us = unsafe {
                cuda_ffi::cuda_bench_addition(
                    d_in1_cuda.as_ptr(),
                    d_in2_cuda.as_ptr(),
                    d_out_cuda.as_mut_ptr(),
                    d_mod_vals_cuda.as_ptr(),
                    total as u32,
                    n_power,
                    rns_count as u32,
                    256,
                    100,
                )
            };
            print_result("Addition:", gpu_us, cuda_us, cpu_us);
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
            let cuda_us = unsafe {
                cuda_ffi::cuda_bench_multiply(
                    d_in1_cuda.as_ptr(),
                    d_in2_cuda.as_ptr(),
                    d_out_cuda.as_mut_ptr(),
                    d_mod_vals_cuda.as_ptr(),
                    d_mod_bits_cuda.as_ptr(),
                    d_mod_mus_cuda.as_ptr(),
                    total as u32,
                    n_power,
                    rns_count as u32,
                    256,
                    100,
                )
            };
            print_result("Barrett Mul:", gpu_us, cuda_us, cpu_us);
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
            let cuda_us = unsafe {
                cuda_ffi::cuda_bench_sk_multiply(
                    d_in1_cuda.as_ptr(),
                    d_in2_cuda.as_ptr(),
                    d_out_cuda.as_mut_ptr(),
                    d_mod_vals_cuda.as_ptr(),
                    d_mod_bits_cuda.as_ptr(),
                    d_mod_mus_cuda.as_ptr(),
                    total as u32,
                    n_power,
                    rns_count as u32,
                    256,
                    100,
                )
            };
            print_result("SK Multiply:", gpu_us, cuda_us, cpu_us);
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
            let cuda_us = unsafe {
                cuda_ffi::cuda_bench_cipher_plain_mul(
                    d_in1_cuda.as_ptr(),
                    d_in2_cuda.as_ptr(),
                    d_out_cuda.as_mut_ptr(),
                    d_mod_vals_cuda.as_ptr(),
                    d_mod_bits_cuda.as_ptr(),
                    d_mod_mus_cuda.as_ptr(),
                    total as u32,
                    n_power,
                    rns_count as u32,
                    256,
                    100,
                )
            };
            print_result("Cipher×Plain:", gpu_us, cuda_us, cpu_us);
        }

        println!();
    }
}
