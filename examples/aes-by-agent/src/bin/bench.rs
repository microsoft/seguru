//! AES-128 ECB GPU benchmark: SeGuRu vs CUDA C++
//! Run with: cargo run --bin bench --features bench --release -p aes-gpu

#[path = "../cuda_ffi.rs"]
#[allow(dead_code)]
mod cuda_ffi;

use aes_gpu::aes_common;
use aes_gpu::*;
use gpu_host::cuda_ctx;
use std::time::Instant;

const WARMUP: i32 = 3;
const ITERS: i32 = 100;

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

struct BenchResult {
    kernel: String,
    size_label: String,
    num_blocks: u32,
    seguru_us: f64,
    cuda_us: f64,
}

impl BenchResult {
    fn ratio(&self) -> f64 {
        self.seguru_us / self.cuda_us
    }
    fn throughput_gbps(&self, us: f64) -> f64 {
        let bytes = self.num_blocks as f64 * 16.0;
        bytes / us / 1e3 // bytes / microseconds = MB/s, /1e3 = GB/s
    }
}

fn main() {
    let key: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
        0x4f, 0x3c,
    ];
    let enc_keys = aes_common::key_expansion(&key);
    let dec_keys = aes_common::inv_round_keys(&enc_keys);
    let te_tables = build_te_tables();
    let td_tables = build_td_tables();
    let inv_sbox_packed: Vec<u32> = aes_common::INV_SBOX
        .chunks(4)
        .map(|c| ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) | (c[3] as u32))
        .collect();

    let sizes: Vec<(&str, u32)> = vec![
        ("1K", 1024),
        ("4K", 4096),
        ("16K", 16384),
        ("64K", 65536),
        ("256K", 262144),
        ("1M", 1048576),
        ("4M", 4194304),
        ("16M", 16777216),
        ("64M", 67108864), // 1 GB of data (64M blocks × 16 bytes)
    ];

    let mut results: Vec<BenchResult> = Vec::new();

    // Print CSV header
    println!("kernel,size,num_blocks,seguru_us,cuda_us,ratio,seguru_gbps,cuda_gbps");

    cuda_ctx(0, |ctx, m| {
        for &(label, num_blocks) in &sizes {
            let block_size: u32 = 256;
            let grid_size: u32 = (num_blocks + block_size - 1) / block_size;
            let shared_bytes: u32 = 1024 * 4;
            let n_words = (num_blocks * 4) as usize;

            // Generate test data
            let plaintext: Vec<u8> = (0..num_blocks * 16).map(|i| (i % 256) as u8).collect();
            let input_u32 = bytes_to_u32_be(&plaintext);

            // ── SeGuRu T-table encrypt ──
            let d_input = ctx.new_tensor_view(input_u32.as_slice()).unwrap();
            let d_rk = ctx.new_tensor_view(enc_keys.as_slice()).unwrap();
            let d_te = ctx.new_tensor_view(te_tables.as_slice()).unwrap();
            let mut d_output = ctx.new_tensor_view(&vec![0u32; n_words] as &[u32]).unwrap();

            // Warmup
            for _ in 0..WARMUP {
                let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_bytes);
                aes128_encrypt_ttable_kernel::launch(
                    config, ctx, m, &d_input, &mut d_output, &d_rk, &d_te, num_blocks,
                ).unwrap();
                ctx.sync().unwrap();
            }
            // Timed runs
            let mut times = Vec::new();
            for _ in 0..ITERS {
                let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_bytes);
                let start = Instant::now();
                aes128_encrypt_ttable_kernel::launch(
                    config, ctx, m, &d_input, &mut d_output, &d_rk, &d_te, num_blocks,
                ).unwrap();
                ctx.sync().unwrap();
                times.push(start.elapsed().as_nanos() as f64 / 1000.0);
            }
            let seguru_enc_us = median(&mut times);

            // CUDA T-table encrypt
            let mut cuda_output = vec![0u8; (num_blocks * 16) as usize];
            let cuda_enc_us = unsafe {
                cuda_ffi::bench_aes128_encrypt_ttable(
                    plaintext.as_ptr(),
                    cuda_output.as_mut_ptr(),
                    enc_keys.as_ptr(),
                    num_blocks,
                    WARMUP,
                    ITERS,
                ) as f64
            };

            results.push(BenchResult {
                kernel: "encrypt_ttable".into(),
                size_label: label.into(),
                num_blocks,
                seguru_us: seguru_enc_us,
                cuda_us: cuda_enc_us,
            });

            // ── SeGuRu T-table decrypt ──
            // Use the encrypted output as input for decrypt
            let mut encrypted_u32 = vec![0u32; n_words];
            d_output.copy_to_host(&mut encrypted_u32).unwrap();

            let d_enc_input = ctx.new_tensor_view(encrypted_u32.as_slice()).unwrap();
            let d_dec_rk = ctx.new_tensor_view(dec_keys.as_slice()).unwrap();
            let d_td = ctx.new_tensor_view(td_tables.as_slice()).unwrap();
            let d_inv_sbox = ctx.new_tensor_view(inv_sbox_packed.as_slice()).unwrap();
            let mut d_dec_output = ctx.new_tensor_view(&vec![0u32; n_words] as &[u32]).unwrap();

            // Warmup
            for _ in 0..WARMUP {
                let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_bytes);
                aes128_decrypt_ttable_kernel::launch(
                    config, ctx, m, &d_enc_input, &mut d_dec_output,
                    &d_dec_rk, &d_td, &d_inv_sbox, num_blocks,
                ).unwrap();
                ctx.sync().unwrap();
            }
            let mut times = Vec::new();
            for _ in 0..ITERS {
                let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_bytes);
                let start = Instant::now();
                aes128_decrypt_ttable_kernel::launch(
                    config, ctx, m, &d_enc_input, &mut d_dec_output,
                    &d_dec_rk, &d_td, &d_inv_sbox, num_blocks,
                ).unwrap();
                ctx.sync().unwrap();
                times.push(start.elapsed().as_nanos() as f64 / 1000.0);
            }
            let seguru_dec_us = median(&mut times);

            // CUDA T-table decrypt
            let encrypted_bytes = u32_be_to_bytes(&encrypted_u32);
            let mut cuda_dec_output = vec![0u8; (num_blocks * 16) as usize];
            let cuda_dec_us = unsafe {
                cuda_ffi::bench_aes128_decrypt_ttable(
                    encrypted_bytes.as_ptr(),
                    cuda_dec_output.as_mut_ptr(),
                    dec_keys.as_ptr(),
                    num_blocks,
                    WARMUP,
                    ITERS,
                ) as f64
            };

            results.push(BenchResult {
                kernel: "decrypt_ttable".into(),
                size_label: label.into(),
                num_blocks,
                seguru_us: seguru_dec_us,
                cuda_us: cuda_dec_us,
            });
        }
    });

    // Print CSV results
    for r in &results {
        println!(
            "{},{},{},{:.2},{:.2},{:.4},{:.2},{:.2}",
            r.kernel,
            r.size_label,
            r.num_blocks,
            r.seguru_us,
            r.cuda_us,
            r.ratio(),
            r.throughput_gbps(r.seguru_us),
            r.throughput_gbps(r.cuda_us),
        );
    }

    // Print summary table to stderr
    eprintln!("\n{:-<90}", "");
    eprintln!(
        "{:<20} {:>8} {:>12} {:>12} {:>8} {:>12} {:>12}",
        "Kernel", "Blocks", "SeGuRu(µs)", "CUDA(µs)", "Ratio", "SeGuRu GB/s", "CUDA GB/s"
    );
    eprintln!("{:-<90}", "");
    for r in &results {
        eprintln!(
            "{:<20} {:>8} {:>12.2} {:>12.2} {:>8.4} {:>12.2} {:>12.2}",
            r.kernel,
            r.num_blocks,
            r.seguru_us,
            r.cuda_us,
            r.ratio(),
            r.throughput_gbps(r.seguru_us),
            r.throughput_gbps(r.cuda_us),
        );
    }
    eprintln!("{:-<90}", "");

    // Average ratio
    let avg_ratio: f64 = results.iter().map(|r| r.ratio()).sum::<f64>() / results.len() as f64;
    eprintln!("Average SeGuRu/CUDA ratio: {:.4}×", avg_ratio);
}
