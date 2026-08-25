//! AES-128 ECB benchmark: SeGuRu vs hand-written CUDA vs single-core CPU.
//!
//! The CUDA reference is an instruction-for-instruction mirror of the SeGuRu
//! kernel, plus a "classic" textbook CUDA kernel for context. All GPU timings
//! are kernel-only; host/device transfers happen once outside the timed loop.

use std::time::Instant;

use aes_gpu::cuda_ffi::{CudaAes, CudaKernel};
use aes_gpu::*;
use gpu_host::gpu_config;

const WARMUP: u32 = 10;
const KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

struct Row {
    label: &'static str,
    bytes: usize,
    sg_enc: f64,
    sg_dec: f64,
    cu_enc: f64,
    cu_dec: f64,
    cu_classic: f64,
    cpu_enc: Option<f64>,
}

fn gbps(bytes: usize, us: f64) -> f64 {
    bytes as f64 / (us * 1e-6) / 1e9
}

fn iters_for(bytes: usize) -> u32 {
    if bytes <= 1 << 20 {
        2000
    } else if bytes <= 1 << 24 {
        300
    } else if bytes <= 1 << 28 {
        50
    } else {
        20
    }
}

fn main() {
    let enc_rk = tables::key_expansion(&KEY);
    let dec_rk = tables::inv_round_keys(&enc_rk);
    let enc_staged = staged_round_keys(&enc_rk);
    let dec_staged = staged_round_keys(&dec_rk);
    let isb = inv_sbox_u32();

    let sizes: &[(&str, usize)] = &[
        ("16 KiB", 1 << 14),
        ("1 MiB", 1 << 20),
        ("16 MiB", 1 << 24),
        ("256 MiB", 1 << 28),
        ("1 GiB", 1 << 30),
    ];

    let mut rows = Vec::new();

    for &(label, bytes) in sizes {
        let n_blocks = bytes / 16;
        let plaintext: Vec<u8> = (0..bytes).map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8).collect();
        let h_in = bytes_to_blocks(&plaintext);
        let padded = h_in.len();
        let grid = grid_dim_for(n_blocks);
        let iters = iters_for(bytes);

        let (sg_enc, sg_dec, sg_ct) = gpu_host::cuda_ctx(0, |ctx, m| {
            let d_in = ctx.new_tensor_view::<[U32_4]>(&h_in).unwrap();
            let d_enc_rk = ctx.new_tensor_view(enc_staged.as_slice()).unwrap();
            let d_dec_rk = ctx.new_tensor_view(dec_staged.as_slice()).unwrap();
            let d_te0 = ctx.new_tensor_view(tables::TE0.as_slice()).unwrap();
            let d_td0 = ctx.new_tensor_view(tables::TD0.as_slice()).unwrap();
            let d_isb = ctx.new_tensor_view(isb.as_slice()).unwrap();
            let zeros = vec![U32_4::default(); padded];
            let mut d_ct = ctx.new_tensor_view::<[U32_4]>(&zeros).unwrap();
            let mut d_pt = ctx.new_tensor_view::<[U32_4]>(&zeros).unwrap();

            let time = |f: &mut dyn FnMut()| -> f64 {
                for _ in 0..WARMUP {
                    f();
                }
                ctx.sync().unwrap();
                let t = Instant::now();
                for _ in 0..iters {
                    f();
                }
                ctx.sync().unwrap();
                t.elapsed().as_secs_f64() * 1e6 / iters as f64
            };

            let enc_us = time(&mut || {
                let cfg = gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                aes128_encrypt::launch(cfg, ctx, m, &d_in, &mut d_ct, &d_enc_rk, &d_te0).unwrap();
            });
            let dec_us = time(&mut || {
                let cfg = gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
                aes128_decrypt::launch(cfg, ctx, m, &d_ct, &mut d_pt, &d_dec_rk, &d_td0, &d_isb)
                    .unwrap();
            });

            let mut h_ct = vec![U32_4::default(); padded];
            let mut h_pt = vec![U32_4::default(); padded];
            d_ct.copy_to_host(&mut h_ct).unwrap();
            d_pt.copy_to_host(&mut h_pt).unwrap();
            assert_eq!(
                blocks_to_bytes(&h_pt, bytes),
                plaintext,
                "SeGuRu roundtrip mismatch at {label}"
            );
            (enc_us, dec_us, h_ct)
        });

        let (cu_enc, cu_dec, cu_classic) = {
            let mut cuda = CudaAes::new(&h_in, &enc_rk, &dec_rk);
            let enc = cuda.bench(CudaKernel::EncryptOpt, WARMUP, iters);
            let cu_ct = cuda.output();
            assert_eq!(
                blocks_to_bytes(&cu_ct, bytes),
                blocks_to_bytes(&sg_ct, bytes),
                "SeGuRu and CUDA ciphertext differ at {label}"
            );
            let classic = cuda.bench(CudaKernel::EncryptClassic, WARMUP, iters);
            let cu_ct2 = cuda.output();
            assert_eq!(
                blocks_to_bytes(&cu_ct2, bytes),
                blocks_to_bytes(&sg_ct, bytes),
                "classic CUDA ciphertext differs at {label}"
            );
            let dec = cuda.bench(CudaKernel::DecryptOpt, WARMUP, iters);
            (enc, dec, classic)
        };

        // Single-core CPU AES only for the small sizes; it is ~4 orders of
        // magnitude slower and dominates the benchmark run time otherwise.
        let cpu_enc = if bytes <= (1 << 20) {
            let t = Instant::now();
            let out = cpu::encrypt_ecb(&KEY, &plaintext);
            let us = t.elapsed().as_secs_f64() * 1e6;
            assert_eq!(out, blocks_to_bytes(&sg_ct, bytes), "CPU/GPU ciphertext differ at {label}");
            Some(us)
        } else {
            None
        };

        rows.push(Row {
            label,
            bytes,
            sg_enc,
            sg_dec,
            cu_enc,
            cu_dec,
            cu_classic,
            cpu_enc,
        });
        println!("done: {label}");
    }

    println!("\nAES-128 ECB encryption (kernel time, mean over timed iterations)\n");
    println!(
        "| Size | SeGuRu (us) | CUDA mirror (us) | CUDA classic (us) | SG/CUDA | SeGuRu GB/s | CPU (us) | GPU speedup |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let cpu = r.cpu_enc.map(|c| format!("{c:.0}")).unwrap_or_else(|| "-".into());
        let sp = r
            .cpu_enc
            .map(|c| format!("{:.0}x", c / r.sg_enc))
            .unwrap_or_else(|| "-".into());
        println!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.2}x | {:.1} | {} | {} |",
            r.label,
            r.sg_enc,
            r.cu_enc,
            r.cu_classic,
            r.sg_enc / r.cu_enc,
            gbps(r.bytes, r.sg_enc),
            cpu,
            sp
        );
    }

    println!("\nAES-128 ECB decryption\n");
    println!("| Size | SeGuRu (us) | CUDA mirror (us) | SG/CUDA | SeGuRu GB/s |");
    println!("|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {:.1} | {:.1} | {:.2}x | {:.1} |",
            r.label,
            r.sg_dec,
            r.cu_dec,
            r.sg_dec / r.cu_dec,
            gbps(r.bytes, r.sg_dec)
        );
    }
}
