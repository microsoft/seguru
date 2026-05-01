use std::time::Instant;

use gpusorting_by_agent::{
    BIN_PART_SIZE, DOWNSWEEP_THREADS, PART_SIZE, RADIX, RADIX_LOG, RADIX_PASSES, SCAN_THREADS,
    UPSWEEP_THREADS,
};

#[cfg(feature = "bench")]
mod ffi {
    extern "C" {
        pub fn cuda_bench_sort(
            h_keys: *const u32,
            h_out: *mut u32,
            size: u32,
            warmup: u32,
            iters: u32,
        ) -> f32;
    }
}

fn seguru_sort(data: &[u32], iters: usize, warmup: usize) -> (f64, Vec<u32>) {
    let mut result = vec![0u32; data.len()];
    let us = gpu_host::cuda_ctx(0, |ctx, m| {
        let size = data.len() as u32;
        let thread_blocks = (size + PART_SIZE - 1) / PART_SIZE;
        let padded_thread_blocks =
            ((thread_blocks + SCAN_THREADS - 1) / SCAN_THREADS) * SCAN_THREADS;
        let global_hist_len = (RADIX * RADIX_PASSES) as usize;
        let pass_hist_len = (RADIX * padded_thread_blocks) as usize;

        let h_zero_gh = vec![0u32; global_hist_len];
        let h_zero_ph = vec![0u32; pass_hist_len];

        // Reinterpret data as U32_4 slices for type-safe vectorized GPU loads.
        // U32_4 guarantees correct alignment (16 bytes) and size (4 × u32).
        assert!(data.len() % 4 == 0, "data length must be multiple of 4 for U32_4");
        let sort_u32_4: &[gpu::U32_4] = unsafe {
            core::slice::from_raw_parts(
                data.as_ptr() as *const gpu::U32_4,
                data.len() / 4,
            )
        };
        let alt_u32_4 = vec![gpu::U32_4::default(); data.len() / 4];

        // Allocate device buffers ONCE — reuse across iterations
        let mut d_sort = ctx.new_tensor_view::<[gpu::U32_4]>(sort_u32_4).unwrap();
        let mut d_alt = ctx.new_tensor_view::<[gpu::U32_4]>(&alt_u32_4).unwrap();
        let mut d_global_hist = ctx.new_tensor_view::<[u32]>(&h_zero_gh).unwrap();
        let mut d_ph = ctx.new_tensor_view::<[u32]>(&h_zero_ph).unwrap();

        // Helper: run one sort using pre-allocated buffers
        let mut run_sort = |ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
                            m: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>| {
            // Reset input data and zero histograms (no reallocation)
            d_sort.copy_from_host(sort_u32_4).unwrap();
            d_global_hist.memset(0).unwrap();

            for pass in 0..RADIX_PASSES {
                let radix_shift = pass * RADIX_LOG;
                // Zero pass histogram each pass
                d_ph.memset(0).unwrap();

                let us_cfg = gpu_host::gpu_config!(
                    thread_blocks, 1, 1,
                    UPSWEEP_THREADS, 1, 1,
                    RADIX * 2 * 4
                );
                let sc_cfg = gpu_host::gpu_config!(
                    RADIX, 1, 1,
                    SCAN_THREADS, 1, 1,
                    SCAN_THREADS * 4
                );
                let ds_cfg = gpu_host::gpu_config!(
                    thread_blocks, 1, 1,
                    DOWNSWEEP_THREADS, 1, 1,
                    (BIN_PART_SIZE + RADIX) * 4
                );

                if pass % 2 == 0 {
                    gpusorting_by_agent::upsweep::radix_upsweep::launch(
                        us_cfg, ctx, m, &d_sort, &mut d_global_hist, &mut d_ph,
                        size, radix_shift, padded_thread_blocks,
                    ).unwrap();
                } else {
                    gpusorting_by_agent::upsweep::radix_upsweep::launch(
                        us_cfg, ctx, m, &d_alt, &mut d_global_hist, &mut d_ph,
                        size, radix_shift, padded_thread_blocks,
                    ).unwrap();
                }
                gpusorting_by_agent::scan::radix_scan::launch(
                    sc_cfg, ctx, m, &mut d_ph, padded_thread_blocks,
                ).unwrap();
                if pass % 2 == 0 {
                    gpusorting_by_agent::downsweep::radix_downsweep::launch(
                        ds_cfg, ctx, m, &d_sort, &mut d_alt.flatten(),
                        &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks,
                    ).unwrap();
                } else {
                    gpusorting_by_agent::downsweep::radix_downsweep::launch(
                        ds_cfg, ctx, m, &d_alt, &mut d_sort.flatten(),
                        &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks,
                    ).unwrap();
                }
            }
        };

        // Warmup
        for _ in 0..warmup {
            run_sort(ctx, m);
        }
        ctx.sync().unwrap();

        // Timed runs
        let start = Instant::now();
        for _ in 0..iters {
            run_sort(ctx, m);
        }
        ctx.sync().unwrap();
        let elapsed = start.elapsed();

        let mut result_u32_4 = vec![gpu::U32_4::default(); data.len() / 4];
        d_sort.copy_to_host(&mut result_u32_4).unwrap();
        // Reinterpret back to u32
        unsafe {
            core::ptr::copy_nonoverlapping(
                result_u32_4.as_ptr() as *const u32,
                result.as_mut_ptr(),
                data.len(),
            );
        }
        elapsed.as_micros() as f64 / iters as f64
    });
    (us, result)
}

fn main() {
    println!("GPU Radix Sort Benchmark — SeGuRu vs CUDA");
    println!("==========================================\n");

    let warmup = 5;
    let iters = 50;

    println!(
        "{:<12} {:>12} {:>12} {:>10} {:>14}",
        "Size", "SeGuRu (µs)", "CUDA (µs)", "SG/CUDA", "M keys/sec"
    );
    println!("{:-<64}", "");

    for &log_n in &[16u32, 18, 20, 22, 24] {
        let n = 1u32 << log_n;
        let data: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2654435761u32)).collect();

        // SeGuRu
        let (seguru_us, seguru_result) = seguru_sort(&data, iters, warmup);

        // CUDA reference
        #[cfg(feature = "bench")]
        let cuda_ms = unsafe {
            let mut cuda_out = vec![0u32; data.len()];
            ffi::cuda_bench_sort(
                data.as_ptr(),
                cuda_out.as_mut_ptr(),
                n,
                warmup as u32,
                iters as u32,
            )
        };
        #[cfg(feature = "bench")]
        let cuda_us = (cuda_ms as f64) * 1000.0;
        #[cfg(not(feature = "bench"))]
        let cuda_us = f64::NAN;

        let ratio = seguru_us / cuda_us;
        let mkeys = (n as f64) / seguru_us; // M keys/sec

        println!(
            "{:<12} {:>10.1} {:>10.1} {:>10.2}× {:>12.1}",
            format!("2^{log_n} ({n})"),
            seguru_us,
            cuda_us,
            ratio,
            mkeys,
        );

        // Verify correctness
        let mut expected = data.clone();
        expected.sort();
        assert_eq!(seguru_result, expected, "SeGuRu sort incorrect at size {n}");
    }
}
