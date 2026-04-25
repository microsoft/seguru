#[cfg(test)]
mod sort_tests {
    use gpu_host::cuda_ctx;

    use crate::{
        DOWNSWEEP_THREADS, PART_SIZE, RADIX, RADIX_LOG, RADIX_PASSES, SCAN_THREADS,
        UPSWEEP_THREADS, BIN_PART_SIZE,
    };

    // ============================================================================
    // Orchestrator: dispatch_radix_sort (host-side)
    //
    // CUDA reference: DeviceRadixSortDispatcher.cuh
    //   cudaMemset(globalHistogram, 0, sizeof(uint32_t) * RADIX * RADIX_PASSES);
    //   for (uint32_t radixShift = 0; radixShift < 32; radixShift += RADIX_LOG) {
    //       Upsweep<<<threadBlocks, 128, ...>>>(sort, globalHist, passHist, size, radixShift);
    //       Scan<<<RADIX, 128, ...>>>(passHist, threadBlocks);
    //       DownsweepKeysOnly<<<threadBlocks, 512, ...>>>(sort, alt, globalHist, passHist, ...);
    //       swap(sort, alt);
    //   }
    // ============================================================================

    fn run_sort(input: &mut [u32]) {
        cuda_ctx(0, |ctx, m| {
            let size = input.len() as u32;
            let thread_blocks = (size + PART_SIZE - 1) / PART_SIZE;
            // Pad thread_blocks to multiple of SCAN_THREADS for chunk_mut alignment
            let padded_thread_blocks =
                ((thread_blocks + SCAN_THREADS - 1) / SCAN_THREADS) * SCAN_THREADS;
            let global_hist_len = (RADIX * RADIX_PASSES) as usize;
            let pass_hist_len = (RADIX * padded_thread_blocks) as usize;

            let mut h_global_hist = vec![0u32; global_hist_len];
            let mut h_pass_hist = vec![0u32; pass_hist_len];
            let mut h_alt = vec![0u32; input.len()];

            let mut d_data = ctx.new_tensor_view::<[u32]>(input).expect("alloc data");
            let mut d_alt = ctx.new_tensor_view::<[u32]>(&h_alt).expect("alloc alt");
            let mut d_global_hist =
                ctx.new_tensor_view::<[u32]>(&h_global_hist).expect("alloc global_hist");
            let mut _d_pass_hist =
                ctx.new_tensor_view::<[u32]>(&h_pass_hist).expect("alloc pass_hist");

            for pass in 0..RADIX_PASSES {
                let radix_shift = pass * RADIX_LOG;

                // Zero pass histogram each pass
                h_pass_hist.iter_mut().for_each(|v| *v = 0);
                let mut d_pass_hist_fresh =
                    ctx.new_tensor_view::<[u32]>(&h_pass_hist).expect("alloc pass_hist");

                // 1. Upsweep
                let upsweep_smem = RADIX * 2 * 4; // 2048 bytes
                let upsweep_config =
                    gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
                if pass % 2 == 0 {
                    crate::upsweep::radix_upsweep::launch(
                        upsweep_config, ctx, m,
                        &d_data, &mut d_global_hist, &mut d_pass_hist_fresh,
                        size, radix_shift, padded_thread_blocks,
                    ).expect("upsweep launch failed");
                } else {
                    crate::upsweep::radix_upsweep::launch(
                        upsweep_config, ctx, m,
                        &d_alt, &mut d_global_hist, &mut d_pass_hist_fresh,
                        size, radix_shift, padded_thread_blocks,
                    ).expect("upsweep launch failed");
                }

                // 2. Scan
                let scan_smem = SCAN_THREADS * 4; // 512 bytes
                let scan_config =
                    gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
                crate::scan::radix_scan::launch(
                    scan_config, ctx, m, &mut d_pass_hist_fresh, padded_thread_blocks,
                ).expect("scan launch failed");

                // 3. Downsweep
                let downsweep_smem = (BIN_PART_SIZE + RADIX) * 4;
                let downsweep_config =
                    gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
                if pass % 2 == 0 {
                    crate::downsweep::radix_downsweep::launch(
                        downsweep_config, ctx, m,
                        &d_data, &mut d_alt, &d_global_hist, &d_pass_hist_fresh,
                        size, radix_shift, padded_thread_blocks,
                    ).expect("downsweep launch failed");
                } else {
                    crate::downsweep::radix_downsweep::launch(
                        downsweep_config, ctx, m,
                        &d_alt, &mut d_data, &d_global_hist, &d_pass_hist_fresh,
                        size, radix_shift, padded_thread_blocks,
                    ).expect("downsweep launch failed");
                }

                _d_pass_hist = d_pass_hist_fresh;
            }

            // After 4 passes (even), result is in d_data
            d_data.copy_to_host(input).expect("copy back failed");
        });
    }

    #[test]
    fn test_sort_small_sequential() {
        let mut data: Vec<u32> = (0..64).rev().collect();
        let mut expected = data.clone();
        expected.sort();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_small_random() {
        let mut data: Vec<u32> = vec![
            42, 17, 93, 5, 67, 31, 88, 12, 55, 73, 1, 99, 23, 45, 8, 76, 34, 61, 0, 50, 28, 85,
            14, 69, 3, 92, 37, 58, 81, 19, 44, 100,
        ];
        let mut expected = data.clone();
        expected.sort();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut data: Vec<u32> = (0..128).collect();
        let expected = data.clone();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_all_same() {
        let mut data = vec![42u32; 256];
        let expected = data.clone();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_large_random() {
        let n = 8192;
        let mut data: Vec<u32> = (0..n as u32).map(|i: u32| i.wrapping_mul(2654435761u32) & 0xFFFF).collect();
        let mut expected = data.clone();
        expected.sort();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_powers_of_two() {
        let mut data: Vec<u32> = (0..32).map(|i| 1u32 << (i % 32)).collect();
        let mut expected = data.clone();
        expected.sort();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_sort_multi_block() {
        // 16384 > PART_SIZE(7680), exercises multi-block upsweep/downsweep
        let n = 16384u32;
        let mut data: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2654435761u32)).collect();
        let mut expected = data.clone();
        expected.sort();
        run_sort(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn bench_sort_1m() {
        let n = 1 << 20; // ~1M elements
        let data: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761u32)).collect();

        cuda_ctx(0, |ctx, m| {
            let size = data.len() as u32;
            let thread_blocks = (size + PART_SIZE - 1) / PART_SIZE;
            let padded_thread_blocks =
                ((thread_blocks + SCAN_THREADS - 1) / SCAN_THREADS) * SCAN_THREADS;
            let global_hist_len = (RADIX * RADIX_PASSES) as usize;
            let pass_hist_len = (RADIX * padded_thread_blocks) as usize;

            // Pre-allocate host zeroed buffers
            let h_zero_gh = vec![0u32; global_hist_len];
            let h_zero_ph = vec![0u32; pass_hist_len];
            let h_alt = vec![0u32; data.len()];

            // Warmup run
            {
                let mut d_sort = ctx.new_tensor_view::<[u32]>(&data).unwrap();
                let mut d_alt = ctx.new_tensor_view::<[u32]>(&h_alt).unwrap();
                let mut d_global_hist = ctx.new_tensor_view::<[u32]>(&h_zero_gh).unwrap();
                for pass in 0..RADIX_PASSES {
                    let radix_shift = pass * RADIX_LOG;
                    let mut d_ph = ctx.new_tensor_view::<[u32]>(&h_zero_ph).unwrap();
                    let us_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
                    if pass % 2 == 0 {
                        crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_sort, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    } else {
                        crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_alt, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    }
                    let sc_cfg = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
                    crate::scan::radix_scan::launch(sc_cfg, ctx, m, &mut d_ph, padded_thread_blocks).unwrap();
                    let ds_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, (BIN_PART_SIZE + RADIX) * 4);
                    if pass % 2 == 0 {
                        crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_sort, &mut d_alt, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    } else {
                        crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_alt, &mut d_sort, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    }
                }
            }

            // Benchmark: 10 iterations
            let iters = 10;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let mut d_sort = ctx.new_tensor_view::<[u32]>(&data).unwrap();
                let mut d_alt = ctx.new_tensor_view::<[u32]>(&h_alt).unwrap();
                let mut d_global_hist = ctx.new_tensor_view::<[u32]>(&h_zero_gh).unwrap();
                for pass in 0..RADIX_PASSES {
                    let radix_shift = pass * RADIX_LOG;
                    let mut d_ph = ctx.new_tensor_view::<[u32]>(&h_zero_ph).unwrap();
                    let us_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
                    if pass % 2 == 0 {
                        crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_sort, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    } else {
                        crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_alt, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    }
                    let sc_cfg = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
                    crate::scan::radix_scan::launch(sc_cfg, ctx, m, &mut d_ph, padded_thread_blocks).unwrap();
                    let ds_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, (BIN_PART_SIZE + RADIX) * 4);
                    if pass % 2 == 0 {
                        crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_sort, &mut d_alt, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    } else {
                        crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_alt, &mut d_sort, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                    }
                }
                // Sync to ensure GPU finished
                let mut result = vec![0u32; data.len()];
                d_sort.copy_to_host(&mut result).unwrap();
            }
            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / iters as f64;
            eprintln!(
                "bench_sort_1m: {:.1} us/sort ({} iters, {:.1} ms total, n={})",
                avg_us, iters, elapsed.as_millis() as f64, n
            );

            // Verify correctness on last run
            let mut d_sort = ctx.new_tensor_view::<[u32]>(&data).unwrap();
            let mut d_alt = ctx.new_tensor_view::<[u32]>(&vec![0u32; data.len()]).unwrap();
            let mut d_global_hist = ctx.new_tensor_view::<[u32]>(&h_zero_gh).unwrap();
            for pass in 0..RADIX_PASSES {
                let radix_shift = pass * RADIX_LOG;
                let mut d_ph = ctx.new_tensor_view::<[u32]>(&h_zero_ph).unwrap();
                let us_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
                if pass % 2 == 0 {
                    crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_sort, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                } else {
                    crate::upsweep::radix_upsweep::launch(us_cfg, ctx, m, &d_alt, &mut d_global_hist, &mut d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                }
                let sc_cfg = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
                crate::scan::radix_scan::launch(sc_cfg, ctx, m, &mut d_ph, padded_thread_blocks).unwrap();
                let ds_cfg = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, (BIN_PART_SIZE + RADIX) * 4);
                if pass % 2 == 0 {
                    crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_sort, &mut d_alt, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                } else {
                    crate::downsweep::radix_downsweep::launch(ds_cfg, ctx, m, &d_alt, &mut d_sort, &d_global_hist, &d_ph, size, radix_shift, padded_thread_blocks).unwrap();
                }
            }
            let mut result = vec![0u32; data.len()];
            d_sort.copy_to_host(&mut result).unwrap();
            let mut expected = data.clone();
            expected.sort();
            assert_eq!(result, expected);
        });
    }
}
