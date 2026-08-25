//! Host-side driver: four LSD passes of upsweep → scan → downsweep.

use gpu_host::gpu_config;

use crate::{
    clear::{clear_grid, clear_padded_len, clear_u32, CLEAR_THREADS},
    downsweep::radix_downsweep,
    padded_thread_blocks, pack_padded,
    scan::radix_scan,
    thread_blocks, unpack,
    upsweep::radix_upsweep,
    BIN_PART_SIZE, DOWNSWEEP_THREADS, RADIX, RADIX_LOG, RADIX_PASSES, SCAN_THREADS, SMEM_WORDS,
    U32_4,
    UPSWEEP_THREADS,
};

/// Sort `keys` ascending on the GPU.
///
/// `iters` sorts are run back to back inside one context, after `warmup` untimed
/// ones; the returned time is the mean milliseconds of a single sort. Allocation
/// and the host transfers stay outside the timed region, so this is a kernel-only
/// measurement. A radix sort's cost is data independent, so re-sorting the
/// already-sorted buffer measures exactly the same work.
pub fn radix_sort_timed(keys: &[u32], warmup: usize, iters: usize) -> (Vec<u32>, f64) {
    let n = keys.len();
    let host_in = pack_padded(keys);
    let vec_len = host_in.len();
    let tb = thread_blocks(n);
    let ptb = padded_thread_blocks(n);
    let iters = iters.max(1);

    let gh_len = clear_padded_len((RADIX * RADIX_PASSES) as usize);
    let ph_len = clear_padded_len((RADIX * ptb) as usize);

    let mut elapsed_ms = 0.0f64;
    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let scratch = vec![U32_4::default(); vec_len];
        let zeros_gh = vec![0u32; gh_len];
        let zeros_ph = vec![0u32; ph_len];

        let mut d_a = ctx.new_tensor_view::<[U32_4]>(&host_in).unwrap();
        let mut d_b = ctx.new_tensor_view::<[U32_4]>(&scratch).unwrap();
        let mut d_gh = ctx.new_tensor_view::<[u32]>(&zeros_gh).unwrap();
        let mut d_ph = ctx.new_tensor_view::<[u32]>(&zeros_ph).unwrap();

        // One complete four-pass sort. A macro rather than a closure, because the
        // context and module types are not nameable here and every launch needs a
        // fresh mutable borrow of the buffers.
        macro_rules! sort_once {
            () => {{
                let gh_cfg = gpu_config!(clear_grid(gh_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
                clear_u32::launch(gh_cfg, ctx, m, &mut d_gh).unwrap();

                for pass in 0..RADIX_PASSES {
                    let shift = pass * RADIX_LOG;

                    let ph_cfg =
                        gpu_config!(clear_grid(ph_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
                    clear_u32::launch(ph_cfg, ctx, m, &mut d_ph).unwrap();

                    let up_cfg = gpu_config!(tb, 1, 1, @const UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
                    let scan_cfg =
                        gpu_config!(RADIX, 1, 1, @const SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
                    let down_cfg = gpu_config!(
                        tb, 1, 1, @const DOWNSWEEP_THREADS, 1, 1,
                        SMEM_WORDS * 4
                    );

                    if pass % 2 == 0 {
                        radix_upsweep::launch(
                            up_cfg, ctx, m, &d_a, &mut d_gh, &mut d_ph, shift, ptb,
                        )
                        .unwrap();
                        radix_scan::launch(scan_cfg, ctx, m, &mut d_ph, ptb).unwrap();
                        radix_downsweep::launch(
                            down_cfg, ctx, m,
                            &d_a.flatten(), &mut d_b.flatten(), &d_gh, &d_ph, shift, ptb,
                        )
                        .unwrap();
                    } else {
                        radix_upsweep::launch(
                            up_cfg, ctx, m, &d_b, &mut d_gh, &mut d_ph, shift, ptb,
                        )
                        .unwrap();
                        radix_scan::launch(scan_cfg, ctx, m, &mut d_ph, ptb).unwrap();
                        radix_downsweep::launch(
                            down_cfg, ctx, m,
                            &d_b.flatten(), &mut d_a.flatten(), &d_gh, &d_ph, shift, ptb,
                        )
                        .unwrap();
                    }
                }
            }};
        }

        for _ in 0..warmup {
            sort_once!();
        }
        ctx.sync().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            sort_once!();
        }
        ctx.sync().unwrap();
        elapsed_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let mut host_out = vec![U32_4::default(); vec_len];
        d_a.copy_to_host(&mut host_out).unwrap();
        unpack(&host_out, n)
    });
    (out, elapsed_ms)
}

/// Sort `keys` ascending on the GPU.
pub fn radix_sort(keys: &[u32]) -> Vec<u32> {
    radix_sort_timed(keys, 0, 1).0
}
