//! Host driver for the OneSweep sort: one histogram, one scan, four fused passes.
//!
//! Contrast with `driver.rs`, which launches `clear + upsweep + scan + downsweep`
//! for each of the four digits — twelve kernels and two reads of the key array
//! per digit. This launches six kernels in total and reads the keys once per
//! digit.

use gpu_host::gpu_config;

use crate::{
    clear::{clear_grid, clear_padded_len, clear_u32, CLEAR_THREADS},
    onesweep::{
        digit_binning_pass, global_histogram, onesweep_scan, pass_hist_len, BIN_SMEM_WORDS,
        GH_SMEM_WORDS,
    },
    pack_padded, thread_blocks, unpack, BIN_PART_SIZE, DOWNSWEEP_THREADS, PART_SIZE, RADIX,
    RADIX_LOG, RADIX_PASSES, U32_4, UPSWEEP_THREADS,
};

/// Sort `keys` ascending with OneSweep. Same contract as `radix_sort_timed`:
/// kernel-only timing, mean milliseconds of one sort.
pub fn onesweep_sort_timed(keys: &[u32], warmup: usize, iters: usize) -> (Vec<u32>, f64) {
    onesweep_sort_inner(keys, warmup, iters, false)
}

/// When `skip_lookback` is set the binning kernel neither publishes its tile
/// reduction nor consumes a predecessor prefix, so the decoupled look-back is
/// elided entirely. The result is wrong by construction; this exists only to
/// price the look-back, which is otherwise indistinguishable from the rest of the
/// binning kernel.
pub fn onesweep_sort_inner(
    keys: &[u32],
    warmup: usize,
    iters: usize,
    skip_lookback: bool,
) -> (Vec<u32>, f64) {
    let n = keys.len();
    let host_in = pack_padded(keys);
    let vec_len = host_in.len();
    let hist_blocks = thread_blocks(n);
    // The binning tile and the histogram tile are both PART_SIZE keys here, so the
    // two grids coincide; kept separate because they need not.
    let tiles = ((vec_len * 4) as u32).div_ceil(BIN_PART_SIZE);
    let iters = iters.max(1);

    let gh_len = clear_padded_len((RADIX * RADIX_PASSES) as usize);
    let ph_len = clear_padded_len(pass_hist_len(tiles));
    let idx_len = clear_padded_len(RADIX_PASSES as usize);

    let lb: u32 = if skip_lookback { 0 } else { 1 };

    let mut elapsed_ms = 0.0f64;
    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let scratch = vec![U32_4::default(); vec_len];
        let zeros_gh = vec![0u32; gh_len];
        let zeros_ph = vec![0u32; ph_len];
        let zeros_idx = vec![0u32; idx_len];

        let mut d_a = ctx.new_tensor_view::<[U32_4]>(&host_in).unwrap();
        let mut d_b = ctx.new_tensor_view::<[U32_4]>(&scratch).unwrap();
        let mut d_gh = ctx.new_tensor_view::<[u32]>(&zeros_gh).unwrap();
        let mut d_ph = ctx.new_tensor_view::<[u32]>(&zeros_ph).unwrap();
        let mut d_idx = ctx.new_tensor_view::<[u32]>(&zeros_idx).unwrap();

        macro_rules! sort_once {
            () => {{
                // The look-back flags and the tile counters must start clean: a
                // stale FLAG_INCLUSIVE from the previous sort would let a tile
                // terminate its look-back on garbage. (Deliberately seeding them
                // INCLUSIVE to short-circuit the spin does *not* work: the tile's
                // own `atomic_addi` publish then lands on an already-tagged slot,
                // the flag field wraps to 3, and successors spin forever. The
                // `lb` kernel argument elides the look-back instead.)
                let gh_cfg = gpu_config!(clear_grid(gh_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
                clear_u32::launch(gh_cfg, ctx, m, &mut d_gh).unwrap();
                let ph_cfg = gpu_config!(clear_grid(ph_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
                clear_u32::launch(ph_cfg, ctx, m, &mut d_ph).unwrap();
                let idx_cfg = gpu_config!(clear_grid(idx_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
                clear_u32::launch(idx_cfg, ctx, m, &mut d_idx).unwrap();

                // One pass over the keys builds all four digits' histograms.
                let ghist_cfg = gpu_config!(
                    hist_blocks, 1, 1, @const UPSWEEP_THREADS, 1, 1, GH_SMEM_WORDS * 4
                );
                global_histogram::launch(ghist_cfg, ctx, m, &d_a, &mut d_gh).unwrap();

                // One block per digit pass seeds that pass's look-back chain.
                let scan_cfg =
                    gpu_config!(RADIX_PASSES, 1, 1, @const RADIX, 1, 1, RADIX * 4);
                onesweep_scan::launch(scan_cfg, ctx, m, &d_gh, &mut d_ph, tiles).unwrap();

                for pass in 0..RADIX_PASSES {
                    let shift = pass * RADIX_LOG;
                    let bin_cfg = gpu_config!(
                        tiles, 1, 1, @const DOWNSWEEP_THREADS, 1, 1, BIN_SMEM_WORDS * 4
                    );
                    if pass % 2 == 0 {
                        digit_binning_pass::launch(
                            bin_cfg, ctx, m,
                            &d_a.flatten(), &mut d_b.flatten(), &mut d_ph, &mut d_idx, shift, tiles, lb,
                        )
                        .unwrap();
                    } else {
                        digit_binning_pass::launch(
                            bin_cfg, ctx, m,
                            &d_b.flatten(), &mut d_a.flatten(), &mut d_ph, &mut d_idx, shift, tiles, lb,
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

/// Sort `keys` ascending with OneSweep.
pub fn onesweep_sort(keys: &[u32]) -> Vec<u32> {
    onesweep_sort_timed(keys, 0, 1).0
}

const _: () = assert!(PART_SIZE == BIN_PART_SIZE);
