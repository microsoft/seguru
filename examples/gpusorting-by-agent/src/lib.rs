#![no_std]

pub mod utils;
pub mod upsweep;
pub mod scan;
pub mod downsweep;

// 8-bit LSD radix sort constants
pub const RADIX: u32 = 256;
pub const RADIX_LOG: u32 = 8;
pub const RADIX_MASK: u32 = 255;
pub const RADIX_PASSES: u32 = 4;

// Upsweep kernel config
pub const UPSWEEP_THREADS: u32 = 128;
pub const PART_SIZE: u32 = 7680;

// Scan kernel config
pub const SCAN_THREADS: u32 = 128;

// Downsweep kernel config
pub const DOWNSWEEP_THREADS: u32 = 512;
pub const BIN_WARPS: u32 = 16;
pub const BIN_KEYS_PER_THREAD: u32 = 15;
pub const BIN_PART_SIZE: u32 = 7680;
pub const BIN_SUB_PART_SIZE: u32 = 480;
pub const BIN_HISTS_SIZE: u32 = 4096;

pub const LANE_COUNT: u32 = 32;
pub const LANE_MASK: u32 = 31;
pub const LANE_LOG: u32 = 5;

#[cfg(test)]
mod tests;

#[cfg(test)]
extern crate alloc;

#[cfg(test)]
pub fn dispatch_radix_sort(
    ctx: &gpu_host::GpuCtxZeroGuard<'_, '_>,
    m: &gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    sort_buf: &mut gpu_host::TensorViewMut<[u32]>,
    alt_buf: &mut gpu_host::TensorViewMut<[u32]>,
    global_hist: &mut gpu_host::TensorViewMut<[u32]>,
    pass_hist: &mut gpu_host::TensorViewMut<[u32]>,
    size: u32,
) {
    let thread_blocks = (size + PART_SIZE - 1) / PART_SIZE;
    let upsweep_smem = (RADIX * 2) * 4;
    let scan_smem = SCAN_THREADS * 4;
    let downsweep_smem = (BIN_PART_SIZE + RADIX) * 4;

    let shifts = [0u32, 8, 16, 24];

    for (pass_idx, &shift) in shifts.iter().enumerate() {
        // Zero global histogram
        let zeros = alloc::vec![0u32; (RADIX * RADIX_PASSES) as usize];
        global_hist.copy_from_host(&zeros).expect("zero ghist");

        if pass_idx % 2 == 0 {
            // Even pass: sort → alt
            let config = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
            upsweep::radix_upsweep::launch(config, ctx, m, sort_buf, global_hist, pass_hist, size, shift).expect("upsweep");

            let config = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
            scan::radix_scan::launch(config, ctx, m, pass_hist, thread_blocks).expect("scan");

            let config = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
            downsweep::radix_downsweep::launch(config, ctx, m, sort_buf, alt_buf, global_hist, pass_hist, size, shift).expect("downsweep");
        } else {
            // Odd pass: alt → sort
            let config = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
            upsweep::radix_upsweep::launch(config, ctx, m, alt_buf, global_hist, pass_hist, size, shift).expect("upsweep");

            let config = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
            scan::radix_scan::launch(config, ctx, m, pass_hist, thread_blocks).expect("scan");

            let config = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
            downsweep::radix_downsweep::launch(config, ctx, m, alt_buf, sort_buf, global_hist, pass_hist, size, shift).expect("downsweep");
        }
    }
}
