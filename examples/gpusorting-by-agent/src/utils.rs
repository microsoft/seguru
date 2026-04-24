use gpu::prelude::*;

/// Inclusive warp scan (sum) using shuffle-up.
/// Equivalent to CUDA InclusiveWarpScan in Utils.cuh.
/// Each lane gets the sum of all values at lane indices <= its own.
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan(val: u32) -> u32 {
    let mut x = val;
    let mut i: u32 = 1;
    while i <= 16 {
        let (t, _) = gpu::shuffle!(up, x, i, 32);
        if lane_id() >= i {
            x += t;
        }
        i <<= 1;
    }
    x
}

/// Exclusive warp scan: inclusive scan, then shift right by 1 via shuffle.
/// Equivalent to CUDA ExclusiveWarpScan in Utils.cuh.
#[gpu::device]
#[inline(always)]
pub fn exclusive_warp_scan(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let (t, _) = gpu::shuffle!(up, inc, 1u32, 32);
    if lane_id() != 0 { t } else { 0 }
}

/// Inclusive warp scan with circular shift: after inclusive scan,
/// the result is rotated so lane 0 gets lane 31's value, lane 1 gets lane 0's, etc.
/// Used to convert inclusive scan → exclusive by rotating results.
/// Equivalent to CUDA InclusiveWarpScanCircularShift in Utils.cuh.
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan_circular_shift(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let next_lane = (lane_id() + 1) & 31;
    let (shifted, _) = gpu::shuffle!(idx, inc, next_lane, 32);
    shifted
}

/// Warp reduce sum using xor shuffle.
/// Equivalent to CUDA WarpReduceSum in Utils.cuh.
#[gpu::device]
#[inline(always)]
pub fn warp_reduce_sum(val: u32) -> u32 {
    let mut x = val;
    let mut mask: u32 = 16;
    while mask > 0 {
        let (t, _) = gpu::shuffle!(xor, x, mask, 32);
        x += t;
        mask >>= 1;
    }
    x
}
