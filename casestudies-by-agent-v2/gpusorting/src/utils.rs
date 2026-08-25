//! Warp-level collective primitives.
//!
//! Every routine here is written so that *whole warps* participate. Callers must
//! never invoke them from a conditional that splits a warp; instead, widen the
//! guard to a warp boundary and feed the identity element (`0`) to the lanes that
//! have no real work. `scan_32_leading` below packages that idiom up.

use crunchy::unroll;
use gpu::prelude::*;

pub const LANE_COUNT: u32 = 32;
pub const LANE_LOG: u32 = 5;
pub const LANE_MASK: u32 = 31;

/// Inclusive prefix sum across the 32 lanes of a warp.
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan(val: u32) -> u32 {
    let mut x = val;
    let lane = lane_id();
    unroll! {
        for k in 0..5 {
            let delta: u32 = 1 << k;
            let (t, _) = gpu::shuffle!(up, x, delta, 32);
            if lane >= delta {
                x += t;
            }
        }
    }
    x
}

/// Exclusive prefix sum across the 32 lanes of a warp.
#[gpu::device]
#[inline(always)]
pub fn exclusive_warp_scan(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let (t, _) = gpu::shuffle!(up, inc, 1u32, 32);
    if lane_id() != 0 { t } else { 0 }
}

/// Inclusive scan rotated left by one lane: lane `k` receives the inclusive sum
/// of lanes `0..k`, and lane 0 receives the total of the whole warp.
///
/// This is the trick CUB uses to fold "exclusive prefix" and "warp total" into a
/// single register.
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan_circular_shift(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let src = (lane_id() + LANE_MASK) & LANE_MASK;
    let (shifted, _) = gpu::shuffle!(idx, inc, src, 32);
    shifted
}

/// Index of the lowest set bit of `mask`, i.e. CUDA's `__ffs(mask) - 1`.
///
/// Lowered to a single `brev`+`clz` pair by the SeGuRu backend now that the
/// `cttz` intrinsic is supported; the previous port emulated this with a
/// 32-iteration serial loop executed once per key.
#[gpu::device]
#[inline(always)]
pub fn lowest_set_bit(mask: u32) -> u32 {
    mask.trailing_zeros()
}

/// Bitmask of the lanes strictly below the caller, i.e. CUDA's `getLaneMaskLt()`.
#[gpu::device]
#[inline(always)]
pub fn lane_mask_lt() -> u32 {
    let lane = lane_id();
    if lane > 0 { (1u32 << lane) - 1 } else { 0 }
}
