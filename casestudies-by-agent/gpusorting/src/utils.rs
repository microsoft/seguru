use gpu::prelude::*;

// ============================================================================
// Warp-level primitives — SeGuRu equivalents of CUDA Utils.cuh
// See cuda-ref/GPUSortingCUDA/Utils.cuh for originals.
//
// CUDA InclusiveWarpScan:
//   for (int i = 1; i <= 16; i <<= 1) {
//       const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
//       if (getLaneId() >= i) val += t;
//   }
//   return val;
//
// CUDA InclusiveWarpScanCircularShift:
//   [InclusiveWarpScan body]
//   return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
//
// CUDA ExclusiveWarpScan:
//   [InclusiveWarpScan body]
//   const uint32_t t = __shfl_up_sync(0xffffffff, val, 1, 32);
//   return getLaneId() ? t : 0;
//
// CUDA WarpReduceSum:
//   for (int mask = 16; mask; mask >>= 1)
//       val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);
//   return val;
//
// Note: CUDA "Active" variants use __activemask(). SeGuRu always uses full-
// warp shuffles since partial-warp scans occur under guards where active
// threads form complete warps.
//
// IMPORTANT: SeGuRu JIT hangs when warp shuffle scans are called inside
// narrow conditionals (tid < 8). For inter-warp scans on few elements,
// use sequential_exclusive_scan_smem / sequential_inclusive_scan_smem instead.
// ============================================================================

/// Inclusive warp scan (sum) via shuffle-up.
/// CUDA: InclusiveWarpScan in Utils.cuh
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan(val: u32) -> u32 {
    let mut x = val;
    // CUDA: for (int i = 1; i <= 16; i <<= 1)
    //           val += __shfl_up_sync(0xffffffff, val, i, 32) if getLaneId() >= i
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

/// Exclusive warp scan: inclusive scan then shift right by 1.
/// CUDA: ExclusiveWarpScan in Utils.cuh
///   return getLaneId() ? __shfl_up_sync(0xffffffff, val, 1, 32) : 0;
#[gpu::device]
#[inline(always)]
pub fn exclusive_warp_scan(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let (t, _) = gpu::shuffle!(up, inc, 1u32, 32);
    if lane_id() != 0 { t } else { 0 }
}

/// Inclusive warp scan with circular shift: lane k gets lane (k-1 mod 32)'s
/// inclusive scan result. This converts inclusive→exclusive with the warp
/// total landing in lane 0 (which gets lane 31's value).
/// CUDA: InclusiveWarpScanCircularShift in Utils.cuh
///   return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
///   // (getLaneId() + 31) & 31 = (lane - 1) mod 32  →  lane k reads from lane k-1
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan_circular_shift(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    // CUDA: source_lane = (getLaneId() + LANE_MASK) & LANE_MASK = (lane + 31) & 31
    let src_lane = (lane_id() + 31) & 31;
    let (shifted, _) = gpu::shuffle!(idx, inc, src_lane, 32);
    shifted
}

/// Warp reduce sum via xor shuffle.
/// CUDA: WarpReduceSum in Utils.cuh
///   for (int mask = 16; mask; mask >>= 1)
///       val += __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);
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
