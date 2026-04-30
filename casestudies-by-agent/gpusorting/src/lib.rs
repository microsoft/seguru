pub mod downsweep;
pub mod scan;
pub mod upsweep;
pub mod utils;

#[cfg(test)]
mod tests;

// ============================================================================
// Constants from CUDA DeviceRadixSort.cu / DeviceRadixSortDispatcher.cuh
// ============================================================================
pub const RADIX: u32 = 256;
pub const RADIX_LOG: u32 = 8;
pub const RADIX_MASK: u32 = 255;
pub const RADIX_PASSES: u32 = 4;

pub const LANE_COUNT: u32 = 32;
pub const LANE_MASK: u32 = 31;
pub const LANE_LOG: u32 = 5;

// Upsweep: 128 threads, PART_SIZE = 7680 keys/block
pub const UPSWEEP_THREADS: u32 = 128;
pub const PART_SIZE: u32 = 7680;

// Scan: 128 threads
pub const SCAN_THREADS: u32 = 128;

// Downsweep: 512 threads (16 warps), BIN_PART_SIZE = 7680 keys/block
pub const DOWNSWEEP_THREADS: u32 = 512;
pub const BIN_PART_SIZE: u32 = 7680;
pub const BIN_HISTS_SIZE: u32 = 4096; // BIN_WARPS * RADIX
pub const BIN_SUB_PART_SIZE: u32 = 480; // BIN_PART_SIZE / BIN_WARPS
pub const BIN_WARPS: u32 = 16;
pub const BIN_KEYS_PER_THREAD: u32 = 15; // BIN_PART_SIZE / DOWNSWEEP_THREADS
