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
