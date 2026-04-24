# GPUSorting DeviceRadixSort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port DeviceRadixSort from GPUSorting (CUDA) to SeGuRu Rust GPU kernels using shuffle-based warp scans and SharedAtomic ranking (no ballot/vote intrinsics).

**Architecture:** 8-bit LSD radix sort with 4 passes (shifts 0,8,16,24). Three GPU kernels per pass: Upsweep (histogram), Scan (prefix sum), Downsweep (scatter). Host orchestrator drives ping-pong buffers. CUDA reference build for benchmarking.

**Tech Stack:** SeGuRu (`gpu`, `gpu_host`), `gpu::shuffle!` for warp scans, `SharedAtomic` for ranking, `rand` for test data.

---

## File Structure

```
examples/gpusorting-by-agent/
├── Cargo.toml              # Crate manifest (gpu, gpu_host, rand)
├── src/
│   ├── lib.rs              # Constants, module declarations, host orchestrator
│   ├── utils.rs            # Warp-level shuffle scan primitives
│   ├── upsweep.rs          # Upsweep kernel (per-block histogram)
│   ├── scan.rs             # Scan kernel (exclusive prefix sum)
│   └── downsweep.rs        # Downsweep kernel (rank + scatter)
├── cuda/
│   ├── Makefile            # nvcc build for reference
│   └── device_radix_sort_bench.cu  # Standalone CUDA benchmark
├── README.md
└── REPORT.md
```

## Constants (from CUDA reference)

```rust
const RADIX: u32 = 256;
const RADIX_LOG: u32 = 8;
const RADIX_MASK: u32 = 255;
const RADIX_PASSES: u32 = 4;

// Upsweep
const UPSWEEP_THREADS: u32 = 128;
const PART_SIZE: u32 = 7680;

// Scan
const SCAN_THREADS: u32 = 128;

// Downsweep
const DOWNSWEEP_THREADS: u32 = 512;
const BIN_WARPS: u32 = 16; // 512 / 32
const BIN_KEYS_PER_THREAD: u32 = 15;
const BIN_PART_SIZE: u32 = 7680; // 512 * 15
const BIN_SUB_PART_SIZE: u32 = 480; // 32 * 15
const BIN_HISTS_SIZE: u32 = 4096; // 16 warps * 256 bins
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `examples/gpusorting-by-agent/Cargo.toml`
- Create: `examples/gpusorting-by-agent/src/lib.rs`
- Modify: `examples/Cargo.toml` (add workspace member)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "gpusorting-by-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
gpu = { workspace = true }
gpu_host = { workspace = true }

[dev-dependencies]
rand = "0.9"
```

- [ ] **Step 2: Create src/lib.rs with constants and module declarations**

```rust
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
```

- [ ] **Step 3: Add workspace member to examples/Cargo.toml**

Add `"gpusorting-by-agent"` to the `[workspace] members` list in `examples/Cargo.toml`.

- [ ] **Step 4: Verify scaffolding compiles**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -20`
Expected: Compilation errors about missing modules (utils, upsweep, scan, downsweep, tests) — that's fine, scaffolding is correct.

- [ ] **Step 5: Commit**

```bash
git add examples/gpusorting-by-agent/ examples/Cargo.toml
git commit -m "feat(gpusorting): scaffold crate with constants and module structure"
```

---

### Task 2: Warp-Level Shuffle Utilities

**Files:**
- Create: `examples/gpusorting-by-agent/src/utils.rs`

These are shuffle-based replacements for the CUDA `Utils.cuh` warp scan primitives. No ballot/vote — pure shuffle operations.

- [ ] **Step 1: Write utils.rs with warp scan primitives**

```rust
#![allow(unused)]
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
        if gpu::dim::lane_id() >= i {
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
    if gpu::dim::lane_id() != 0 { t } else { 0 }
}

/// Inclusive warp scan with circular shift: the result of lane 31
/// goes to lane 0, lane 0's result goes to lane 1, etc.
/// Equivalent to CUDA InclusiveWarpScanCircularShift in Utils.cuh.
/// Used to convert inclusive scan to exclusive by rotating.
#[gpu::device]
#[inline(always)]
pub fn inclusive_warp_scan_circular_shift(val: u32) -> u32 {
    let inc = inclusive_warp_scan(val);
    let next_lane = (gpu::dim::lane_id() + 1) & LANE_MASK;
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

const LANE_MASK: u32 = 31;
```

- [ ] **Step 2: Verify compilation**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -20`
Expected: Only errors from missing test/kernel modules, not from utils.rs.

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/utils.rs
git commit -m "feat(gpusorting): add shuffle-based warp scan utilities"
```

---

### Task 3: Upsweep Kernel (Per-Block Histogram)

**Files:**
- Create: `examples/gpusorting-by-agent/src/upsweep.rs`

The Upsweep kernel computes per-block histograms of 8-bit digits. Each block processes PART_SIZE (7680) keys. Uses shared memory atomics for histogram accumulation and shuffle-based warp scans for the prefix sum that feeds into the global histogram.

**Algorithm:**
1. Clear shared memory (2 × 256 bins for 2 "waves" of 64 threads)
2. Each thread loops over its partition, extracts 8-bit digit, atomically increments shared histogram
3. Reduce 2 wave histograms to 1
4. Write per-block histogram to `passHist[digit * gridDim + blockIdx]`
5. Warp-scan the histogram and atomically add to `globalHist`

- [ ] **Step 1: Write upsweep.rs**

```rust
use gpu::prelude::*;
use crate::{RADIX, RADIX_MASK, PART_SIZE, LANE_MASK};

/// Upsweep kernel: compute per-block histogram of 8-bit digits.
///
/// - `sort`: input keys
/// - `global_hist`: global histogram [RADIX * 4] — accumulated across all blocks
/// - `pass_hist`: per-block histograms [RADIX * num_blocks]
/// - `size`: number of keys
/// - `radix_shift`: bit shift for current pass (0, 8, 16, or 24)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_upsweep(
    sort: &[u32],
    global_hist: &mut [u32],
    pass_hist: &mut [u32],
    size: u32,
    radix_shift: u32,
) {
    let tid = gpu::thread_id::<gpu::DimX>();
    let block_id = gpu::block_id::<gpu::DimX>();
    let block_dim = gpu::block_dim::<gpu::DimX>();
    let grid_dim = gpu::grid_dim::<gpu::DimX>();

    // Allocate shared memory: 2 histograms of 256 bins each (for 2 "waves")
    let s_hist = smem_alloc.alloc::<u32>((RADIX * 2) as usize);

    // Clear shared memory
    {
        let mut sc = s_hist.chunk_mut(gpu::MapLinear::new(1));
        let mut i = tid;
        while i < RADIX * 2 {
            sc[i as usize] = 0;
            i += block_dim;
        }
    }
    gpu::sync_threads();

    // Compute histogram using shared memory atomics
    // Each "wave" of 64 threads gets its own 256-bin histogram
    let wave_offset = (tid / 64) * RADIX;
    let s_hist_atomic = gpu::sync::SharedAtomic::new(&mut *s_hist);

    if block_id < grid_dim - 1 {
        // Full partition — process PART_SIZE keys
        let part_start = block_id * PART_SIZE;
        let part_end = part_start + PART_SIZE;
        let mut i = part_start + tid;
        while i < part_end {
            let key = sort[i as usize];
            let digit = (key >> radix_shift) & RADIX_MASK;
            s_hist_atomic.index((wave_offset + digit) as usize).atomic_addi(1);
            i += block_dim;
        }
    } else {
        // Last block — handle tail
        let part_start = block_id * PART_SIZE;
        let mut i = part_start + tid;
        while i < size {
            let key = sort[i as usize];
            let digit = (key >> radix_shift) & RADIX_MASK;
            s_hist_atomic.index((wave_offset + digit) as usize).atomic_addi(1);
            i += block_dim;
        }
    }
    gpu::sync_threads();

    // Reduce 2 wave histograms into the first one, write to pass_hist
    // Then do warp-level inclusive scan with circular shift for global_hist update
    {
        let mut sc = s_hist.chunk_mut(gpu::MapLinear::new(1));
        let mut i = tid;
        while i < RADIX {
            // Merge wave 1 into wave 0
            let val = sc[i as usize] + sc[(i + RADIX) as usize];
            sc[i as usize] = val;

            // Write per-block histogram
            let mut ph = gpu::chunk_mut(pass_hist, gpu::MapLinear::new(1));
            ph[(i * grid_dim + block_id) as usize] = val;

            // Circular-shift inclusive scan for exclusive prefix sum
            sc[i as usize] = crate::utils::inclusive_warp_scan_circular_shift(val);
            i += block_dim;
        }
    }
    gpu::sync_threads();

    // Second-level scan: scan the per-warp totals
    {
        let mut sc = s_hist.chunk_mut(gpu::MapLinear::new(1));
        if tid < (RADIX >> 5) {
            let warp_total = sc[(tid << 5) as usize];
            sc[(tid << 5) as usize] = crate::utils::exclusive_warp_scan(warp_total);
        }
    }
    gpu::sync_threads();

    // Atomically add final prefix sums to global histogram
    {
        let sc = s_hist.chunk_mut(gpu::MapLinear::new(1));
        let global_atomic = gpu::sync::Atomic::new(global_hist);

        let mut i = tid;
        while i < RADIX {
            let lane = gpu::dim::lane_id();
            let mut val = sc[i as usize];
            // Add cross-warp prefix
            if lane != 0 {
                let (prev, _) = gpu::shuffle!(idx, sc[(i - 1) as usize], 1u32, 32);
                val += prev;
            }
            let hist_idx = i + (radix_shift << 5);
            global_atomic.index(hist_idx as usize).atomic_addi(val);
            i += block_dim;
        }
    }
}
```

**Note:** The exact shared memory access patterns may need adjustment during implementation. The CUDA reference uses `volatile` pointers and `reinterpret_cast<uint4*>` for vectorized loads which we won't have. The core algorithm is: shared-memory atomic histogram → warp scan → atomic add to global.

- [ ] **Step 2: Verify compilation**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -30`
Expected: May have type errors that need fixing. Fix any issues with SharedAtomic API usage, chunk_mut patterns.

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/upsweep.rs
git commit -m "feat(gpusorting): add upsweep histogram kernel"
```

---

### Task 4: Scan Kernel (Exclusive Prefix Sum)

**Files:**
- Create: `examples/gpusorting-by-agent/src/scan.rs`

The Scan kernel performs an exclusive prefix sum on the per-block pass histogram. Launched with 256 blocks (one per digit), 128 threads each. Each block scans the `passHist[digit * numBlocks .. (digit+1) * numBlocks]` array.

**Algorithm:**
1. Load chunks of `threadBlocks` elements into shared memory
2. Warp-level inclusive scan within each chunk
3. Two-level scan: scan warp totals
4. Write back with circular shift to convert inclusive → exclusive
5. Accumulate running reduction across chunks

- [ ] **Step 1: Write scan.rs**

```rust
use gpu::prelude::*;
use crate::LANE_MASK;

/// Scan kernel: exclusive prefix sum of per-block histograms.
/// Launched with RADIX (256) blocks, SCAN_THREADS (128) threads each.
/// Each block processes one digit's histogram across all blocks.
///
/// - `pass_hist`: per-block histograms [RADIX * num_blocks], modified in-place
/// - `thread_blocks`: number of partition blocks (from upsweep)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_scan(
    pass_hist: &mut [u32],
    thread_blocks: u32,
) {
    let tid = gpu::thread_id::<gpu::DimX>();
    let block_id = gpu::block_id::<gpu::DimX>();
    let block_dim = gpu::block_dim::<gpu::DimX>();

    let s_scan = smem_alloc.alloc::<u32>(block_dim as usize);
    let digit_offset = block_id * thread_blocks;

    let circular_lane_shift = (gpu::dim::lane_id() + 1) & LANE_MASK;
    let partitions_end = (thread_blocks / block_dim) * block_dim;

    let mut reduction: u32 = 0;

    // Process full chunks
    let mut i = tid;
    while i < partitions_end {
        // Load into shared memory
        {
            let mut sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
            sc[tid as usize] = pass_hist[(i + digit_offset) as usize];
        }

        // Intra-warp inclusive scan
        {
            let mut sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
            sc[tid as usize] = crate::utils::inclusive_warp_scan(sc[tid as usize]);
        }
        gpu::sync_threads();

        // Inter-warp scan: scan the last element of each warp
        if tid < (block_dim >> 5) {
            let mut sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
            let idx = ((tid + 1) << 5) - 1;
            sc[idx as usize] = crate::utils::inclusive_warp_scan(sc[idx as usize]);
        }
        gpu::sync_threads();

        // Write back with circular shift + cross-warp prefix + running reduction
        {
            let sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
            let lane = gpu::dim::lane_id();

            let scan_val = if lane != LANE_MASK { sc[tid as usize] } else { 0 };

            let cross_warp = if tid >= 32 {
                let (prev_warp_total, _) = gpu::shuffle!(idx, sc[((tid & !LANE_MASK) - 1) as usize], 0u32, 32);
                prev_warp_total
            } else {
                0
            };

            let out_idx = circular_lane_shift + (i & !LANE_MASK);
            let mut ph = gpu::chunk_mut(pass_hist, gpu::MapLinear::new(1));
            ph[(out_idx + digit_offset) as usize] = scan_val + cross_warp + reduction;
        }

        // Update running reduction
        {
            let sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
            reduction += sc[(block_dim - 1) as usize];
        }
        gpu::sync_threads();

        i += block_dim;
    }

    // Process tail (partial chunk)
    if i < thread_blocks {
        let mut sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
        sc[tid as usize] = pass_hist[(i + digit_offset) as usize];
        sc[tid as usize] = crate::utils::inclusive_warp_scan(sc[tid as usize]);
    }
    gpu::sync_threads();

    if tid < (block_dim >> 5) {
        let mut sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
        let idx = ((tid + 1) << 5) - 1;
        sc[idx as usize] = crate::utils::inclusive_warp_scan(sc[idx as usize]);
    }
    gpu::sync_threads();

    {
        let sc = s_scan.chunk_mut(gpu::MapLinear::new(1));
        let lane = gpu::dim::lane_id();
        let out_idx = circular_lane_shift + (i & !LANE_MASK);

        if out_idx < thread_blocks {
            let scan_val = if lane != LANE_MASK { sc[tid as usize] } else { 0 };
            let cross_warp = if tid >= 32 {
                let (prev, _) = gpu::shuffle!(idx, sc[((tid & !LANE_MASK) - 1) as usize], 0u32, 32);
                prev
            } else {
                0
            };

            let mut ph = gpu::chunk_mut(pass_hist, gpu::MapLinear::new(1));
            ph[(out_idx + digit_offset) as usize] = scan_val + cross_warp + reduction;
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -30`

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/scan.rs
git commit -m "feat(gpusorting): add scan prefix-sum kernel"
```

---

### Task 5: Downsweep Kernel (Rank + Scatter)

**Files:**
- Create: `examples/gpusorting-by-agent/src/downsweep.rs`

The Downsweep kernel is the most complex. It ranks each key within its block using SharedAtomic (replacing CUDA's ballot+popc+ffs WLMS), then scatters keys to their sorted positions.

**Algorithm (shuffle-based, no ballot):**
1. Clear per-warp shared histograms (16 warps × 256 bins = 4096 entries)
2. Each thread loads 15 keys from its partition
3. For each key: extract digit, atomicAdd on per-warp histogram → returns local rank
4. Exclusive prefix sum across warp histograms
5. Update offsets with cross-warp prefix and global histogram offset
6. Scatter keys to shared memory using offsets
7. Scatter from shared memory to global output using `s_localHistogram` offsets

- [ ] **Step 1: Write downsweep.rs**

```rust
use gpu::prelude::*;
use crate::{
    RADIX, RADIX_LOG, RADIX_MASK, LANE_MASK, LANE_LOG,
    BIN_KEYS_PER_THREAD, BIN_PART_SIZE, BIN_SUB_PART_SIZE, BIN_HISTS_SIZE,
};

/// Downsweep kernel: rank and scatter keys to sorted output.
/// Uses SharedAtomic for per-warp digit ranking (replaces CUDA ballot+popc+ffs).
///
/// - `sort`: input keys for this pass
/// - `alt`: output buffer (sorted for this pass)
/// - `global_hist`: global histogram from upsweep [RADIX * 4]
/// - `pass_hist`: scanned per-block histograms [RADIX * num_blocks]
/// - `size`: number of keys
/// - `radix_shift`: bit shift for current pass (0, 8, 16, or 24)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_downsweep(
    sort: &[u32],
    alt: &mut [u32],
    global_hist: &[u32],
    pass_hist: &[u32],
    size: u32,
    radix_shift: u32,
) {
    let tid = gpu::thread_id::<gpu::DimX>();
    let block_id = gpu::block_id::<gpu::DimX>();
    let block_dim = gpu::block_dim::<gpu::DimX>();
    let grid_dim = gpu::grid_dim::<gpu::DimX>();
    let lane_id = gpu::dim::lane_id();
    let warp_idx = tid >> LANE_LOG;

    // Shared memory layout:
    // [0 .. BIN_HISTS_SIZE): per-warp histograms (16 warps × 256 bins)
    // After re-use: also used for key staging during scatter
    // [BIN_HISTS_SIZE .. BIN_HISTS_SIZE + RADIX): local histogram for final offsets
    let s_warp_hists = smem_alloc.alloc::<u32>(BIN_HISTS_SIZE as usize);
    let s_local_hist = smem_alloc.alloc::<u32>(RADIX as usize);

    // Clear per-warp histograms
    {
        let mut sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let mut i = tid;
        while i < BIN_HISTS_SIZE {
            sc[i as usize] = 0;
            i += block_dim;
        }
    }
    gpu::sync_threads();

    // Load keys into registers
    let part_start = block_id * BIN_PART_SIZE;
    let sub_part_start = warp_idx * BIN_SUB_PART_SIZE;
    let mut keys = [0u32; 15]; // BIN_KEYS_PER_THREAD = 15
    let mut offsets = [0u32; 15];

    if block_id < grid_dim - 1 {
        let mut i: u32 = 0;
        let mut t = lane_id + sub_part_start + part_start;
        while i < BIN_KEYS_PER_THREAD {
            keys[i as usize] = sort[t as usize];
            i += 1;
            t += 32; // LANE_COUNT
        }
    } else {
        // Last block: pad with 0xffffffff for out-of-bounds
        let mut i: u32 = 0;
        let mut t = lane_id + sub_part_start + part_start;
        while i < BIN_KEYS_PER_THREAD {
            keys[i as usize] = if t < size { sort[t as usize] } else { 0xffffffff };
            i += 1;
            t += 32;
        }
    }
    gpu::sync_threads();

    // Rank keys using SharedAtomic per-warp histogram
    // (Replaces CUDA WLMS ballot+popc+ffs)
    // Each thread atomically increments its warp's histogram bin for the key's digit.
    // atomicAdd returns the OLD value = local rank within this warp for this digit.
    {
        let s_atomic = gpu::sync::SharedAtomic::new(&mut *s_warp_hists);
        let warp_hist_offset = warp_idx << RADIX_LOG; // warp_idx * 256

        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
            let rank = s_atomic.index((warp_hist_offset + digit) as usize).atomic_addi(1);
            offsets[i as usize] = rank;
            i += 1;
        }
    }
    gpu::sync_threads();

    // Exclusive prefix sum across warp histograms
    // For each digit d, sum the counts from warp 0, 1, ..., 15
    // and compute exclusive prefix so each warp knows its starting offset
    if tid < RADIX {
        let mut sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let mut reduction = sc[tid as usize]; // warp 0's count for digit tid
        let mut w: u32 = 1;
        while w < (BIN_HISTS_SIZE / RADIX) {
            let idx = tid + w * RADIX;
            let warp_count = sc[idx as usize];
            reduction += warp_count;
            // Store exclusive prefix for this warp
            sc[idx as usize] = reduction - warp_count;
            w += 1;
        }

        // Exclusive prefix sum of the per-digit totals using warp scan
        sc[tid as usize] = crate::utils::inclusive_warp_scan_circular_shift(reduction);
    }
    gpu::sync_threads();

    // Second-level scan of warp totals
    if tid < (RADIX >> 5) {
        let mut sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let idx = tid << 5;
        sc[idx as usize] = crate::utils::exclusive_warp_scan(sc[idx as usize]);
    }
    gpu::sync_threads();

    // Finalize: add cross-warp prefix
    if tid < RADIX && lane_id != 0 {
        let mut sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let (prev, _) = gpu::shuffle!(idx, sc[(tid - 1) as usize], 1u32, 32);
        sc[tid as usize] += prev;
    }
    gpu::sync_threads();

    // Update offsets: add warp-level prefix and block-level prefix
    {
        let sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let warp_hist_offset = warp_idx << RADIX_LOG;

        if warp_idx != 0 {
            let mut i: u32 = 0;
            while i < BIN_KEYS_PER_THREAD {
                let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
                offsets[i as usize] += sc[(warp_hist_offset + digit) as usize]
                    + sc[digit as usize];
                i += 1;
            }
        } else {
            let mut i: u32 = 0;
            while i < BIN_KEYS_PER_THREAD {
                let digit = (keys[i as usize] >> radix_shift) & RADIX_MASK;
                offsets[i as usize] += sc[digit as usize];
                i += 1;
            }
        }
    }

    // Load global offsets into s_local_hist
    if tid < RADIX {
        let mut slh = s_local_hist.chunk_mut(gpu::MapLinear::new(1));
        let sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        slh[tid as usize] = global_hist[(tid + (radix_shift << 5)) as usize]
            + pass_hist[(tid * grid_dim + block_id) as usize]
            - sc[tid as usize];
    }
    gpu::sync_threads();

    // Scatter keys into shared memory (using offsets as indices)
    {
        let mut sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let mut i: u32 = 0;
        while i < BIN_KEYS_PER_THREAD {
            sc[offsets[i as usize] as usize] = keys[i as usize];
            i += 1;
        }
    }
    gpu::sync_threads();

    // Scatter from shared memory to global output
    {
        let sc = s_warp_hists.chunk_mut(gpu::MapLinear::new(1));
        let slh = s_local_hist.chunk_mut(gpu::MapLinear::new(1));
        let mut alt_chunk = gpu::chunk_mut(alt, gpu::MapLinear::new(1));

        let limit = if block_id < grid_dim - 1 {
            BIN_PART_SIZE
        } else {
            size - part_start
        };

        let mut i = tid;
        while i < limit {
            let key = sc[i as usize];
            let digit = (key >> radix_shift) & RADIX_MASK;
            let global_pos = slh[digit as usize] + i;
            alt_chunk[global_pos as usize] = key;
            i += block_dim;
        }
    }
}
```

**Note:** The scatter-to-global step writes each key to a data-dependent position. This uses `chunk_mut` with `MapLinear::new(1)` treating the entire output as a flat array. If SeGuRu's bounds checking rejects this pattern (since multiple threads may target non-adjacent positions), we may need to use `unsafe` pointer writes or a `reshape_map!` with a larger mapping. This is the highest-risk step and may need iteration.

- [ ] **Step 2: Verify compilation**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -40`
Expected: Fix any type/API issues.

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/downsweep.rs
git commit -m "feat(gpusorting): add downsweep rank+scatter kernel"
```

---

### Task 6: Host Orchestrator

**Files:**
- Modify: `examples/gpusorting-by-agent/src/lib.rs` (add orchestrator function)

The host orchestrator drives the 4-pass radix sort: for each of the 4 radix passes (shifts 0, 8, 16, 24), launch Upsweep → Scan → Downsweep, ping-ponging between two buffers.

- [ ] **Step 1: Add host orchestrator to lib.rs**

Add to the bottom of `src/lib.rs` (before the `#[cfg(test)]` block):

```rust
#[cfg(test)]
pub fn dispatch_radix_sort(
    ctx: &gpu_host::GpuContext,
    m: &gpu_host::Module,
    sort_buf: &mut gpu_host::TensorView<u32>,
    alt_buf: &mut gpu_host::TensorView<u32>,
    global_hist: &mut gpu_host::TensorView<u32>,
    pass_hist: &mut gpu_host::TensorView<u32>,
    size: u32,
) {
    let thread_blocks = (size + PART_SIZE - 1) / PART_SIZE;

    // Zero global histogram
    let zero_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    global_hist.copy_from_host(&zero_hist).expect("zero global_hist");

    let upsweep_smem = (RADIX * 2) * core::mem::size_of::<u32>() as u32;
    let scan_smem = SCAN_THREADS * core::mem::size_of::<u32>() as u32;
    let downsweep_smem = (BIN_HISTS_SIZE + RADIX) * core::mem::size_of::<u32>() as u32;

    // Pass 0: sort → alt
    {
        let config_up = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
        upsweep::radix_upsweep::launch(config_up, ctx, m, sort_buf, global_hist, pass_hist, size, 0u32).expect("upsweep 0");
        let config_scan = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
        scan::radix_scan::launch(config_scan, ctx, m, pass_hist, thread_blocks).expect("scan 0");
        let config_down = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
        downsweep::radix_downsweep::launch(config_down, ctx, m, sort_buf, alt_buf, global_hist, pass_hist, size, 0u32).expect("downsweep 0");
    }

    // Pass 1: alt → sort
    {
        let zero_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
        global_hist.copy_from_host(&zero_hist).expect("zero global_hist");
        let config_up = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
        upsweep::radix_upsweep::launch(config_up, ctx, m, alt_buf, global_hist, pass_hist, size, 8u32).expect("upsweep 1");
        let config_scan = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
        scan::radix_scan::launch(config_scan, ctx, m, pass_hist, thread_blocks).expect("scan 1");
        let config_down = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
        downsweep::radix_downsweep::launch(config_down, ctx, m, alt_buf, sort_buf, global_hist, pass_hist, size, 8u32).expect("downsweep 1");
    }

    // Pass 2: sort → alt
    {
        let zero_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
        global_hist.copy_from_host(&zero_hist).expect("zero global_hist");
        let config_up = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
        upsweep::radix_upsweep::launch(config_up, ctx, m, sort_buf, global_hist, pass_hist, size, 16u32).expect("upsweep 2");
        let config_scan = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
        scan::radix_scan::launch(config_scan, ctx, m, pass_hist, thread_blocks).expect("scan 2");
        let config_down = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
        downsweep::radix_downsweep::launch(config_down, ctx, m, sort_buf, alt_buf, global_hist, pass_hist, size, 16u32).expect("downsweep 2");
    }

    // Pass 3: alt → sort (final result in sort)
    {
        let zero_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
        global_hist.copy_from_host(&zero_hist).expect("zero global_hist");
        let config_up = gpu_host::gpu_config!(thread_blocks, 1, 1, UPSWEEP_THREADS, 1, 1, upsweep_smem);
        upsweep::radix_upsweep::launch(config_up, ctx, m, alt_buf, global_hist, pass_hist, size, 24u32).expect("upsweep 3");
        let config_scan = gpu_host::gpu_config!(RADIX, 1, 1, SCAN_THREADS, 1, 1, scan_smem);
        scan::radix_scan::launch(config_scan, ctx, m, pass_hist, thread_blocks).expect("scan 3");
        let config_down = gpu_host::gpu_config!(thread_blocks, 1, 1, DOWNSWEEP_THREADS, 1, 1, downsweep_smem);
        downsweep::radix_downsweep::launch(config_down, ctx, m, alt_buf, sort_buf, global_hist, pass_hist, size, 24u32).expect("downsweep 3");
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd examples && cargo check -p gpusorting-by-agent 2>&1 | head -30`

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/lib.rs
git commit -m "feat(gpusorting): add host-side 4-pass orchestrator"
```

---

### Task 7: Correctness Tests

**Files:**
- Create: `examples/gpusorting-by-agent/src/tests.rs`

- [ ] **Step 1: Write tests.rs with correctness and edge-case tests**

```rust
use super::*;
use gpu_host::cuda_ctx;
use rand::Rng;

fn run_sort(input: &[u32]) -> Vec<u32> {
    let n = input.len() as u32;
    let thread_blocks = (n + PART_SIZE - 1) / PART_SIZE;
    let hist_size = (RADIX * thread_blocks) as usize;

    let mut result = input.to_vec();
    let mut alt = vec![0u32; n as usize];
    let mut global_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    let mut pass_hist = vec![0u32; hist_size];

    cuda_ctx(0, |ctx, m| {
        let mut d_sort = ctx.new_tensor_view(result.as_mut_slice()).expect("alloc sort");
        let mut d_alt = ctx.new_tensor_view(alt.as_mut_slice()).expect("alloc alt");
        let mut d_global = ctx.new_tensor_view(global_hist.as_mut_slice()).expect("alloc ghist");
        let mut d_pass = ctx.new_tensor_view(pass_hist.as_mut_slice()).expect("alloc phist");

        dispatch_radix_sort(ctx, m, &mut d_sort, &mut d_alt, &mut d_global, &mut d_pass, n);

        d_sort.copy_to_host(&mut result).expect("copy back");
    });

    result
}

fn is_sorted(data: &[u32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

fn is_permutation(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() { return false; }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort();
    sb.sort();
    sa == sb
}

#[test]
fn test_sort_small_random() {
    let mut rng = rand::rng();
    let input: Vec<u32> = (0..PART_SIZE).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Result is not sorted");
    assert!(is_permutation(&input, &result), "Result is not a permutation of input");
}

#[test]
fn test_sort_1m_random() {
    let mut rng = rand::rng();
    let n = 1_000_000u32;
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "1M random: not sorted");
    assert!(is_permutation(&input, &result), "1M random: not a permutation");
}

#[test]
fn test_sort_already_sorted() {
    let input: Vec<u32> = (0..PART_SIZE).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result));
    assert_eq!(result, input);
}

#[test]
fn test_sort_reverse() {
    let input: Vec<u32> = (0..PART_SIZE).rev().collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result));
}

#[test]
fn test_sort_all_same() {
    let input = vec![42u32; PART_SIZE as usize];
    let result = run_sort(&input);
    assert!(is_sorted(&result));
    assert_eq!(result, input);
}

#[test]
fn test_sort_all_zeros() {
    let input = vec![0u32; PART_SIZE as usize];
    let result = run_sort(&input);
    assert_eq!(result, input);
}

#[test]
fn test_sort_non_multiple_size() {
    // Size that's not a multiple of PART_SIZE to test tail handling
    let mut rng = rand::rng();
    let n = PART_SIZE + 123;
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Non-multiple size: not sorted");
    assert!(is_permutation(&input, &result), "Non-multiple size: not a permutation");
}

#[test]
fn test_sort_two_partitions() {
    let mut rng = rand::rng();
    let n = PART_SIZE * 2;
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Two partitions: not sorted");
    assert!(is_permutation(&input, &result), "Two partitions: not a permutation");
}
```

- [ ] **Step 2: Run tests**

Run: `cd examples && cargo test -p gpusorting-by-agent -- --test-threads=1 2>&1`
Expected: Tests should either pass (if kernels are correct) or fail with clear error messages. Debug any failures.

- [ ] **Step 3: Iterate on kernel bugs until all tests pass**

This is the debugging step. Common issues:
- SharedAtomic API misuse (wrong type, wrong index)
- Off-by-one in partition boundaries
- Shared memory size too small
- chunk_mut scatter writes rejected by bounds checking
- Warp scan edge cases (lane 0 handling)

Fix issues in upsweep.rs, scan.rs, or downsweep.rs as needed.

- [ ] **Step 4: Commit**

```bash
git add examples/gpusorting-by-agent/src/tests.rs
git commit -m "feat(gpusorting): add correctness tests (1M random, edge cases)"
```

---

### Task 8: CUDA Reference Build

**Files:**
- Create: `examples/gpusorting-by-agent/cuda/Makefile`
- Create: `examples/gpusorting-by-agent/cuda/device_radix_sort_bench.cu`

Extract minimal CUDA code from GPUSorting for benchmarking comparison.

- [ ] **Step 1: Create standalone CUDA benchmark file**

Create `cuda/device_radix_sort_bench.cu` that includes the 3 kernels (Upsweep, Scan, DownsweepKeysOnly), the Utils.cuh warp primitives inlined, and a benchmark harness that:
1. Allocates buffers for a given size
2. Fills with random data
3. Times the 4-pass sort
4. Validates output is sorted
5. Prints throughput (M keys/sec)

The file should be self-contained — no external headers.

- [ ] **Step 2: Create Makefile**

```makefile
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 --std=c++17

all: bench

bench: device_radix_sort_bench.cu
	$(NVCC) $(NVCC_FLAGS) -o bench $<

run: bench
	./bench

clean:
	rm -f bench

.PHONY: all run clean
```

- [ ] **Step 3: Build and test CUDA reference**

Run: `cd examples/gpusorting-by-agent/cuda && make && ./bench 2>&1`
Expected: CUDA reference sorts correctly and prints throughput.

- [ ] **Step 4: Commit**

```bash
git add examples/gpusorting-by-agent/cuda/
git commit -m "feat(gpusorting): add CUDA reference benchmark"
```

---

### Task 9: Performance Benchmark Test

**Files:**
- Modify: `examples/gpusorting-by-agent/src/tests.rs` (add bench test)

- [ ] **Step 1: Add 16M performance benchmark test**

Add to `tests.rs`:

```rust
#[test]
fn bench_sort_16m() {
    let mut rng = rand::rng();
    let n = 16 * 1024 * 1024u32; // 16M elements
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();

    let thread_blocks = (n + PART_SIZE - 1) / PART_SIZE;
    let hist_size = (RADIX * thread_blocks) as usize;
    let mut result = input.clone();
    let mut alt = vec![0u32; n as usize];
    let mut global_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    let mut pass_hist = vec![0u32; hist_size];

    cuda_ctx(0, |ctx, m| {
        let mut d_sort = ctx.new_tensor_view(result.as_mut_slice()).expect("alloc sort");
        let mut d_alt = ctx.new_tensor_view(alt.as_mut_slice()).expect("alloc alt");
        let mut d_global = ctx.new_tensor_view(global_hist.as_mut_slice()).expect("alloc ghist");
        let mut d_pass = ctx.new_tensor_view(pass_hist.as_mut_slice()).expect("alloc phist");

        // Warmup
        dispatch_radix_sort(ctx, m, &mut d_sort, &mut d_alt, &mut d_global, &mut d_pass, n);

        // Reset
        d_sort.copy_from_host(&input).expect("reset");

        // Timed run
        let start = std::time::Instant::now();
        dispatch_radix_sort(ctx, m, &mut d_sort, &mut d_alt, &mut d_global, &mut d_pass, n);
        let elapsed = start.elapsed();

        d_sort.copy_to_host(&mut result).expect("copy back");

        let ms = elapsed.as_secs_f64() * 1000.0;
        let throughput = n as f64 / elapsed.as_secs_f64() / 1e6;
        println!("16M sort: {:.2} ms, {:.1} M keys/sec", ms, throughput);
    });

    assert!(is_sorted(&result), "16M bench: not sorted");
}
```

- [ ] **Step 2: Run benchmark**

Run: `cd examples && cargo test -p gpusorting-by-agent bench_sort_16m -- --nocapture --test-threads=1 2>&1`

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/src/tests.rs
git commit -m "feat(gpusorting): add 16M performance benchmark"
```

---

### Task 10: Documentation and Report

**Files:**
- Create: `examples/gpusorting-by-agent/README.md`
- Create: `examples/gpusorting-by-agent/REPORT.md`

- [ ] **Step 1: Write README.md**

Include: project overview, algorithm description, build instructions, test commands, CUDA reference comparison, project structure.

- [ ] **Step 2: Write REPORT.md**

Include: porting effort summary (time spent, LOC comparison CUDA vs Rust), kernel-by-kernel status, test results, performance comparison vs CUDA reference, key technical decisions (shuffle-based ranking vs ballot).

- [ ] **Step 3: Commit**

```bash
git add examples/gpusorting-by-agent/README.md examples/gpusorting-by-agent/REPORT.md
git commit -m "docs(gpusorting): add README and porting report"
```
