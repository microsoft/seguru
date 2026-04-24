# GPUSorting — SeGuRu Port Design Spec

## Overview

Port the DeviceRadixSort algorithm from [GPUSorting](https://github.com/b0nes164/GPUSorting) to SeGuRu Rust GPU kernels. Start with DeviceRadixSort (reduce-then-scan radix sort), then incrementally add OneSweep and SplitSort.

**Scope:** u32 keys only, correctness-first, with CUDA reference build for benchmarking.

## Source Reference

- Repository: https://github.com/b0nes164/GPUSorting
- Algorithm: DeviceRadixSort — a reduce-then-scan radix sort that processes 4 bits per pass (8 passes for 32-bit keys)
- CUDA files:
  - `GPUSortingCUDA/Sort/DeviceRadixSort.cu` (479 LOC, 4 global kernels)
  - `GPUSortingCUDA/Utils.cuh` (helper functions: warp scan, lane masking)
  - `GPUSortingCUDA/UtilityKernels.cuh` (init/validate utilities)

## Algorithm

DeviceRadixSort processes keys 4 bits at a time (radix-16) over 8 passes:

1. **Upsweep**: Each block computes a histogram of digit frequencies for its partition of the input
2. **Scan**: Global exclusive prefix sum across all block histograms to compute scatter offsets
3. **Downsweep**: Each block scatters its elements to the correct output positions using the prefix sums
4. **Init**: Zero-initializes the histogram buffer between passes

Each pass sorts on a different 4-bit digit, from LSB to MSB.

## Crate Structure

```
examples/gpusorting-by-agent/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Module declarations, constants, host orchestrator
│   ├── utils.rs            # Warp-level primitives (scan, ballot, popc, ffs)
│   ├── device_radix_sort.rs # 4 GPU kernels: init, upsweep, scan, downsweep
│   └── tests.rs            # Correctness + performance tests
├── cuda/
│   ├── Makefile
│   ├── device_radix_sort.cu # CUDA reference (extracted from GPUSorting)
│   └── bench_main.cu       # Benchmark driver
├── README.md
└── REPORT.md
```

## GPU Kernels

### 1. `init_radix_sort` kernel
- Zeroes the global histogram buffer
- Simple: one thread per element, `chunk_mut` with `MapLinear`

### 2. `upsweep` kernel (digit histogram)
- Each block processes `PART_SIZE` elements (256 threads × elements-per-thread)
- Extract 4-bit digit from each key: `(key >> shift) & 0xF`
- Build per-block histogram of 16 digit bins using shared memory
- Write block histogram to global `passHistogram[digit * numBlocks + blockIdx.x]`

**Shared memory:** 16 bins × u32, updated via `SharedAtomic` or local accumulation + warp reduce

### 3. `scan` kernel (exclusive prefix sum)
- Performs exclusive prefix sum on the flattened histogram array
- Uses Blelloch-style scan with warp-level shuffle primitives
- Input: `passHistogram[16 * numBlocks]`, output: same array with prefix sums

**Warp primitives needed:**
- `gpu::shuffle!(up, ...)` for inclusive warp scan
- `gpu::cg::ThreadWarpTile` for warp operations

### 4. `downsweep` kernel (scatter)
- Each block re-reads its partition of keys
- Computes local rank within the block for each key's digit
- Uses the global prefix sum + local rank to scatter keys to output
- Requires: digit extraction, local histogram scan, scatter write

**Scatter writes:** Use `chunk_mut` with `reshape_map!` for output positioning, or use `gpu::asm!` for raw store if `reshape_map!` cannot express arbitrary scatter.

## Warp Primitives — Shuffle-Based (No ballot/vote)

CUDA's `__ballot_sync`, `__popc`, `__ffs` are deprecated vote intrinsics. Instead, we use SeGuRu's existing shuffle and reduce primitives to build warp-level histograms and scans:

**Warp-level inclusive scan** (for prefix sums within a warp):
```rust
fn warp_inclusive_scan(val: u32) -> u32 {
    let mut x = val;
    for i in [1, 2, 4, 8, 16] {
        let (peer, _) = gpu::shuffle!(up, x, i, 32);
        if gpu::dim::lane_id() >= i { x += peer; }
    }
    x
}
```

**Warp-level reduce** (for histogram totals):
```rust
// Use gpu::cg::ThreadWarpTile subgroup_reduce for add/max/min
let warp = gpu::cg::ThreadWarpTile::<32>;
let total = warp.subgroup_reduce(gpu::cg::ReduxAdd, value);
```

**Per-digit ranking within a warp:**
- Each thread computes a one-hot vector: `1 << digit` for its digit
- Warp shuffle-scan accumulates counts per digit across lanes
- Local rank = scan result for this thread's digit - 1

**Cross-warp coordination:**
- `SharedAtomic` for accumulating warp histograms into block-level histogram
- `gpu::sync_threads()` barriers between phases

## Host-Side Orchestrator

```rust
pub fn device_radix_sort(ctx: &GpuContext, module: &Module, keys: &mut [u32]) {
    let n = keys.len();
    let num_blocks = (n + PART_SIZE - 1) / PART_SIZE;
    
    // Allocate: input, output (ping-pong), histogram
    let mut buf_a = ctx.new_tensor_view(keys)?;
    let mut buf_b = ctx.alloc::<u32>(n)?;
    let mut histogram = ctx.alloc::<u32>(16 * num_blocks)?;
    
    for pass in 0..8 {
        let shift = pass * 4;
        
        // 1. Init histogram to zero
        init_radix_sort::launch(config, ctx, module, &mut histogram)?;
        
        // 2. Upsweep: compute per-block digit histograms
        upsweep::launch(config, ctx, module, &buf_a, &mut histogram, shift, n)?;
        
        // 3. Scan: exclusive prefix sum of histogram
        scan::launch(config, ctx, module, &mut histogram, num_blocks)?;
        
        // 4. Downsweep: scatter to output
        downsweep::launch(config, ctx, module, &buf_a, &mut buf_b, &histogram, shift, n)?;
        
        // Ping-pong swap
        std::mem::swap(&mut buf_a, &mut buf_b);
    }
    
    // Copy result back if needed
}
```

## Constants

From the CUDA reference:
- `RADIX_BITS = 4` (16 bins per pass)
- `RADIX_MASK = 0xF`
- `NUM_PASSES = 8` (32-bit key / 4 bits per pass)
- `BLOCK_SIZE = 256` threads
- `KEYS_PER_THREAD = 15` (tunable)
- `PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD` = 3840

## Testing Strategy

### Correctness Tests (1M elements)
1. **Sorted output**: verify `output[i] <= output[i+1]` for all i
2. **Completeness**: verify output is a permutation of input (same histogram)
3. **Edge cases**: already sorted, reverse sorted, all same value, all zeros, single element
4. **Random**: multiple random seeds

### Performance Benchmark (16M elements)
1. Measure SeGuRu kernel execution time
2. Compare against CUDA reference build
3. Report throughput in M keys/sec

## CUDA Reference Build

Extract minimal CUDA code from GPUSorting repo:
- `DeviceRadixSort.cu` kernels
- `Utils.cuh` warp primitives
- Simple benchmark harness with timing
- Makefile with `nvcc` compilation

## Incremental Extensions (Future)

After DeviceRadixSort is working:
1. **OneSweep** — single-pass with decoupled lookback (4 kernels, requires atomic CAS)
2. **SplitSort** — segmented hybrid sort (12+ kernels, most complex)
3. **Key-value pairs** — add pair variants of each algorithm
4. **u64 keys** — template/generic support

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Shuffle-based ranking slower than ballot+popc | Acceptable for correctness-first; can optimize with native intrinsics later if SeGuRu adds them |
| Scatter writes via `chunk_mut` | If `reshape_map!` can't express arbitrary scatter, explore `unsafe` raw pointer writes or restructure algorithm |
| Shared memory atomic contention | Start with simple `SharedAtomic`, optimize with warp-level local accumulation if slow |
| Multi-kernel pass overhead | SeGuRu has ~1.4-1.9× launch overhead; acceptable for correctness-first goal |
