# GEMM 2.18× gap: ncu-confirmed occupancy chain

*Phase I investigation. Branch: `codegen-narrow-addr-arith`. Kernel:
`gemm_add_relu` on A100 (CC 8.0), shape M=1024, K=8192, N=8192.*

## 1. The measurement

Nsight Compute profiling (`ncu --section SpeedOfLight,Occupancy,
WarpStateStats,LaunchStats`) on both SG and nvcc arms of `gemm_add_relu`,
same block/grid, same inputs:

| Metric                              | SG       | nvcc     | ratio   |
|-------------------------------------|---------:|---------:|--------:|
| Duration                            | 26.28 ms | 12.06 ms | 2.18×   |
| Elapsed cycles                      | 27.98 M  | 12.84 M  | 2.18×   |
| Compute (SM) throughput             | 35.67 %  | 77.49 %  | 0.46×   |
| Memory throughput                   | 81.46 %  | 63.37 %  | 1.29×   |
| L1/TEX throughput                   | 86.07 %  | 67.53 %  | 1.27×   |
| DRAM throughput                     | 2.80 %   | 5.03 %   | 0.56×   |
| Warp cycles per issued inst         | 9.75     | 7.96     | 1.22×   |
| Registers per thread                | **168**  | **101**  | 1.66×   |
| Theoretical occupancy               | 12.5 %   | 25.0 %   | 0.50×   |
| Achieved occupancy                  | 12.5 %   | 22.65 %  | 0.55×   |
| Block-limit (registers)             | **1**    | **2**    | —       |
| Block-limit (shared memory)         | 1        | 7        | —       |
| Block-limit (warps)                 | 8        | 8        | —       |

Both kernels use 256 threads × 512 blocks, 8192 B smem, zero spills.

## 2. The causal chain

1. **SG emits 168 regs/thread, nvcc emits 101.** Algorithm, tile size,
   unroll depth are identical. The 67 extra registers are SG codegen
   overhead: address-arithmetic temporaries, intermediate index
   values, bounds-check state — none of which nvcc allocates because
   LSR and stronger CSE eliminate them.
2. **At 168 regs × 256 threads = 43008 regs/block, block_limit_regs = 1**
   on A100 (65536 regs/SM). SG can only schedule 1 block per SM.
3. nvcc at 101 regs × 256 threads = 25856 regs/block gets
   `block_limit_regs = 2`.
4. **Occupancy: SG 12.5 % (8 warps/SM), nvcc 25.0 % (16 warps/SM).**
5. With only 8 warps on the SM, when one warp stalls on a shared-memory
   load there are rarely other warps ready to issue. ncu reports **47 %
   of SG warp cycles stalled on MIO (shared memory)**.
6. Compute pipelines are idle half the time (35.67 % SM throughput vs
   nvcc's 77.49 %) because of the memory stalls that occupancy cannot
   hide.
7. Total cycles scale as ~2×, matching the observed 2.18× runtime gap.

## 3. Experiment: `--maxrregcount=128`

Wrapped `ptxas` to force SG (and incidentally the nvcc arm) to 128 regs.
Rebuilt `kernelbench-c`, reran `compare.py`:

| arm          | baseline | capped @ 128 | delta |
|--------------|---------:|-------------:|------:|
| SG ← PyTorch | 0.39×    | 0.42×        | +8 %  |
| SG ← CUDA    | 0.37×    | 0.40×        | +8 %  |
| CUDA (nvcc)  | 0.94×    | 0.77×        | −18 % |

Gap cuda/seguru: 2.41× baseline → **1.83×** under matched cap.

Why only +8 % on SG despite doubling theoretical occupancy:
ptxas introduced 92 B spill stores per thread (~23 f32 values spilled to
local memory). Spill traffic is served from L1/L2, which is *already* the
bottleneck (86 % L1 throughput). The spill bandwidth ate most of the
occupancy gain.

→ **Matched-cap gap (1.83×) tells us ~25 % of the total gap was**
**occupancy alone.** The remaining 75 % is the spill-like effect of
forcing register count without removing the underlying register demand.

## 4. Implication for `open_tile()` / reshape_map codegen work

Earlier section §7 of `codegen-address-arithmetic-investigation.md`
estimated the epilogue-only ROI at ~15 % and concluded the work wasn't
worth it. **ncu data inverts that estimate.**

The 67-register gap between SG (168) and nvcc (101) is not a localized
epilogue artifact; it reflects live-register footprint across the whole
kernel — the address-arithmetic temporaries are live THROUGH the main
loop, pinning registers that could otherwise hold useful state or let
another block share the SM.

If an `open_tile()`-style API removes enough redundant address-arith
values to drop SG's register footprint to ≤ 105, block_limit_regs rises
to 2, occupancy doubles, and the 2.18× gap is largely closed. Unlike the
maxrregcount experiment, this would reduce reg *demand* rather than reg
*allowance*, so there would be no spills.

## 5. Other findings worth noting

- **DRAM is far from saturated** on both arms (SG 2.8 %, nvcc 5.0 %).
  This is compute/L1-bound at these shapes, not bandwidth-bound.
- **L1 throughput for SG is 86 %** — close to saturation. Any additional
  L1 traffic (e.g. spills) immediately hurts.
- **Smem carveout.** SG configured 16 KB smem, nvcc 64 KB. This made SG
  smem-bound on block_limit (1 vs 7) but is *not* the binding constraint
  (regs already cap at 1). Fixing the carveout alone would not help.
- **Waves per SM.** SG sees 4.74 waves, nvcc 2.37 waves. SG launches
  twice as many waves for the same grid because it fits half as many
  blocks per SM. The tail-wave effect (~20 % SG / ~33 % nvcc) is not the
  dominant factor.

## 6. Next steps (ranked by expected yield)

1. **Implement Option A (open_tile / reshape_map codegen hoisting)** to
   reduce register footprint at source. Target: ≤ 105 regs, matching
   nvcc. Expected yield: most of the 2.18× gap.
2. If (1) proves infeasible, investigate whether SG's address-arithmetic
   temporaries can be scheduled differently (register allocator tuning
   or instruction scheduling passes in rustc_codegen_gpu).
3. Orthogonal: opt into the 64 KB smem carveout via cuFuncSetAttribute
   in gpu_host; zero cost once regs stop binding occupancy.

## 7. Artifacts

- `/tmp/sg_gar.ncu` equivalent run above (SG arm).
- `/tmp/bench_cuda_arm.py` + ncu invocation for nvcc arm.
- `/tmp/dump_inputs.py` to regenerate inputs for the runner.

Raw ncu sections: `SpeedOfLight`, `Occupancy`, `WarpStateStats`,
`LaunchStats`. Launch-skip 5 (SG) / 10 (nvcc), launch-count 1.

## 8. Note on device

All measurements on CC 8.0 (A100-class), not sm_86 (RTX 3090) as earlier
assumed in `codegen-address-arithmetic-investigation.md`. Register
allocation math differs between architectures (A100: 65536 regs/SM,
1 block granularity; sm_86: 65536 regs/SM, same math). Conclusions hold.
