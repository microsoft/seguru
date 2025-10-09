## Optimizing kernels

1. Use vectorized types ([f32; 4] or [f32; 8]) instead of scalar f32 when possible

Vectorized load/store can dramatically reduce instruction overhead. For example,
loading 4 floats individually requires 4 instructions, whereas loading them as
[f32; 4] requires only 1 instruction. This can improve memory throughput by
3–4×.

2. Prefer constant loops and minimize branching.

BBranch divergence is costly on GPUs. Loops with a known constant bound are
ideal because the compiler (e.g., NVCC) can safely unroll them, eliminating loop
overhead and enabling further optimizations.

3. use `crunchy::unroll` to enable `mem2reg` optimization.

NOTE: Local memory in GPU is just an abraction of global memory and so it would
be super slow.

LLVM’s `mem2reg` pass can promote local memory to registers only when memory
accesses are static and predictable. Dynamic loops may prevent this
optimization. To ensure registers are used instead of local memory:

Explicitly unroll loops using the crunchy::unroll macro.

Ensure all memory accesses are constant and predictable after unrolling.

In matmul_forward, failing to promote local memory to registers can result in 4× or higher

4. Prefer u32/i32 over usize/u64/i64 when possible.

Using wider types unnecessarily increases register pressure and memory usage. In
the matmul_forward example, replacing i32 with usize incurs a ~13% performance
penalty.