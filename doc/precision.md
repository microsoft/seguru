## Compiler-Induced Numerical Inconsistencies

LLVM compiler is known to have numerical inconsistancy issues if enabling full fast-math optimizations.

According to [Testing GPU Numerics](https://arxiv.org/pdf/2410.09172) and [Expression Isolation of Compiler-Induced Numerical Inconsistencies in Heterogeneous Code](https://web.cs.ucdavis.edu/~rubio/includes/isc23.pdf), the largest error in `clang -O3 -ffast-math` can be 402% larger than `nvcc -O0`, while `nvcc -O3 -ffast-math`'s error cost is 39%.

When running llm.rs with default fast-math optimizations, we observe random but frequent -inf results. This issue disappears when setting --nvptx-prec-divf32=1. Notably, -inf results in loss never appear in our unit tests, even with random inputs.

To address this without sacrificing performance, we updated fused_classifier_kernel3 from:

```rust
let prob = ((logits_block[ix as u32] - sp.offset).exp()) * sp.scale;
losses_chunk[0] = -prob.ln();
```

to

```rust
losses_chunk[0] = -(logits_block[ix as u32] - sp.offset) - sp.scale.ln()
```

This change avoids underflow because (logits_block[ix as u32] - sp.offset).exp() is more likely to become zero when using approximate math.

## Optimization flags that can cause inconsistant numerical results
llc's float optimization flags provides fine-grained control over floating-point behavior, balancing performance with numerical stability on GPU targets. By default, we enabled
```
--fp-contract=fast
--nvptx-prec-divf32=0
--nvptx-approx-log2f32
--nvptx-prec-sqrtf32=0
--nvptx-rsqrt-approx-opt
--denormal-fp-math-f32=preserve-sign
--denormal-fp-math=preserve-sign
```

To disable those fastmath, you can set env USE_FAST=false and USE_FTZ=false, or use `-C llvm-args=xxx` to override them.
See `crates/rustc_codegen_gpu/tests/float4.rs` and `crates/rustc_codegen_gpu/tests/float-fastmath.rs`

To gain maximum perfornance, you may pass more via `-C llvm-args=xxx`.

```
--enable-no-infs-fp-math
--enable-no-nans-fp-math
--enable-approx-func-fp-math
--enable-unsafe-fp-math
```

