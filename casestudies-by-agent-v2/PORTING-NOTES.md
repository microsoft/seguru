# Porting to SeGuRu: working notes

These are the facts an agent needs to write SeGuRu GPU code in this workspace.
They were established empirically while porting the AES case study; treat
`aes/` as the worked reference example.

## Environment

`source /home/ziqiaozhou/seguru/casestudies-by-agent-v2/env.sh` before every
cargo invocation. It puts LLVM 20 (`mlir-opt` is required by the codegen
backend) and CUDA 13.3 on `PATH`. Without it the build panics with
`mlir-opt not found`.

```bash
source /home/ziqiaozhou/seguru/casestudies-by-agent-v2/env.sh
cd /home/ziqiaozhou/seguru/casestudies-by-agent-v2
cargo test  --release -p <crate> --lib
cargo build --release -p <crate>
```

`.cargo/config.toml` already selects `rustc-gpu` and sets `USE_FAST`,
`USE_FTZ`, `NVPTX_FEATURES`. The GPU is an NVIDIA A100 80GB PCIe.

## The two crates

* `gpu` — device side. Source of truth: `/home/ziqiaozhou/seguru/crates/gpu/src/`.
* `gpu_host` — host side (`cuda_ctx`, `new_tensor_view`, `gpu_config!`).
  Source: `/home/ziqiaozhou/seguru/crates/gpu-host/src/` and
  `/home/ziqiaozhou/seguru/crates/cuda_bindings/src/`.

Grep those directories for exact signatures rather than guessing.

## Kernel anatomy

```rust
use gpu::*;

#[gpu::cuda_kernel]
pub fn my_kernel(input: &[f32], output: &mut [f32], n: u32) {
    // `Config` is an implicit generic: Config::BDIM_X / BDIM_Y / BDIM_Z /
    // GDIM_* / SHARED_SIZE are compile-time constants when the launch config
    // marks them `@const`. Asserting on them is the idiomatic way to pin a
    // block shape.
    assert!(Config::BDIM_X == 256);
    ...
}
```

* `&mut [T]` parameters become `GpuGlobal<'_, [T]>`; `&[T]` stay plain slices.
* Launch: `my_kernel::launch(config, ctx, m, &d_in, &mut d_out, n)`.
* `#[gpu::device]` marks a helper function callable from kernels.

## Host side

```rust
gpu_host::cuda_ctx(0, |ctx, m| {
    let d_in  = ctx.new_tensor_view(host_slice).unwrap();
    let mut d_out = ctx.new_tensor_view(zeros.as_slice()).unwrap();
    let cfg = gpu_host::gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
    my_kernel::launch(cfg, ctx, m, &d_in, &mut d_out, n).unwrap();
    d_out.copy_to_host(&mut host_out).unwrap();
});
```

`gpu_config!(gx, gy, gz, bx, by, bz, shared_bytes)`; prefix any argument with
`@const` to make it a compile-time constant visible as `Config::*` in the
kernel. `ctx.sync()` blocks until the device is idle (use it for timing).

## Safe indexing: chunks and maps

Writing through `&mut [T]` requires proving each thread writes a disjoint set
of elements. That is what `chunk_mut` + a map does:

```rust
let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
let mut out = chunk_mut(output, reshape_map!([K] | [nthreads] => layout: [t0, i0]));
out[k] = value;   // k in 0..K, thread-local index
```

`reshape_map!([local dims] | [thread dims] => layout: [...])` is documented at
length in `crates/gpu/src/chunk_impl.rs` — read that doc comment. `layout`
lists dimensions low-to-high; `[t0, i0]` means `global = tid + i0 * nthreads`
(strided, coalesced), `[i0, t0]` means `global = i0 + tid * K` (contiguous per
thread). Simpler prebuilt maps: `MapLinear`, `MapContinuousLinear`, `Map2D`.

Reads through `&[T]` are ordinary bounds-checked slice indexing and are always
allowed.

## Shared memory

```rust
let mut smem = GpuShared::<[f32; 256]>::zero();
{
    let mut c = smem.chunk_mut(MapContinuousLinear::new(1));
    c[0] = src[thread_id::<DimX>() as usize];
}
sync_threads();
let s = &*smem;      // Deref gives &[f32; 256] for reads
let v = s[i];
```

Dynamic shared memory: declare the kernel `#[gpu::cuda_kernel(dynamic_shared)]`
and use the injected `smem_alloc.alloc::<T>(len)`, which yields
`&'static mut GpuShared<[T]>`.

### Hard-won rule

**`chunk_mut` (and any `sync_data` API) must not appear inside divergent
control flow.** This fails compilation with
`Invalid use of diversed data in GPU code` / `InvalidDiversedData`. Restructure
so every thread executes the `chunk_mut`, and make the *data* uniform instead:
e.g. pad the host buffer so every thread has a valid element to stage rather
than guarding the staging with `if tid < 44`.

## Performance rules

From `doc/optimization.md` plus what the AES port confirmed:

1. Use vector types (`U32_4`, `Float4`, `Float2`, `Float8`) for global loads
   and stores. One 128-bit access beats four 32-bit ones.
2. Prefer `u32`/`i32` over `usize`/`u64` — narrower types cut register pressure
   (~13% measured on matmul).
3. Unroll with `crunchy::unroll!` so LLVM's `mem2reg` can promote local arrays
   to registers. A dynamic loop over a local array leaves it in local memory,
   which is global memory in disguise and can cost 4× or more.
4. Give each thread several independent work items; the unroller interleaves
   them and hides shared-memory and global latency.
5. Avoid tail predicates by padding buffers on the host so the grid maps
   exactly onto the data.
6. `__constant__`-style broadcast has no analogue for lane-divergent lookups;
   stage divergent lookup tables in `GpuShared`. (In the AES study, the
   `__constant__` formulation was 44× slower than the shared-memory one.)

## Rules for this task

* **No `unsafe`** anywhere except in a `cuda_ffi` module whose sole purpose is
  binding a CUDA C++ reference for benchmarking. In particular do not use
  `slice::from_raw_parts` on the host to reinterpret buffers — build
  `Vec<U32_4>` with `U32_4::new([a, b, c, d])` instead.
* Every kernel must be covered by a test that checks it against an independent
  CPU implementation, and `cargo test --release -p <crate> --lib` must pass.
* Comment only what needs clarification.
