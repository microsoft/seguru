# Writing GPU kernels in SeGuRu

A practical guide for someone who knows CUDA. Every code excerpt below is taken
from working code in this workspace; the file it comes from is named next to it.
`PORTING-NOTES.md` is the condensed version of the same material — read that
first if you only want the rules.

Two crates matter:

* `gpu` — the device side (`/home/ziqiaozhou/seguru/crates/gpu/src/`).
* `gpu_host` — the host side (`/home/ziqiaozhou/seguru/crates/gpu-host/src/`).

When in doubt, grep those directories for the exact signature instead of
guessing.

---

## 1. Kernel anatomy

A kernel is a free function marked `#[gpu::cuda_kernel]`. Slice parameters are
the interface to device memory: `&[T]` is a read-only buffer, `&mut [T]` becomes
a `GpuGlobal<'_, [T]>` that can only be written through a *chunk* (section 3).

```rust
// gpusorting/src/clear.rs
#[gpu::cuda_kernel]
pub fn clear_u32(buf: &mut [u32]) {
    assert!(Config::BDIM_X == CLEAR_THREADS);
    ...
}
```

`Config` is an implicit generic parameter that the macro injects. It exposes the
launch geometry as associated constants — `Config::BDIM_X`, `BDIM_Y`, `BDIM_Z`,
`GDIM_*`, `SHARED_SIZE` — for every launch parameter the host marked `@const`.
`assert!(Config::BDIM_X == N)` is the idiomatic way to pin a block shape: it is a
compile-time check once `BDIM_X` is constant, and it lets the rest of the kernel
treat the block size as a literal.

Helper functions callable from a kernel are marked `#[gpu::device]`:

```rust
// gpusorting/src/utils.rs
#[gpu::device]
#[inline(always)]
pub fn lowest_set_bit(mask: u32) -> u32 {
    mask.trailing_zeros()
}
```

Dynamic shared memory is requested by the kernel attribute, which injects a
`smem_alloc` binding into the body:

```rust
// gpusorting/src/upsweep.rs
#[gpu::cuda_kernel(dynamic_shared)]
pub fn radix_upsweep(
    sort: &[U32_4],
    global_hist: &mut [u32],
    pass_hist: &mut [u32],
    radix_shift: u32,
    padded_thread_blocks: u32,
) {
    let smem = smem_alloc.alloc::<u32>((RADIX * SUB_HISTS) as usize);
```

Import either `gpu::*` (as `aes/src/lib.rs` does) or `gpu::prelude::*` (as every
`gpusorting` module does). The prelude gives you `thread_id`, `block_id`,
`grid_dim`, `block_dim`, `lane_id`, `sync_threads`, `ballot_sync`, `chunk_mut`,
`reshape_map!`, `GpuShared`, `MapLinear`, `MapContinuousLinear`, `Map2D`, and the
vector types `U32_4`, `Float2`, `Float4`, `Float8`.

---

## 2. The host side

```rust
// gpusorting/src/driver.rs (abridged)
let out = gpu_host::cuda_ctx(0, |ctx, m| {
    let mut d_a  = ctx.new_tensor_view::<[U32_4]>(&host_in).unwrap();
    let mut d_gh = ctx.new_tensor_view::<[u32]>(&zeros_gh).unwrap();

    let cfg = gpu_config!(clear_grid(gh_len), 1, 1, @const CLEAR_THREADS, 1, 1, 0);
    clear_u32::launch(cfg, ctx, m, &mut d_gh).unwrap();

    ctx.sync().unwrap();

    let mut host_out = vec![U32_4::default(); vec_len];
    d_a.copy_to_host(&mut host_out).unwrap();
    unpack(&host_out, n)
});
```

* `gpu_host::cuda_ctx(device, |ctx, m| { ... })` creates a context and a loaded
  module; the closure's return value is the return value of `cuda_ctx`.
* `ctx.new_tensor_view::<[T]>(host_slice)` allocates device memory and uploads.
* `<kernel>::launch(config, ctx, m, args...)` launches; the argument list mirrors
  the kernel signature, with `&`/`&mut` on the tensor views.
* `d.copy_to_host(&mut host_vec)` downloads.
* `ctx.sync()` blocks until the device is idle. Launches are asynchronous, so a
  timed region must be bracketed by `ctx.sync()` on both sides.
* `d.flatten()` reinterprets a `[U32_4]` view as a `[u32]` view, which is how
  `gpusorting` hands the same buffer to a vectorised upsweep and a scalar
  downsweep.

`gpu_config!(gx, gy, gz, bx, by, bz, shared_bytes)` builds the launch
configuration. Prefixing an argument with `@const` promotes it to a compile-time
constant that the kernel sees through `Config`. In practice the block dimensions
are always `@const` (the kernel asserts on them) and the grid dimension is not:

```rust
// gpusorting/src/driver.rs
let up_cfg   = gpu_config!(tb, 1, 1, @const UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
let scan_cfg = gpu_config!(RADIX, 1, 1, @const SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
```

The last argument is the dynamic shared-memory size **in bytes** — note the `* 4`
for `u32` elements.

---

## 3. Chunking: how SeGuRu proves writes are disjoint

This is the part with no CUDA analogue, and the part worth understanding first.

Reads through `&[T]` are ordinary bounds-checked slice indexing and are always
allowed. Writes are not: a `&mut [T]` parameter arrives as `GpuGlobal`, which has
no `IndexMut`. To write, you first carve the buffer into per-thread chunks:

```rust
let mut out = chunk_mut(buffer, map);
out[k] = value;      // k is a *thread-local* index
```

The `map` is a proof obligation discharged at compile time: it tells the
compiler how local index `k` on thread `t` becomes a global index, in a form it
can check is injective. Since the mapping is injective, no two threads can
address the same element, so the write is race-free without any runtime check.

### Prebuilt maps

* `MapContinuousLinear::new(w)` — thread `t` owns the contiguous run
  `[t*w, t*w + w)`.
* `MapLinear::new(w)` — thread `t` owns `w`-wide groups strided by the total
  thread count: `t*w + i%w + (i/w)*w*nthreads`. With `w == 1` this is the
  familiar coalesced `tid, tid + nthreads, tid + 2*nthreads, ...` pattern.

```rust
// gpusorting/src/upsweep.rs — zero both shared sub-histograms, coalesced
let mut z = smem.chunk_mut(MapLinear::new(1));
unroll! { for k in 0..4 { z[k] = 0u32; } }
```

### `reshape_map!`

Anything more structured uses the macro. Its authoritative documentation is the
doc comment on `reshape_map!` in
`/home/ziqiaozhou/seguru/crates/gpu/src/chunk_impl.rs`; read it before inventing
a new map. The shape is

```text
reshape_map!([local dims] | [thread dims] => layout: [permutation])
```

* Each dimension is either `D` or a tuple `(D, TD)`. `D` is the extent of the
  *index* (how many distinct values that dimension takes); `TD` is the extent of
  the corresponding *array* dimension. When `TD` is omitted it equals `D`.
  Accesses are valid for `0 <= id_k < min(D_k, TD_k)`.
* `layout` lists the dimensions **low to high**, i.e. innermost (unit-stride)
  first. `i0, i1, ...` name local dimensions, `t0, t1, ...` name thread
  dimensions; bare integers work too. A leading `-` reverses that dimension.
* The address is

  ```text
  global = sum_k  id[p_k] * product_{j<k} TD[p_j]
  ```

  In words: the first entry of `layout` has stride 1, the second has stride
  equal to the first entry's `TD`, the third has stride `TD[p_0] * TD[p_1]`, and
  so on.

Simplest case, from `aes/src/lib.rs`:

```rust
let nthreads = grid_dim::<DimX>() * Config::BDIM_X;
let mut out =
    chunk_mut(output, reshape_map!([BLOCKS_PER_THREAD] | [nthreads] => layout: [t0, i0]));
...
out[k as u32] = U32_4::new([...]);
```

One local dimension of extent 4, one (flattened) thread dimension of extent
`nthreads`, layout `[t0, i0]`: thread index has stride 1, local index has stride
`nthreads`. So `out[k]` is `output[gid + k * nthreads]` — the coalesced,
strided-by-grid pattern. Writing `layout: [i0, t0]` instead would give
`output[k + gid * 4]`, i.e. four contiguous elements per thread.

### Worked example 1 — a thread dimension whose array extent differs

```rust
// gpusorting/src/upsweep.rs
let mut ph = chunk_mut(
    pass_hist,
    reshape_map!([2u32] | [UPSWEEP_THREADS, (grid_dim::<DimX>(), padded_thread_blocks)]
                 => layout: [t1, t0, i0]),
);
ph[k as u32] = total;
```

`pass_hist` is logically `[RADIX][padded_thread_blocks]`: entry
`digit * padded_thread_blocks + block`. There are `UPSWEEP_THREADS == 128`
threads and `RADIX == 256` digits, so each thread owns two digits,
`tid` and `tid + 128` — that is the local dimension `[2]`.

Thread dimension `t0` is the 128 threads in a block; thread dimension `t1` is the
grid, written `(grid_dim::<DimX>(), padded_thread_blocks)`: the *index* ranges
over the launched blocks, but the *array* dimension is `padded_thread_blocks`,
which is larger (it is `thread_blocks` rounded up to a multiple of
`SCAN_THREADS`, so that `radix_scan` can chunk the same buffer evenly).

`layout: [t1, t0, i0]` then gives

```text
global = bid + tid * padded_thread_blocks + k * padded_thread_blocks * 128
       = (tid + k*128) * padded_thread_blocks + bid
       = digit * padded_thread_blocks + block
```

which is exactly the intended layout. Note that `t1` is **first**, so its `TD`
override does real work: it sets the stride of `t0`.

### Worked example 2 — and the bug to avoid

```rust
// gpusorting/src/scan.rs
let local = padded_thread_blocks / SCAN_THREADS;
let mut ph = chunk_mut(
    pass_hist,
    reshape_map!([local] | [SCAN_THREADS, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
);
```

`radix_scan` launches one block per digit and walks that digit's row of
`pass_hist`. With `layout: [t0, i0, t1]`:

```text
global = tid + i0 * SCAN_THREADS + bid * SCAN_THREADS * local
       = bid * padded_thread_blocks + i0 * SCAN_THREADS + tid
```

because `SCAN_THREADS * local == padded_thread_blocks` by construction. The row
stride is produced by the *preceding* dimensions' `TD`s, not by anything written
on `t1`.

**The trap:** it is tempting to write the block dimension here as
`(grid_dim::<DimX>(), padded_thread_blocks)`, by analogy with the upsweep, to
"say" that the row stride is `padded_thread_blocks`. That is wrong. Per the
address formula, the *last* dimension in `layout` contributes
`id * product_{j<k} TD[p_j]`; its own `TD` never enters the stride and only
bounds the index. The stride silently stays whatever the preceding `TD`s
multiply out to, addresses run off the end of the buffer, and the program dies at
run time with `CUDA_ERROR_ILLEGAL_ADDRESS` — with no compile-time complaint,
because the map is still injective on its declared domain.

Rule of thumb: a `(D, TD)` override only affects addressing of the dimensions
that come **after** it in `layout`. If you need a dimension to have a specific
stride, either place it first (upsweep) or make the product of the preceding
`TD`s equal to the stride you want (scan).

### Worked example 3 — grid-strided clear

```rust
// gpusorting/src/clear.rs
let mut c = chunk_mut(
    buf,
    reshape_map!([CLEAR_PER_THREAD] | [CLEAR_THREADS, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
);
unroll! { for k in 0..4 { c[k as u32] = 0u32; } }
```

`global = tid + k * 256 + bid * 1024`: each block owns a contiguous 1024-element
tile, and within it the four writes of a thread are strided by the block size, so
every warp write is one 128-byte transaction.

---

## 4. Shared memory

Two flavours. Statically sized:

```rust
// aes/src/lib.rs
let mut te_smem = GpuShared::<[u32; BLOCK_DIM as usize]>::zero();
{
    let mut te_chunk = te_smem.chunk_mut(MapContinuousLinear::new(1));
    te_chunk[0] = te0[tid as usize];
}
sync_threads();
let te = &*te_smem;          // Deref gives &[u32; 256] for reads
let v = te[a as usize];
```

and dynamically sized, which requires `#[gpu::cuda_kernel(dynamic_shared)]` and
the matching byte count in `gpu_config!`:

```rust
// gpusorting/src/downsweep.rs
let smem = smem_alloc.alloc::<u32>((BIN_PART_SIZE + RADIX) as usize);
...
let key = *smem[i as usize];   // read
```

Reading is `*smem[i]` (index, then deref). Writing needs one of two things:

* If the index is **thread-structured** — derivable from the thread id and a
  constant local index — use `chunk_mut` with a map, as above. This is free.
* If the index is **data dependent** — computed from the values being processed —
  no map can prove disjointness, so the write must go through an atomic (next
  section).

Note that a `GpuShared` buffer may be reused for different purposes across
`sync_threads()` boundaries; `gpusorting/src/downsweep.rs` uses one allocation
first as sixteen per-warp 256-bin histograms and then as the 7680-element local
scatter buffer.

---

## 5. Data-dependent (scatter) writes

`gpu::sync::Atomic` wraps a global `&mut [T]`; `gpu::sync::SharedAtomic` wraps a
`GpuShared`. Both give `.index(i)` and then a read-modify-write method —
`atomic_assign`, `atomic_addi`, `atomic_addf`, `atomic_maxu`, `atomic_minu`, and
so on (the full list is generated by `def_atomic_rmw_kinds!` in
`crates/gpu/src/sync.rs`).

```rust
// gpusorting/src/upsweep.rs — histogram, index depends on the key
let hist = gpu::sync::SharedAtomic::new(&mut *smem);
hist.index((wave + d) as usize).atomic_addi(1u32);

// gpusorting/src/downsweep.rs — final scatter, index depends on the rank
let out = gpu::sync::Atomic::new(alt);
out.index((base + i) as usize).atomic_assign(key);
```

`atomic_assign` is a plain store as far as the algorithm is concerned; it is
atomic only so that the compiler need not prove uniqueness. It is not free, but
it is cheap: inspecting the generated PTX for `gpusorting` shows the shared
writes lowering to `atom.shared.exch.b32`, the global scatter to
`atom.global.exch.b32`, and the histogram increments to `atom.shared.add.u32` /
`atom.global.add.u32` — the same instructions the hand-written CUDA emits for a
scatter, except the exchange where CUDA would use `st`.

The cost of that difference is why it is worth restructuring an algorithm to be
thread-structured when you can. `gpusorting/src/scan.rs` is the example: the CUDA
original converts an inclusive scan into an exclusive one by writing to a
circularly lane-shifted address, which is a scatter; the SeGuRu version keeps
each thread writing its own slot through `chunk_mut` and recovers the exclusive
value with one extra `shfl.up`.

---

## 6. The divergence rule

**`chunk_mut`, and any API annotated `sync_data` in the `gpu` crate, must not
appear inside divergent control flow.** Violating it is a compile error, not a
run-time surprise:

```text
Invalid use of diversed data in GPU code
InvalidDiversedData
Failed at stage GpuForGpu
```

The reason is that a chunk's validity argument is about *all* threads of the
scope; a chunk created by only some of them is meaningless. The fix is always
the same shape: hoist the `chunk_mut` so every thread executes it, and move the
non-uniformity into the data.

**AES.** The kernel stages the 44-word round-key schedule into shared memory. The
natural CUDA spelling is

```cuda
if (tid < 44) rk[tid] = round_keys[tid];   // rejected in SeGuRu
```

Instead the host pads the schedule to `BLOCK_DIM` words and every thread stages
exactly one word, unconditionally:

```rust
// aes/src/lib.rs
pub fn staged_round_keys(rk: &[u32; 44]) -> Vec<u32> {
    let mut v = vec![0u32; BLOCK_DIM as usize];
    v[..44].copy_from_slice(rk);
    v
}
```

```rust
// aes/src/lib.rs — no predicate in the kernel
let mut rk_chunk = rk_smem.chunk_mut(MapContinuousLinear::new(1));
rk_chunk[0] = round_keys[tid as usize];
```

**gpusorting.** The same rule bites on inter-warp scans. The CUDA original scans
four or eight warp totals under `if (tid < 8)`, which splits warp 0. The SeGuRu
version widens the guard to a whole warp and feeds the identity element to the
lanes that have no real work:

```rust
// gpusorting/src/upsweep.rs
if tid < 32 {
    let groups = RADIX >> LANE_LOG;   // 8
    let v = if tid < groups { *smem[(tid << LANE_LOG) as usize] } else { 0u32 };
    let s = crate::utils::exclusive_warp_scan(v);
    if tid < groups {
        let w = gpu::sync::SharedAtomic::new(&mut *smem);
        w.index((tid << LANE_LOG) as usize).atomic_assign(s);
    }
}
```

This is good practice in CUDA anyway — `__shfl_sync` on a partial warp is
fragile — and here it is mandatory. The same idiom appears in
`gpusorting/src/scan.rs` (`if tid < 32` around a scan of `WARPS == 4` totals) and
`gpusorting/src/downsweep.rs` (`if tid < RADIX`, which is eight whole warps).
`gpusorting/src/utils.rs` documents the convention at module level: warp
collectives are written assuming whole warps participate.

---

## 7. Warp primitives

| SeGuRu | CUDA |
|---|---|
| `lane_id()` | `threadIdx.x & 31` |
| `sync_threads()` | `__syncthreads()` |
| `ballot_sync(mask, pred)` | `__ballot_sync(mask, pred)` |
| `gpu::shuffle!(up\|down\|xor\|idx, val, arg, width)` | `__shfl_{up,down,xor,}_sync` |
| `x.count_ones()` | `__popc(x)` |
| `x.trailing_zeros()` | `__ffs(x) - 1` |

`gpu::shuffle!` returns a `(value, predicate)` pair; the second element is the
"source lane was valid" flag, and most call sites discard it:

```rust
// gpusorting/src/utils.rs — Hillis-Steele warp scan
let mut x = val;
let lane = lane_id();
unroll! {
    for k in 0..5 {
        let delta: u32 = 1 << k;
        let (t, _) = gpu::shuffle!(up, x, delta, 32);
        if lane >= delta { x += t; }
    }
}
```

`gpusorting/src/utils.rs` is a small library of these: `inclusive_warp_scan`,
`exclusive_warp_scan`, `inclusive_warp_scan_circular_shift` (an inclusive scan
rotated left by one lane, so a single register carries both the exclusive prefix
and the warp total), `lane_mask_lt` (CUDA's `getLaneMaskLt()`), and
`lowest_set_bit`. Reuse them rather than re-deriving them.

The warp-level multi-split in `gpusorting/src/downsweep.rs` shows the ballot
idiom in full: eight `ballot_sync` calls peel the digit bit by bit until `flags`
holds exactly the lanes whose key shares this key's digit; `(flags & lt).count_ones()`
is the rank within the match group; the lowest matching lane reserves space for
the whole group with one shared atomic and broadcasts the reservation with
`shuffle!(idx, ...)`.

---

## 8. Performance rules

The general rules are in `/home/ziqiaozhou/seguru/doc/optimization.md`. The five
that mattered most in these ports:

1. **Use vector types.** `U32_4`, `Float2`, `Float4`, `Float8`. One 128-bit
   access beats four 32-bit ones. `aes` types its plaintext and ciphertext as
   `[U32_4]` so a 16-byte AES block moves in one instruction; `gpusorting`'s
   upsweep loads keys as `[U32_4]`. Build them without `unsafe` on the host with
   `U32_4::new([a, b, c, d])` (see `pack_padded` in `gpusorting/src/lib.rs`),
   never by reinterpreting a pointer.

2. **Unroll every per-thread loop with `crunchy::unroll!`.** Local arrays are
   promoted to registers by `mem2reg` only if all their accesses have constant
   indices after unrolling. With a runtime index they stay in local memory, which
   on a GPU is global memory in disguise; the cost is a factor of several.
   `gpusorting/src/downsweep.rs` keeps a `[u32; 15]` key array and a `[u32; 15]`
   offset array in registers this way, and `aes/src/lib.rs` keeps
   `[[u32; 4]; 4]` of AES state.

3. **Prefer `u32`/`i32` to `usize`/`u64`.** Wider types cost registers; the
   `matmul_forward` measurement in `doc/optimization.md` puts it at about 13%.
   Both reference crates carry `u32` throughout and only cast at the point of
   indexing.

4. **Give each thread several independent work items.** The unroller interleaves
   them, which hides shared-memory and global latency without needing more
   occupancy. `aes` encrypts `BLOCKS_PER_THREAD == 4` blocks per thread;
   `gpusorting` ranks 15 keys per thread.

5. **Pad on the host so the kernel needs no tail predicate.** `aes` pads device
   buffers to a whole number of `grid * BLOCK_DIM * BLOCKS_PER_THREAD` AES
   blocks; `gpusorting` pads the key array to a whole number of `PART_SIZE`
   partitions with `u32::MAX`, which sorts to the end and is dropped afterwards.
   This buys branch-free kernels, full vector loads, and — as section 6 explains
   — it is often what makes the `chunk_mut` legal in the first place.

A sixth, specific to lookup tables: there is no useful SeGuRu analogue of CUDA's
`__constant__` for *lane-divergent* lookups, and you do not want one. Constant
memory broadcasts one address per warp per cycle; a divergent 256-entry table
lookup serialises into up to 32 transactions. Stage such tables in `GpuShared`.
In the AES study the `__constant__` formulation measured about 44x slower than
the shared-memory one.

---

## 9. Gotchas

* **`crunchy::unroll!` needs literal bounds.** `unroll! { for k in 0..15 { .. } }`
  works; `0..KPT` with a `const` bound does not. Where the constant matters,
  assert it separately — `gpusorting/src/upsweep.rs` ends with
  `const _: () = assert!(RADIX_OVER_THREADS == 2);` next to its `for k in 0..2`,
  and `gpusorting/src/downsweep.rs` guards a partly-unused unroll with
  `if k < HIST_CLEAR_PER_THREAD` inside `for k in 0..8`.

* **Import `unroll` by name.** Every crate here writes `use crunchy::unroll;` and
  calls `unroll! { ... }`. The fully-qualified `crunchy::unroll! { ... }` path
  form was not usable in these crates.

* **The loop variable is not automatically the right type.** Inside
  `unroll!`, `k` is a `usize` literal; cast it where a `u32` is wanted, as in
  `let i = tid + (k as u32) * UPSWEEP_THREADS;`.

* **`use gpu::*` shadows `std::println!`.** The `gpu` crate exports a
  `#[macro_export] macro_rules! println` (device-side `printf`) at its crate
  root, so a glob import of `gpu` brings it into scope and it wins over the
  prelude's `std::println`. In a `#[cfg(test)]` module of such a crate, write
  `std::println!`. Importing `gpu::prelude::*` instead does not have this
  problem: the prelude does not re-export `println`.

* **Shared-memory sizes in `gpu_config!` are bytes.** `SCAN_THREADS * 4` for
  `SCAN_THREADS` `u32`s. Getting this wrong gives a launch failure, not a
  compile error.

* **`ctx.sync()` before and after any timed region.** Launches are asynchronous.

---

## 10. Known toolchain gaps

The backend is not complete, and hitting a gap looks like a codegen panic rather
than a type error. The one encountered in this round:

* **`ctlz` / `cttz` (`leading_zeros`, `trailing_zeros`) were unsupported.** Any
  use panicked with ``GPU intrinsic `cttz` not supported``, which is why the
  previous generation's sort emulated `__ffs` with a 32-iteration serial loop
  executed once per key. The fix was to add them to the `intrinsic_match!` table
  in `/home/ziqiaozhou/seguru/crates/rustc_codegen_gpu/src/builder/intrinsic.rs`,
  next to the existing `sym::ctpop` entry:

  ```rust
  sym::ctpop        => melior_math::ctpop, 1,
  sym::ctlz         => melior_math::ctlz,  1,
  sym::ctlz_nonzero => melior_math::ctlz,  1,
  sym::cttz         => melior_math::cttz,  1,
  sym::cttz_nonzero => melior_math::cttz,  1,
  ```

  Note that Rust lowers `x.trailing_zeros()` to `cttz` and, in some contexts,
  `cttz_nonzero`; both spellings need an entry.

That table is the general recipe: find the `sym::` name of the missing intrinsic,
map it to the corresponding `melior_math` builder, and give its arity. After
editing the backend you must reinstall the driver, or cargo will keep using the
old binary:

```bash
cd /home/ziqiaozhou/seguru
MLIR_SYS_200_PREFIX=/usr/lib/llvm-20 TABLEGEN_200_PREFIX=/usr/lib/llvm-20 \
    cargo install --path ./crates/rustc-gpu --locked
```

Both variables are required; without them `melior-macro` caches an empty include
directory and the build fails with `could not find llvm in ods`.
