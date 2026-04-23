# PolybenchGPU RED-Tier Batch 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 4 RED-tier PolybenchGPU CUDA benchmarks (CORR, GRAMSCHM, DOITGEN, LU) to SeGuRu.

**Architecture:** Each benchmark is a separate crate under `examples/polybench/<name>/` with kernels in `src/lib.rs` and tests comparing GPU output against CPU reference. Complex patterns include multi-kernel pipelines with host loops, sqrt intrinsics, aliased read/write via separate allocations, and 3D-to-2D index packing.

**Tech Stack:** SeGuRu (`gpu` + `gpu_host` crates), CUDA 13.2, LLVM 20

**Environment setup (required for all build/test commands):**
```bash
export PATH=/usr/local/cuda-13.2/bin:/usr/lib/llvm-20/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.2
```

---

## Key SeGuRu Patterns

1. **chunk_mut uses LOCAL indices** — `c[0]` not `c[global_idx]`
2. **Kernel function name must NOT match crate name**
3. **Tests MUST use helper functions** — don't put `cuda_ctx` directly in `#[test]`
4. **For 2D output writes**: `chunk_mut(c, Map2D::new(width))` with `c[(0,0)]`
5. **For 1D output writes**: `chunk_mut(c, MapLinear::new(1))` with `c[0]`
6. **Read-write arrays**: use chunk_mut, read via `c[(0,0)]`, modify, write back
7. **sqrt()**: imported via `use gpu::prelude::*` — call as `val.sqrt()`
8. **Aliased read/write**: Create two separate GPU allocations from same host data (Rust borrow checker prevents `&d_a` and `&mut d_a` simultaneously)
9. **TensorViewMut derefs to TensorView** — can pass `&d_a` where `&[f32]` kernel param expected

## File Structure

```
examples/polybench/corr/Cargo.toml          — crate polybench-corr
examples/polybench/corr/src/lib.rs          — 4 kernels + test
examples/polybench/gramschm/Cargo.toml      — crate polybench-gramschm
examples/polybench/gramschm/src/lib.rs      — 4 kernels + test (host k-loop, CPU kernel1)
examples/polybench/doitgen/Cargo.toml       — crate polybench-doitgen
examples/polybench/doitgen/src/lib.rs       — 2 kernels + test (3D packed into 2D grid)
examples/polybench/lu/Cargo.toml            — crate polybench-lu
examples/polybench/lu/src/lib.rs            — 2 kernels + test (aliased read/write)
examples/Cargo.toml                         — add 4 new members
```

---

### Task 1: CORR — 4-Kernel Correlation Pipeline

**Files:**
- Create: `examples/polybench/corr/Cargo.toml`
- Create: `examples/polybench/corr/src/lib.rs`
- Modify: `examples/Cargo.toml` (add member)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "polybench-corr"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Create src/lib.rs with 4 kernels**

```rust
use gpu::prelude::*;

const FLOAT_N: f32 = 3214212.01;
const EPS: f32 = 0.005;

// mean[j] = sum_i(data[i*m+j]) / FLOAT_N
#[gpu::cuda_kernel]
pub fn corr_mean_kernel(data: &[f32], mean: &mut [f32], m: usize, n: usize) {
    let mut mean = chunk_mut(mean, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j];
            i += 1;
        }
        mean[0] = sum / FLOAT_N;
    }
}

// stddev[j] = sqrt(sum_i((data[i*m+j]-mean[j])^2) / FLOAT_N); clamp to 1.0 if <= EPS
#[gpu::cuda_kernel]
pub fn corr_std_kernel(
    data: &[f32],
    mean: &[f32],
    stddev: &mut [f32],
    m: usize,
    n: usize,
) {
    let mut stddev = chunk_mut(stddev, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j < m {
        let mut sum = 0.0f32;
        let mean_j = mean[j];
        let mut i: usize = 0;
        while i < n {
            let diff = data[i * m + j] - mean_j;
            sum += diff * diff;
            i += 1;
        }
        sum /= FLOAT_N;
        let s = sum.sqrt();
        stddev[0] = if s <= EPS { 1.0 } else { s };
    }
}

// data[i*m+j] = (data[i*m+j] - mean[j]) / (sqrt(FLOAT_N) * stddev[j])
#[gpu::cuda_kernel]
pub fn corr_reduce_kernel(
    mean: &[f32],
    stddev: &[f32],
    data: &mut [f32],
    m: usize,
    n: usize,
) {
    let mut data = chunk_mut(data, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < n && j < m {
        let val = data[0] - mean[j];
        data[0] = val / (FLOAT_N.sqrt() * stddev[j]);
    }
}

// symmat[j1*m+j2] = sum_i(data[i*m+j1] * data[i*m+j2])
#[gpu::cuda_kernel]
pub fn corr_corr_kernel(
    data: &[f32],
    symmat: &mut [f32],
    m: usize,
    n: usize,
) {
    let mut symmat = chunk_mut(symmat, Map2D::new(m));
    let j2 = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let j1 = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j1 < m && j2 < m {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < n {
            sum += data[i * m + j1] * data[i * m + j2];
            i += 1;
        }
        symmat[(0, 0)] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_corr(m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_data: Vec<f32> = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                h_data[i * m + j] = (i * j) as f32 / m as f32;
            }
        }
        let h_data_orig = h_data.clone();
        let mut h_data_gpu = h_data.clone();
        let mut h_mean_gpu: Vec<f32> = vec![0.0; m];
        let mut h_stddev_gpu: Vec<f32> = vec![0.0; m];
        let mut h_symmat_gpu: Vec<f32> = vec![0.0; m * m];

        // CPU reference
        let mut h_mean_cpu: Vec<f32> = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += h_data_orig[i * m + j];
            }
            h_mean_cpu[j] = sum / FLOAT_N;
        }
        let mut h_stddev_cpu: Vec<f32> = vec![0.0; m];
        for j in 0..m {
            let mut sum = 0.0f32;
            for i in 0..n {
                let diff = h_data_orig[i * m + j] - h_mean_cpu[j];
                sum += diff * diff;
            }
            sum /= FLOAT_N;
            let s = sum.sqrt();
            h_stddev_cpu[j] = if s <= EPS { 1.0 } else { s };
        }
        let mut h_data_cpu = h_data_orig.clone();
        let sqrt_fn = FLOAT_N.sqrt();
        for i in 0..n {
            for j in 0..m {
                h_data_cpu[i * m + j] =
                    (h_data_cpu[i * m + j] - h_mean_cpu[j]) / (sqrt_fn * h_stddev_cpu[j]);
            }
        }
        let mut h_symmat_cpu: Vec<f32> = vec![0.0; m * m];
        for j1 in 0..m {
            for j2 in 0..m {
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += h_data_cpu[i * m + j1] * h_data_cpu[i * m + j2];
                }
                h_symmat_cpu[j1 * m + j2] = sum;
            }
        }

        cuda_ctx(0, |ctx, m_module| {
            let d_data_ro =
                ctx.new_tensor_view(h_data.as_slice()).expect("alloc data_ro");
            let mut d_data = ctx
                .new_tensor_view(h_data_gpu.as_mut_slice())
                .expect("alloc data");
            let mut d_mean = ctx
                .new_tensor_view(h_mean_gpu.as_mut_slice())
                .expect("alloc mean");
            let mut d_stddev = ctx
                .new_tensor_view(h_stddev_gpu.as_mut_slice())
                .expect("alloc stddev");
            let mut d_symmat = ctx
                .new_tensor_view(h_symmat_gpu.as_mut_slice())
                .expect("alloc symmat");

            let block_size: u32 = 16;

            // kernel1: mean
            let grid_x = (m as u32 + block_size - 1) / block_size;
            let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
            corr_mean_kernel::launch(
                config, ctx, m_module, &d_data_ro, &mut d_mean, m, n,
            )
            .expect("mean kernel failed");

            // kernel2: stddev
            let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
            corr_std_kernel::launch(
                config, ctx, m_module, &d_data_ro, &d_mean, &mut d_stddev, m, n,
            )
            .expect("std kernel failed");

            // kernel3: reduce
            let grid_y = (n as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            corr_reduce_kernel::launch(
                config, ctx, m_module, &d_mean, &d_stddev, &mut d_data, m, n,
            )
            .expect("reduce kernel failed");

            // kernel4: correlation
            let grid_y = (m as u32 + block_size - 1) / block_size;
            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            corr_corr_kernel::launch(
                config, ctx, m_module, &d_data, &mut d_symmat, m, n,
            )
            .expect("corr kernel failed");

            d_symmat
                .copy_to_host(&mut h_symmat_gpu)
                .expect("copy failed");
        });

        (h_symmat_gpu, h_symmat_cpu)
    }

    #[test]
    fn test_corr() {
        let m = 32;
        let n = 64;
        let (gpu, cpu) = run_corr(m, n);

        for j1 in 0..m {
            for j2 in 0..m {
                assert!(
                    (gpu[j1 * m + j2] - gpu[j2 * m + j1]).abs() < 1e-1,
                    "Not symmetric at ({},{}): {} vs {}",
                    j1, j2, gpu[j1 * m + j2], gpu[j2 * m + j1],
                );
            }
        }

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "symmat is all zeros");
    }
}
```

- [ ] **Step 3: Add to workspace**

Add `"polybench/corr"` to the members list in `examples/Cargo.toml`.

- [ ] **Step 4: Build and test**

```bash
cd /home/sanghle/work/seguru/examples && cargo test -p polybench-corr -- --test-threads=1
```

Expected: test_corr PASS

- [ ] **Step 5: Commit**

```bash
git add examples/polybench/corr/ examples/Cargo.toml
git commit -m "examples: port CORR benchmark from PolybenchGPU"
```

---

### Task 2: GRAMSCHM — 4-Kernel Gram-Schmidt with Host Loop

**Files:**
- Create: `examples/polybench/gramschm/Cargo.toml`
- Create: `examples/polybench/gramschm/src/lib.rs`
- Modify: `examples/Cargo.toml` (add member)

**Design decisions:**
- kernel1 (norm computation) is single-threaded in CUDA — compute on CPU instead (no parallelism benefit, avoids chunk_mut indexing issue)
- kernel3 split into kernel3a (dot product → r) and kernel3b (update a) for chunk compatibility
- Host k-loop iterates 0..nj, launching kernels per iteration

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "polybench-gramschm"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Create src/lib.rs with 3 GPU kernels + CPU kernel1**

```rust
use gpu::prelude::*;

// kernel2: q[i*nj+k] = a[i*nj+k] / r[k*nj+k]
#[gpu::cuda_kernel]
pub fn gramschm_kernel2(
    a: &[f32],
    r_kk: f32,
    q: &mut [f32],
    nj: usize,
    ni: usize,
    k: usize,
) {
    let mut q = chunk_mut(q, MapLinear::new(1));
    let i = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if i < ni {
        q[0] = a[i * nj + k] / r_kk;
    }
}

// kernel3a: r[k*nj+j] = dot(q[:,k], a[:,j]) for j > k
#[gpu::cuda_kernel]
pub fn gramschm_kernel3a(
    q: &[f32],
    a: &[f32],
    r: &mut [f32],
    ni: usize,
    nj: usize,
    k: usize,
) {
    let mut r = chunk_mut(r, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if j > k && j < nj {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < ni {
            sum += q[i * nj + k] * a[i * nj + j];
            i += 1;
        }
        r[0] = sum;
    }
}

// kernel3b: a[i*nj+j] -= q[i*nj+k] * r[k*nj+j] for j > k
#[gpu::cuda_kernel]
pub fn gramschm_kernel3b(
    q: &[f32],
    r: &[f32],
    a: &mut [f32],
    ni: usize,
    nj: usize,
    k: usize,
) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if j > k && j < nj && i < ni {
        a[0] = a[0] - q[i * nj + k] * r[k * nj + j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_gramschm(ni: usize, nj: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Initialize A[i][j] = (i*j+1) / (ni*nj) as f32
        let mut h_a: Vec<f32> = vec![0.0; ni * nj];
        for i in 0..ni {
            for j in 0..nj {
                h_a[i * nj + j] = ((i * j + 1) as f32) / (ni * nj) as f32;
            }
        }
        let mut h_a_cpu = h_a.clone();
        let mut h_r_cpu: Vec<f32> = vec![0.0; nj * nj];
        let mut h_q_cpu: Vec<f32> = vec![0.0; ni * nj];

        // CPU reference (Gram-Schmidt)
        for k in 0..nj {
            let mut nrm = 0.0f32;
            for i in 0..ni {
                nrm += h_a_cpu[i * nj + k] * h_a_cpu[i * nj + k];
            }
            h_r_cpu[k * nj + k] = nrm.sqrt();
            for i in 0..ni {
                h_q_cpu[i * nj + k] = h_a_cpu[i * nj + k] / h_r_cpu[k * nj + k];
            }
            for j in (k + 1)..nj {
                let mut dot = 0.0f32;
                for i in 0..ni {
                    dot += h_q_cpu[i * nj + k] * h_a_cpu[i * nj + j];
                }
                h_r_cpu[k * nj + j] = dot;
                for i in 0..ni {
                    h_a_cpu[i * nj + j] -= h_q_cpu[i * nj + k] * h_r_cpu[k * nj + j];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_r_gpu: Vec<f32> = vec![0.0; nj * nj];
        let mut h_q_gpu: Vec<f32> = vec![0.0; ni * nj];

        cuda_ctx(0, |ctx, m_module| {
            let mut d_a = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a");
            let mut d_r = ctx
                .new_tensor_view(h_r_gpu.as_mut_slice())
                .expect("alloc r");
            let mut d_q = ctx
                .new_tensor_view(h_q_gpu.as_mut_slice())
                .expect("alloc q");

            let block_size: u32 = 256;

            for k in 0..nj {
                // kernel1 on CPU: r[k][k] = sqrt(sum_i(a[i*nj+k]^2))
                let mut h_a_tmp = vec![0.0f32; ni * nj];
                d_a.copy_to_host(&mut h_a_tmp).expect("copy a to host");
                let mut nrm = 0.0f32;
                for i in 0..ni {
                    nrm += h_a_tmp[i * nj + k] * h_a_tmp[i * nj + k];
                }
                h_r_gpu[k * nj + k] = nrm.sqrt();
                d_r.copy_from_host(&h_r_gpu).expect("copy r to device");
                let r_kk = h_r_gpu[k * nj + k];

                // kernel2: q[:,k] = a[:,k] / r[k][k]
                let grid_x = (ni as u32 + block_size - 1) / block_size;
                let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
                gramschm_kernel2::launch(
                    config, ctx, m_module, &d_a, r_kk, &mut d_q, nj, ni, k,
                )
                .expect("kernel2 failed");

                // kernel3a: r[k][j] = dot(q[:,k], a[:,j])
                let grid_x = (nj as u32 + block_size - 1) / block_size;
                let config = gpu_host::gpu_config!(grid_x, 1, 1, block_size, 1, 1, 0);
                gramschm_kernel3a::launch(
                    config, ctx, m_module, &d_q, &d_a, &mut d_r, ni, nj, k,
                )
                .expect("kernel3a failed");

                // kernel3b: a[i][j] -= q[i][k] * r[k][j]
                let grid_x = (nj as u32 + block_size - 1) / block_size;
                let grid_y = (ni as u32 + block_size - 1) / block_size;
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                gramschm_kernel3b::launch(
                    config, ctx, m_module, &d_q, &d_r, &mut d_a, ni, nj, k,
                )
                .expect("kernel3b failed");
            }

            d_a.copy_to_host(&mut h_a_gpu).expect("copy a");
            d_r.copy_to_host(&mut h_r_gpu).expect("copy r");
            d_q.copy_to_host(&mut h_q_gpu).expect("copy q");
        });

        (h_a_gpu, h_a_cpu, h_r_gpu, h_r_cpu)
    }

    #[test]
    fn test_gramschm() {
        let ni = 32;
        let nj = 32;
        let (a_gpu, a_cpu, r_gpu, r_cpu) = run_gramschm(ni, nj);

        // Compare R matrices
        for i in 0..r_gpu.len() {
            assert!(
                (r_gpu[i] - r_cpu[i]).abs() < 1.0,
                "R mismatch at {}: gpu={} cpu={}",
                i, r_gpu[i], r_cpu[i],
            );
        }

        // Verify R diagonal is positive (property of QR)
        for k in 0..nj {
            assert!(
                r_gpu[k * nj + k] > 0.0,
                "R diagonal at {} should be positive: {}",
                k, r_gpu[k * nj + k],
            );
        }
    }
}
```

Note: kernel2's chunk_mut maps thread i → element i of q. We launch ni threads. Thread i writes to q[i] via chunk_mut. But we need q[i*nj+k], not q[i]. The thread grid has ni threads, so MapLinear maps thread 0→q[0], thread 1→q[1], etc. But the actual write location should be q[i*nj+k].

**FIX**: We need to launch ni*nj threads total and only have threads where the second index == k actually write. Or, we can pass a column-slice. Since index_mut exists, pass `d_q` column data via a temporary buffer.

**SIMPLEST FIX for kernel2**: Launch ni*nj threads (full matrix), use Map2D, only write when column == k:

```rust
#[gpu::cuda_kernel]
pub fn gramschm_kernel2(
    a: &[f32],
    r_kk: f32,
    q: &mut [f32],
    nj: usize,
    ni: usize,
    k: usize,
) {
    let mut q = chunk_mut(q, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < ni && j == k {
        q[(0, 0)] = a[i * nj + k] / r_kk;
    }
}
```

Launch with 2D grid: (ceil(nj/bs), ceil(ni/bs)).

**Similarly kernel3a**: Launch nj threads, thread j writes to r[j]. But we need r[k*nj+j]. Use Map2D with width=nj, launch nj*nj threads, only write when row == k:

```rust
#[gpu::cuda_kernel]
pub fn gramschm_kernel3a(
    q: &[f32],
    a: &[f32],
    r: &mut [f32],
    ni: usize,
    nj: usize,
    k: usize,
) {
    let mut r = chunk_mut(r, Map2D::new(nj));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if row == k && j > k && j < nj {
        let mut sum = 0.0f32;
        let mut i: usize = 0;
        while i < ni {
            sum += q[i * nj + k] * a[i * nj + j];
            i += 1;
        }
        r[(0, 0)] = sum;
    }
}
```

Launch with 2D grid: (ceil(nj/bs), ceil(nj/bs)).

- [ ] **Step 3: Add to workspace**

Add `"polybench/gramschm"` to the members list in `examples/Cargo.toml`.

- [ ] **Step 4: Build and test**

```bash
cd /home/sanghle/work/seguru/examples && cargo test -p polybench-gramschm -- --test-threads=1
```

- [ ] **Step 5: Commit**

```bash
git add examples/polybench/gramschm/ examples/Cargo.toml
git commit -m "examples: port GRAMSCHM benchmark from PolybenchGPU"
```

---

### Task 3: DOITGEN — 3D Tensor Operation Packed into 2D Grid

**Files:**
- Create: `examples/polybench/doitgen/Cargo.toml`
- Create: `examples/polybench/doitgen/src/lib.rs`
- Modify: `examples/Cargo.toml` (add member)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "polybench-doitgen"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Create src/lib.rs**

```rust
use gpu::prelude::*;

// sum[r*(nq*np) + q*np + p] = sum_s(A[r*(nq*np) + q*np + s] * C4[s*np + p])
// Pack r,q into Y dimension: qr = r*nq + q
#[gpu::cuda_kernel]
pub fn doitgen_kernel1(
    a: &[f32],
    c4: &[f32],
    sum_arr: &mut [f32],
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut sum_arr = chunk_mut(sum_arr, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        let mut val = 0.0f32;
        let mut s: usize = 0;
        while s < np {
            val += a[r * (nq * np) + q * np + s] * c4[s * np + p];
            s += 1;
        }
        sum_arr[0] = val;
    }
}

// A[r*(nq*np) + q*np + p] = sum[r*(nq*np) + q*np + p]
#[gpu::cuda_kernel]
pub fn doitgen_kernel2(
    sum_arr: &[f32],
    a: &mut [f32],
    nr: usize,
    nq: usize,
    np: usize,
) {
    let mut a = chunk_mut(a, MapLinear::new(1));
    let p = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let qr = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let q = qr % nq;
    let r = qr / nq;
    if p < np && q < nq && r < nr {
        a[0] = sum_arr[r * (nq * np) + q * np + p];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_doitgen(nr: usize, nq: usize, np: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_a: Vec<f32> = vec![0.0; nr * nq * np];
        let mut h_c4: Vec<f32> = vec![0.0; np * np];
        for i in 0..nr {
            for j in 0..nq {
                for k in 0..np {
                    h_a[i * (nq * np) + j * np + k] = (i * j + k) as f32 / np as f32;
                }
            }
        }
        for i in 0..np {
            for j in 0..np {
                h_c4[i * np + j] = (i * j) as f32 / np as f32;
            }
        }

        // CPU reference
        let mut h_a_cpu = h_a.clone();
        for r in 0..nr {
            for q in 0..nq {
                let mut sum_tmp = vec![0.0f32; np];
                for p in 0..np {
                    let mut val = 0.0f32;
                    for s in 0..np {
                        val += h_a_cpu[r * (nq * np) + q * np + s] * h_c4[s * np + p];
                    }
                    sum_tmp[p] = val;
                }
                for p in 0..np {
                    h_a_cpu[r * (nq * np) + q * np + p] = sum_tmp[p];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_sum: Vec<f32> = vec![0.0; nr * nq * np];

        cuda_ctx(0, |ctx, m_module| {
            let d_a_ro = ctx.new_tensor_view(h_a.as_slice()).expect("alloc a_ro");
            let mut d_a = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a");
            let d_c4 = ctx.new_tensor_view(h_c4.as_slice()).expect("alloc c4");
            let mut d_sum = ctx
                .new_tensor_view(h_sum.as_mut_slice())
                .expect("alloc sum");

            let block_size: u32 = 16;
            let grid_x = (np as u32 + block_size - 1) / block_size;
            let grid_y = ((nr * nq) as u32 + block_size - 1) / block_size;

            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            doitgen_kernel1::launch(
                config, ctx, m_module, &d_a_ro, &d_c4, &mut d_sum, nr, nq, np,
            )
            .expect("kernel1 failed");

            let config =
                gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
            doitgen_kernel2::launch(
                config, ctx, m_module, &d_sum, &mut d_a, nr, nq, np,
            )
            .expect("kernel2 failed");

            d_a.copy_to_host(&mut h_a_gpu).expect("copy failed");
        });

        (h_a_gpu, h_a_cpu)
    }

    #[test]
    fn test_doitgen() {
        let nr = 8;
        let nq = 8;
        let np = 8;
        let (gpu, cpu) = run_doitgen(nr, nq, np);

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "result is all zeros");
    }
}
```

- [ ] **Step 3: Add to workspace**

Add `"polybench/doitgen"` to the members list in `examples/Cargo.toml`.

- [ ] **Step 4: Build and test**

```bash
cd /home/sanghle/work/seguru/examples && cargo test -p polybench-doitgen -- --test-threads=1
```

- [ ] **Step 5: Commit**

```bash
git add examples/polybench/doitgen/ examples/Cargo.toml
git commit -m "examples: port DOITGEN benchmark from PolybenchGPU"
```

---

### Task 4: LU — LU Decomposition with Aliased Read/Write

**Files:**
- Create: `examples/polybench/lu/Cargo.toml`
- Create: `examples/polybench/lu/src/lib.rs`
- Modify: `examples/Cargo.toml` (add member)

**Design:** Both kernels need to read from A at arbitrary positions while writing to specific positions. Solution: create two separate GPU allocations from same data — `d_a_read` (read-only) and `d_a_write` (mutable). After each kernel, copy d_a_write back to d_a_read (or re-allocate). Launch full n×n 2D grid with Map2D.

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "polybench-lu"
version = "0.1.0"
edition = "2024"

[dependencies]
gpu = {workspace = true}
gpu_host = {workspace = true}
```

- [ ] **Step 2: Create src/lib.rs**

```rust
use gpu::prelude::*;

// A[k][j] /= A[k][k] for j > k (only row k is modified)
#[gpu::cuda_kernel]
pub fn lu_kernel1(a_read: &[f32], a_write: &mut [f32], n: usize, k: usize) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i == k && j > k && j < n {
        a_write[(0, 0)] = a_read[k * n + j] / a_read[k * n + k];
    }
}

// A[i][j] -= A[i][k] * A[k][j] for i,j > k
#[gpu::cuda_kernel]
pub fn lu_kernel2(a_read: &[f32], a_write: &mut [f32], n: usize, k: usize) {
    let mut a_write = chunk_mut(a_write, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i > k && j > k && i < n && j < n {
        a_write[(0, 0)] = a_read[i * n + j] - a_read[i * n + k] * a_read[k * n + j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_host::cuda_ctx;

    fn run_lu(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h_a: Vec<f32> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                h_a[i * n + j] = (i * j + 1) as f32 / n as f32;
            }
        }

        // CPU reference
        let mut h_a_cpu = h_a.clone();
        for k in 0..n {
            for j in (k + 1)..n {
                h_a_cpu[k * n + j] /= h_a_cpu[k * n + k];
            }
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    h_a_cpu[i * n + j] -= h_a_cpu[i * n + k] * h_a_cpu[k * n + j];
                }
            }
        }

        // GPU
        let mut h_a_gpu = h_a.clone();
        let mut h_a_read = h_a.clone();

        cuda_ctx(0, |ctx, m_module| {
            let mut d_a_write = ctx
                .new_tensor_view(h_a_gpu.as_mut_slice())
                .expect("alloc a_write");
            let mut d_a_read = ctx
                .new_tensor_view(h_a_read.as_mut_slice())
                .expect("alloc a_read");

            let block_size: u32 = 16;
            let grid_x = (n as u32 + block_size - 1) / block_size;
            let grid_y = (n as u32 + block_size - 1) / block_size;

            for k in 0..n {
                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                lu_kernel1::launch(
                    config, ctx, m_module, &d_a_read, &mut d_a_write, n, k,
                )
                .expect("kernel1 failed");

                // Sync: copy d_a_write → d_a_read for kernel2 to read updated values
                d_a_write.copy_to_host(&mut h_a_gpu).expect("copy to host");
                d_a_read.copy_from_host(&h_a_gpu).expect("copy to device");

                let config =
                    gpu_host::gpu_config!(grid_x, grid_y, 1, block_size, block_size, 1, 0);
                lu_kernel2::launch(
                    config, ctx, m_module, &d_a_read, &mut d_a_write, n, k,
                )
                .expect("kernel2 failed");

                // Sync for next iteration
                d_a_write.copy_to_host(&mut h_a_gpu).expect("copy to host");
                d_a_read.copy_from_host(&h_a_gpu).expect("copy to device");
            }

            d_a_write.copy_to_host(&mut h_a_gpu).expect("copy failed");
        });

        (h_a_gpu, h_a_cpu)
    }

    #[test]
    fn test_lu() {
        let n = 32;
        let (gpu, cpu) = run_lu(n);

        for i in 0..gpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1.0,
                "Mismatch at {}: gpu={} cpu={}",
                i, gpu[i], cpu[i],
            );
        }

        let nonzero = gpu.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero, "result is all zeros");
    }
}
```

- [ ] **Step 3: Add to workspace**

Add `"polybench/lu"` to the members list in `examples/Cargo.toml`.

- [ ] **Step 4: Build and test**

```bash
cd /home/sanghle/work/seguru/examples && cargo test -p polybench-lu -- --test-threads=1
```

- [ ] **Step 5: Commit**

```bash
git add examples/polybench/lu/ examples/Cargo.toml
git commit -m "examples: port LU benchmark from PolybenchGPU"
```

---

### Task 5: Final Integration Test and Combined Commit

- [ ] **Step 1: Run all 4 benchmarks together**

```bash
cd /home/sanghle/work/seguru/examples
cargo test -p polybench-corr -p polybench-gramschm -p polybench-doitgen -p polybench-lu -- --test-threads=1
```

- [ ] **Step 2: Verify all pass, squash or amend if needed**

If any test failed during individual tasks and was fixed, ensure the final state is clean and all tests pass.
