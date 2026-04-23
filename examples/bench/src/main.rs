use gpu::prelude::*;
use std::time::Instant;

// ===== Kernel definitions =====

#[gpu::cuda_kernel]
pub fn bench_vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = chunk_mut(c, MapLinear::new(1));
    let idx = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if idx < n {
        c[0] = a[idx] + b[idx];
    }
}

#[gpu::cuda_kernel]
pub fn bench_vector_add_u32(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let mut c = chunk_mut(c, MapLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        c[0] = a[idx as usize] + b[idx as usize];
    }
}

#[gpu::cuda_kernel(dynamic_shared)]
pub fn bench_reduce_sum(input: &[f32], output: &mut [f32], n: usize) {
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = (block_id::<DimX>() * bdim * 2 + tid) as usize;

    let smem = smem_alloc.alloc::<f32>(bdim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut output_chunk = chunk_mut(
        output,
        reshape_map!([1] | [(bdim, 1), grid_dim::<DimX>()] => layout: [i0, t1, t0]),
    );

    let mut sum = 0.0f32;
    if idx < n {
        sum += input[idx];
    }
    if idx + (bdim as usize) < n {
        sum += input[idx + bdim as usize];
    }
    smem_chunk[0] = sum;
    sync_threads();

    for order in (0..16).rev() {
        let stride = 1u32 << order;
        if stride >= bdim {
            continue;
        }
        let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
        if tid < stride {
            let right = smem_chunk[1];
            let left = smem_chunk[0];
            smem_chunk[0] = left + right;
        }
        sync_threads();
    }

    if tid == 0 {
        output_chunk[0] = *smem[0];
    }
}

#[gpu::cuda_kernel]
pub fn bench_gemm(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < n && j < n {
        let mut sum = 0.0f32;
        let mut k: usize = 0;
        while k < n {
            sum += a[i * n + k] * b[k * n + j];
            k += 1;
        }
        c[(0, 0)] = sum;
    }
}

#[gpu::cuda_kernel]
pub fn bench_gemm_u32(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));
    let j = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    let i = block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>();
    if i < n && j < n {
        let mut sum = 0.0f32;
        let mut k: u32 = 0;
        while k < n {
            sum += a[(i * n + k) as usize] * b[(k * n + j) as usize];
            k += 1;
        }
        c[(0, 0)] = sum;
    }
}

// GEMM using subslice + iterator pattern (like existing matmul example)
#[gpu::cuda_kernel]
pub fn bench_gemm_slice(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let j = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    let i = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    if i < n && j < n {
        let mut sum = 0.0f32;
        let aa: &[f32] = &a[i * n..i * n + n];
        let mut b_idx = j;
        for a_val in aa {
            sum += a_val * b[b_idx];
            b_idx += n;
        }
        c[(0, 0)] = sum;
    }
}

// Shared-memory tiled GEMM using dynamic smem (no bounds checks on ld.shared)
#[gpu::cuda_kernel(dynamic_shared)]
pub fn bench_gemm_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let col = block_id::<DimX>() * 16 + tx;
    let row = block_id::<DimY>() * 16 + ty;

    // Dynamic shared memory — two 16x16 tiles
    let tile_a = smem_alloc.alloc::<f32>(256); // 16*16
    let tile_b = smem_alloc.alloc::<f32>(256); // 16*16

    let mut sum = 0.0f32;

    let num_tiles = (n + 15) / 16;
    let mut t: u32 = 0;
    while t < num_tiles {
        // Load A[row][t*16 + tx] into tile_a
        {
            let a_col = t * 16 + tx;
            let mut chunk_a = tile_a.chunk_mut(MapLinear::new(1));
            if row < n && a_col < n {
                chunk_a[0] = a[(row * n + a_col) as usize];
            } else {
                chunk_a[0] = 0.0;
            }
        }

        // Load B[t*16 + ty][col] into tile_b
        {
            let b_row = t * 16 + ty;
            let mut chunk_b = tile_b.chunk_mut(MapLinear::new(1));
            if b_row < n && col < n {
                chunk_b[0] = b[(b_row * n + col) as usize];
            } else {
                chunk_b[0] = 0.0;
            }
        }

        sync_threads();

        // Compute from shared memory — *smem[i] on dynamic alloc = direct ld.shared!
        let mut k: u32 = 0;
        while k < 16 {
            sum += *tile_a[(ty * 16 + k) as usize] * *tile_b[(k * 16 + tx) as usize];
            k += 1;
        }

        sync_threads();
        t += 1;
    }

    if row < n && col < n {
        c[(0, 0)] = sum;
    }
}

// ===== Main benchmark runner =====
fn main() {
    let iters = 100;

    gpu_host::cuda_ctx(0, |ctx, m| {
        // ----- Launch Overhead (empty-ish kernel) -----
        {
            let n: usize = 1;
            let mut h_c = vec![0.0f32; n];
            let h_a = vec![0.0f32; n];
            let h_b = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();

            let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
            bench_vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            ctx.sync().unwrap();

            let launch_iters = 10000;
            let start = Instant::now();
            for _ in 0..launch_iters {
                let config = gpu_host::gpu_config!(1, 1, 1, 1, 1, 1, 0);
                bench_vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "launch_overhead SeGuRu: {:.3} us/launch ({} launches)",
                elapsed.as_micros() as f64 / launch_iters as f64,
                launch_iters
            );
        }

        // ----- Vector Add (N=1M) -----
        {
            let n: usize = 1 << 20;
            let h_a = vec![1.0f32; n];
            let h_b = vec![2.0f32; n];
            let mut h_c = vec![0.0f32; n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();
            let bs: u32 = 256;
            let nb: u32 = ((n as u32) + bs - 1) / bs;

            // Warmup
            let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            d_c.copy_to_host(&mut h_c).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_vector_add::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "vector_add (usize) SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );

            // u32 version
            let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
            bench_vector_add_u32::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            ctx.sync().unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, 0);
                bench_vector_add_u32::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "vector_add (u32)   SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );
        }

        // ----- Reduce (N=1M) -----
        {
            let n: usize = 1 << 20;
            let bs: u32 = 256;
            let nb: u32 = ((n as u32) + bs * 2 - 1) / (bs * 2);
            let h_in = vec![1.0f32; n];
            let mut h_out = vec![0.0f32; nb as usize];
            let d_in = ctx.new_tensor_view(h_in.as_slice()).unwrap();
            let mut d_out = ctx.new_tensor_view(h_out.as_mut_slice()).unwrap();
            let smem = bs * 4; // sizeof(f32) = 4

            // Warmup
            let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, smem);
            bench_reduce_sum::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
            d_out.copy_to_host(&mut h_out).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(nb, 1, 1, bs, 1, 1, smem);
                bench_reduce_sum::launch(config, ctx, m, &d_in, &mut d_out, n).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "reduce SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );
        }

        // ----- GEMM (N=512) -----
        {
            let n: usize = 512;
            let sz = n * n;
            let h_a = vec![1.0f32; sz];
            let h_b = vec![1.0f32; sz];
            let mut h_c = vec![0.0f32; sz];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();

            let bx: u32 = 16;
            let by: u32 = 16;
            let gx: u32 = (n as u32 + bx - 1) / bx;
            let gy: u32 = (n as u32 + by - 1) / by;

            // Warmup
            let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_gemm::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            d_c.copy_to_host(&mut h_c).unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_gemm::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "gemm (usize) SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );

            // ----- GEMM u32 -----
            let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_gemm_u32::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            ctx.sync().unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_gemm_u32::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "gemm (u32)   SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );

            // ----- GEMM slice pattern -----
            let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
            bench_gemm_slice::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            ctx.sync().unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, 0);
                bench_gemm_slice::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "gemm (slice) SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );

            // ----- GEMM tiled (shared memory) -----
            let smem_bytes: u32 = 2 * 16 * 16 * 4; // two 16x16 f32 tiles
            let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, smem_bytes);
            bench_gemm_tiled::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            ctx.sync().unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let config = gpu_host::gpu_config!(gx, gy, 1, bx, by, 1, smem_bytes);
                bench_gemm_tiled::launch(config, ctx, m, &d_a, &d_b, &mut d_c, n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let elapsed = start.elapsed();
            println!(
                "gemm (tiled) SeGuRu: {:.3} us/iter (N={}, {} iters)",
                elapsed.as_micros() as f64 / iters as f64,
                n,
                iters
            );
        }
    });
}
