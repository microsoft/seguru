//! KernelBench Level-1 skill-doc stress test.
//!
//! Each kernel is written using ONLY patterns from
//! `docs/cuda-to-seguru-porting-skill.md`. Gaps encountered while porting are
//! noted in comments (search for "GAP:" in this file).
//!
//! Problems (from ScalingIntelligence/KernelBench, level1/):
//!   19_ReLU, 21_Sigmoid, 23_Softmax, 40_LayerNorm, 1_SquareMatmul
//!
//! Sizes reduced from KernelBench where necessary to fit GPU memory.

#![allow(non_snake_case)]

use gpu::cg::{CGOperations, ReduxAdd, ReduxMax, ThreadWarpTile, WarpReduceOp};
use gpu::prelude::*;
use gpu::CacheStreamLoadStore;
use std::time::Instant;

// ======================================================================
// 19_ReLU — elementwise
// ======================================================================
// Skill doc patterns used: chunk_mut + MapContinuousLinear (Memory Write Patterns),
//   u32 params (Golden Rules), bounds guard (Grid-stride-loops DO NOT work).
#[gpu::cuda_kernel]
pub fn relu_kernel(x: &[f32], y: &mut [f32], n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        let v = x[idx as usize];
        y_chunk[0] = if v > 0.0 { v } else { 0.0 };
    }
}

// ======================================================================
// 21_Sigmoid — elementwise + exp
// ======================================================================
#[gpu::cuda_kernel]
pub fn sigmoid_kernel(x: &[f32], y: &mut [f32], n: u32) {
    let mut y_chunk = chunk_mut(y, MapContinuousLinear::new(1));
    let idx = block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>();
    if idx < n {
        let v = x[idx as usize];
        y_chunk[0] = 1.0 / (1.0 + (-v).exp());
    }
}

// ======================================================================
// 23_Softmax (row-wise, dim=1) — warp per row
// ======================================================================
// Skill doc patterns: warp-strided subslice, ThreadWarpTile + redux, reshape_map
//   for strided output, ldcs/stcs cache hints, subslice for row access.
const SOFTMAX_N: u32 = 4096;

#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[f32], y: &mut [f32]) {
    let warp = ThreadWarpTile::<32>;
    let wpb = warp.meta_group_size();
    let row = block_id::<DimX>() * wpb + warp.subgroup_id();
    let lane = warp.thread_rank();

    let row_off = row * SOFTMAX_N;
    let x_row = &x[row_off as usize..(row_off + SOFTMAX_N) as usize];

    // Pass 1: max
    let mut m = f32::NEG_INFINITY;
    let mut i = lane;
    while i < SOFTMAX_N {
        let v = x_row[i as usize].ldcs();
        if v > m { m = v; }
        i += warp.size();
    }
    let mx: f32 = warp.redux(ReduxMax, m);

    // Pass 2: sum of exp
    let mut s = 0.0f32;
    let mut i = lane;
    while i < SOFTMAX_N {
        s += (x_row[i as usize].ldcs() - mx).exp();
        i += warp.size();
    }
    let sm: f32 = warp.redux(ReduxAdd, s);
    let inv = 1.0f32 / sm;

    // Pass 3: output — strided per-lane writes via reshape_map.
    let mut y_chunk = chunk_mut(
        y,
        reshape_map!(
            [SOFTMAX_N / 32] | [32, wpb * grid_dim::<DimX>()] => layout: [t0, i0, t1]
        ),
    );
    let mut slot: u32 = 0;
    let mut i = lane;
    while i < SOFTMAX_N {
        let v = (x_row[i as usize].ldcs() - mx).exp() * inv;
        y_chunk[slot].stcs(v);
        i += warp.size();
        slot += 1;
    }
}

// ======================================================================
// 1_SquareMatmul — 16×16 tile, shared-memory tiled GEMM
// ======================================================================
// Skill doc patterns: shared-memory tiling "key to CUDA parity", GpuShared +
//   chunk_mut(reshape_map!) for LOADS, raw indexing on shared for compute,
//   sync_threads between phases, Map2D for 2D output slot, bounds guard.
#[gpu::cuda_kernel]
pub fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let mut c = chunk_mut(c, Map2D::new(n as usize));

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let col = block_id::<DimX>() * 16 + tx;
    let row = block_id::<DimY>() * 16 + ty;

    let mut tile_a = gpu::GpuShared::<[f32; 256]>::zero();
    let mut tile_b = gpu::GpuShared::<[f32; 256]>::zero();

    // Per-thread disjoint slot: memory = t1*16 + t0 = ty*16 + tx.
    let load_map = reshape_map!([1] | [16, 16] => layout: [i0, t0, t1]);

    let mut sum = 0.0f32;
    let num_tiles = (n + 15) / 16;
    let mut t: u32 = 0;
    while t < num_tiles {
        {
            let mut chunk_a = tile_a.chunk_mut(load_map);
            let a_col = t * 16 + tx;
            chunk_a[0] = if row < n && a_col < n {
                a[(row * n + a_col) as usize]
            } else {
                0.0
            };
        }
        {
            let mut chunk_b = tile_b.chunk_mut(load_map);
            let b_row = t * 16 + ty;
            chunk_b[0] = if b_row < n && col < n {
                b[(b_row * n + col) as usize]
            } else {
                0.0
            };
        }

        sync_threads();

        let mut k: u32 = 0;
        while k < 16 {
            sum += tile_a[(ty * 16 + k) as usize] * tile_b[(k * 16 + tx) as usize];
            k += 1;
        }

        sync_threads();
        t += 1;
    }

    if row < n && col < n {
        c[(0, 0)] = sum;
    }
}

// ======================================================================
// CPU references (for correctness check)
// ======================================================================
fn cpu_relu(x: &[f32], y: &mut [f32]) {
    for (xv, yv) in x.iter().zip(y.iter_mut()) {
        *yv = xv.max(0.0);
    }
}
fn cpu_sigmoid(x: &[f32], y: &mut [f32]) {
    for (xv, yv) in x.iter().zip(y.iter_mut()) {
        *yv = 1.0 / (1.0 + (-xv).exp());
    }
}
fn cpu_softmax_row(x: &[f32], y: &mut [f32], m: usize, n: usize) {
    for r in 0..m {
        let row = &x[r * n..(r + 1) * n];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in row { s += (v - mx).exp(); }
        let inv = 1.0 / s;
        for (i, v) in row.iter().enumerate() {
            y[r * n + i] = (v - mx).exp() * inv;
        }
    }
}
fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // Naive triple loop — only used on small sizes for verification.
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n { s += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = s;
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// ======================================================================
// Bench driver
// ======================================================================
fn main() {
    let iters = 100;

    // Shapes
    let elem_m: usize = 4096;
    let elem_n: usize = 16384;
    let elem_total = elem_m * elem_n;

    let sm_m: usize = 4096;
    let sm_n: usize = SOFTMAX_N as usize;

    let mm_n: usize = 4096;

    let mut rng: u32 = 0x13579bdf;
    let mut rnd = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
    };

    println!("=== KernelBench Level-1 skill-doc stress test ===");

    gpu_host::cuda_ctx(0, |ctx, md| {
        // ---------- ReLU ----------
        {
            let h_x: Vec<f32> = (0..elem_total).map(|_| rnd()).collect();
            let mut h_y = vec![0.0f32; elem_total];
            let mut h_ref = vec![0.0f32; elem_total];
            cpu_relu(&h_x, &mut h_ref);

            let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let n = elem_total as u32;
            let bs: u32 = 256;
            let gs: u32 = n.div_ceil(bs);
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            relu_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, n).unwrap();
            ctx.sync().unwrap();
            d_y.copy_to_host(&mut h_y).unwrap();
            let err = max_abs_diff(&h_y, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
                relu_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, n).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("relu      [{}x{}={}]  SeGuRu: {:8.2} us  err={:.1e}",
                elem_m, elem_n, elem_total, us, err);
        }

        // ---------- Sigmoid ----------
        {
            let h_x: Vec<f32> = (0..elem_total).map(|_| rnd()).collect();
            let mut h_y = vec![0.0f32; elem_total];
            let mut h_ref = vec![0.0f32; elem_total];
            cpu_sigmoid(&h_x, &mut h_ref);

            let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            let n = elem_total as u32;
            let bs: u32 = 256;
            let gs: u32 = n.div_ceil(bs);
            let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
            sigmoid_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, n).unwrap();
            ctx.sync().unwrap();
            d_y.copy_to_host(&mut h_y).unwrap();
            let err = max_abs_diff(&h_y, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gs, 1, 1, bs, 1, 1, 0);
                sigmoid_kernel::launch(cfg, ctx, md, &d_x, &mut d_y, n).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("sigmoid   [{}x{}={}]  SeGuRu: {:8.2} us  err={:.1e}",
                elem_m, elem_n, elem_total, us, err);
        }

        // ---------- Softmax ----------
        {
            let h_x: Vec<f32> = (0..sm_m * sm_n).map(|_| rnd()).collect();
            let mut h_y = vec![0.0f32; sm_m * sm_n];
            let mut h_ref = vec![0.0f32; sm_m * sm_n];
            cpu_softmax_row(&h_x, &mut h_ref, sm_m, sm_n);

            let d_x = ctx.new_tensor_view(h_x.as_slice()).unwrap();
            let mut d_y = ctx.new_tensor_view(h_y.as_mut_slice()).unwrap();
            const WPB: u32 = 8;
            const BDIM: u32 = 32 * WPB;
            let gs: u32 = (sm_m as u32).div_ceil(WPB);
            let cfg = gpu_host::gpu_config!(gs, 1, 1, BDIM, 1, 1, 0);
            softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y).unwrap();
            ctx.sync().unwrap();
            d_y.copy_to_host(&mut h_y).unwrap();
            let err = max_abs_diff(&h_y, &h_ref);

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gs, 1, 1, BDIM, 1, 1, 0);
                softmax_kernel::launch(cfg, ctx, md, &d_x, &mut d_y).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("softmax   [{}x{}]       SeGuRu: {:8.2} us  err={:.1e}",
                sm_m, sm_n, us, err);
        }

        // ---------- Matmul ----------
        {
            // Reference via small CPU check: verify with small size first.
            // Then run perf on full 4096×4096.
            let small_n = 256usize;
            let h_a_s: Vec<f32> = (0..small_n * small_n).map(|_| rnd()).collect();
            let h_b_s: Vec<f32> = (0..small_n * small_n).map(|_| rnd()).collect();
            let mut h_c_s = vec![0.0f32; small_n * small_n];
            let mut h_ref_s = vec![0.0f32; small_n * small_n];
            cpu_matmul(&h_a_s, &h_b_s, &mut h_ref_s, small_n);

            let d_a_s = ctx.new_tensor_view(h_a_s.as_slice()).unwrap();
            let d_b_s = ctx.new_tensor_view(h_b_s.as_slice()).unwrap();
            let mut d_c_s = ctx.new_tensor_view(h_c_s.as_mut_slice()).unwrap();
            let gs: u32 = (small_n as u32).div_ceil(16);
            let cfg = gpu_host::gpu_config!(gs, gs, 1, 16, 16, 1, 0);
            matmul_tiled::launch(cfg, ctx, md, &d_a_s, &d_b_s, &mut d_c_s, small_n as u32).unwrap();
            ctx.sync().unwrap();
            d_c_s.copy_to_host(&mut h_c_s).unwrap();
            let err = max_abs_diff(&h_c_s, &h_ref_s);

            // Perf on 4096×4096 (correctness already validated at small size).
            let h_a: Vec<f32> = (0..mm_n * mm_n).map(|_| rnd()).collect();
            let h_b: Vec<f32> = (0..mm_n * mm_n).map(|_| rnd()).collect();
            let mut h_c = vec![0.0f32; mm_n * mm_n];
            let d_a = ctx.new_tensor_view(h_a.as_slice()).unwrap();
            let d_b = ctx.new_tensor_view(h_b.as_slice()).unwrap();
            let mut d_c = ctx.new_tensor_view(h_c.as_mut_slice()).unwrap();
            let gs: u32 = (mm_n as u32).div_ceil(16);

            let cfg = gpu_host::gpu_config!(gs, gs, 1, 16, 16, 1, 0);
            matmul_tiled::launch(cfg, ctx, md, &d_a, &d_b, &mut d_c, mm_n as u32).unwrap();
            ctx.sync().unwrap();

            let start = Instant::now();
            for _ in 0..iters {
                let cfg = gpu_host::gpu_config!(gs, gs, 1, 16, 16, 1, 0);
                matmul_tiled::launch(cfg, ctx, md, &d_a, &d_b, &mut d_c, mm_n as u32).unwrap();
            }
            ctx.sync().unwrap();
            let us = start.elapsed().as_micros() as f64 / iters as f64;
            println!("matmul    [{}x{}]      SeGuRu: {:8.2} us  err={:.1e} (256×256 check)",
                mm_n, mm_n, us, err);
        }
    });

    println!("=== done ===");
    println!("Run python/run_torch_baseline.py for PyTorch comparison numbers.");
}
