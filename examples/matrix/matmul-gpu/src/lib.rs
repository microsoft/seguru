#![no_std]
#![allow(clippy::too_many_arguments)]
#![deny(clippy::cast_possible_truncation)]

use gpu::*;

#[gpu::cuda_kernel]
pub fn inner_product_kernel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let bid_x = block_id::<gpu::DimX>();
    let bid_y = block_id::<gpu::DimY>();
    let tid_x = thread_id::<gpu::DimX>();
    let tid_y = thread_id::<gpu::DimY>();
    let bdim_x = block_dim::<gpu::DimX>();
    let bdim_y = block_dim::<gpu::DimY>();
    let dim_x = gpu::dim::<gpu::DimX>();
    let dim_y = gpu::dim::<gpu::DimY>();
    let mut row = (bid_y * bdim_y + tid_y) as usize;
    for i in 0..((n - 1) / dim_y as usize + 1) {
        let mut col = (bid_x * bdim_x + tid_x) as usize;
        for j in 0..((n - 1) / dim_x as usize + 1) {
            if row < n && col < n {
                let mut sum = 0.0;
                let aa: &[f32] = &a[row * n..row * n + n];
                let mut b_idx = col;
                for a in aa {
                    sum += a * b[b_idx];
                    b_idx += n;
                }
                c[(j, i)] = sum;
            }
            col += dim_x as usize;
        }
        row += dim_y as usize;
    }
}

/// Tile edge length; a block is `TILE * TILE` threads and owns one `TILE * TILE`
/// tile of `C`.
pub const TILE: usize = 16;

/// `C = alpha * A * B + beta * C`, one thread per element of `C`.
///
/// Counterpart of \[cuda-oxide\] `examples/gemm`'s `sgemm_naive`.
#[gpu::cuda_kernel]
pub fn sgemm_naive_kernel(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    let mut c = chunk_mut(c, Map2D::new(n));
    let row = (block_id::<DimY>() * block_dim::<DimY>() + thread_id::<DimY>()) as usize;
    let col = (block_id::<DimX>() * block_dim::<DimX>() + thread_id::<DimX>()) as usize;
    if row < m && col < n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += a[row * k + i] * b[i * n + col];
        }
        let old = c[(0, 0)];
        c[(0, 0)] = alpha * sum + beta * old;
    }
}

/// `C = alpha * A * B + beta * C` staged through `TILE * TILE` shared-memory tiles.
///
/// Counterpart of \[cuda-oxide\] `examples/tiled_gemm`'s `sgemm_tiled`, which needs
/// two `unsafe` blocks for the same tiles.
///
/// Each tile is written through a `chunk_mut` that gives every thread exactly one
/// element, so the loads cannot race; the borrow ends before `sync_threads`, after
/// which the tile is read shared by the whole block.
///
/// Requires a `TILE * TILE` block. Masking the thread ids to that range is what
/// lets the bounds checks on the shared tiles fold away, since `ty * TILE + i` is
/// then visibly below `TILE * TILE`.
#[gpu::cuda_kernel]
pub fn sgemm_tiled_kernel(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    let mut tile_a = GpuShared::<[f32; TILE * TILE]>::init(0.0f32);
    let mut tile_b = GpuShared::<[f32; TILE * TILE]>::init(0.0f32);

    let tx = thread_id::<DimX>() as usize % TILE;
    let ty = thread_id::<DimY>() as usize % TILE;
    let row = block_id::<DimY>() as usize * TILE + ty;
    let col = block_id::<DimX>() as usize * TILE + tx;

    let num_tiles = k.div_ceil(TILE);
    let mut sum = 0.0f32;
    let mut tile = 0usize;
    while tile < num_tiles {
        let tile_start = tile * TILE;
        {
            let mut chunk_a = tile_a.chunk_mut(MapLinear::new(1));
            let mut chunk_b = tile_b.chunk_mut(MapLinear::new(1));
            let a_col = tile_start + tx;
            chunk_a[0] = if row < m && a_col < k { a[row * k + a_col] } else { 0.0 };
            let b_row = tile_start + ty;
            chunk_b[0] = if b_row < k && col < n { b[b_row * n + col] } else { 0.0 };
        }
        sync_threads();

        let mut i = 0usize;
        while i < TILE {
            sum += tile_a[ty * TILE + i] * tile_b[i * TILE + tx];
            i += 1;
        }
        sync_threads();

        tile += 1;
    }

    let mut c = chunk_mut(c, Map2D::new(n));
    if row < m && col < n {
        let old = c[(0, 0)];
        c[(0, 0)] = alpha * sum + beta * old;
    }
}

