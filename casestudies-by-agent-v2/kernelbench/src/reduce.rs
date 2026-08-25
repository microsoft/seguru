//! Reductions along the last dimension of a `[rows, cols]` tensor.
//!
//! All of these use one CTA of [`ROW_BLOCK`] threads per row, with `Float4`
//! loads so a CTA sweeps [`ROW_TILE`] = 1024 columns per pass, and warp-shuffle
//! cooperative-group reductions (see [`crate::device`]) rather than
//! shared-memory trees.
//!
//! The host pads every row to a whole number of CTA tiles, so `chunk_mut`'s
//! `reshape_map!` covers the buffer exactly; the loop over the tiles keeps a
//! `col < cols4` predicate, which is uniform (and free) whenever `cols` is a
//! multiple of 1024.

use crunchy::unroll;
use gpu::*;

use crate::activation::{dexp, dlog};
use crate::device::{block_exclusive_scan, block_reduce_max, block_reduce_min_i32, block_reduce_sum};
use crate::util::{ROW_BLOCK, ROW_TILE, from_float4, pad_rows, row_stride, to_float4_padded, unpad_rows};

/// Online (single-pass) softmax over each row.
///
/// The statistics pass keeps a running `(max, sum)` pair per thread and rescales
/// the partial sum whenever the running max moves, so the row is read once for
/// the statistics and once for the normalisation instead of three times.
#[gpu::cuda_kernel]
pub fn softmax_kernel(x: &[Float4], y: &mut [Float4], cols4: u32, items: u32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out =
        chunk_mut(y, reshape_map!([items] | [ROW_BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]));
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut m = f32::MIN;
    let mut s = 0.0f32;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            let old = m;
            m = m.max(v[0]).max(v[1]).max(v[2]).max(v[3]);
            s *= dexp(old - m);
            s += dexp(v[0] - m) + dexp(v[1] - m) + dexp(v[2] - m) + dexp(v[3] - m);
        }
        k += 1;
    }

    let row_max = block_reduce_max(m);
    let row_sum = block_reduce_sum(s * dexp(m - row_max));
    let inv = 1.0 / row_sum;

    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            out[k] = Float4::new([
                dexp(v[0] - row_max) * inv,
                dexp(v[1] - row_max) * inv,
                dexp(v[2] - row_max) * inv,
                dexp(v[3] - row_max) * inv,
            ]);
        }
        k += 1;
    }
}

/// `log_softmax(x) = x - max - log(sum(exp(x - max)))`, same online statistics.
#[gpu::cuda_kernel]
pub fn log_softmax_kernel(x: &[Float4], y: &mut [Float4], cols4: u32, items: u32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out =
        chunk_mut(y, reshape_map!([items] | [ROW_BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]));
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut m = f32::MIN;
    let mut s = 0.0f32;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            let old = m;
            m = m.max(v[0]).max(v[1]).max(v[2]).max(v[3]);
            s *= dexp(old - m);
            s += dexp(v[0] - m) + dexp(v[1] - m) + dexp(v[2] - m) + dexp(v[3] - m);
        }
        k += 1;
    }

    let row_max = block_reduce_max(m);
    let row_sum = block_reduce_sum(s * dexp(m - row_max));
    let shift = row_max + dlog(row_sum);

    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            out[k] = Float4::new([v[0] - shift, v[1] - shift, v[2] - shift, v[3] - shift]);
        }
        k += 1;
    }
}

/// `sum` and `mean` along the last dimension. `scale` is `1` or `1/cols`.
#[gpu::cuda_kernel]
pub fn sum_dim_kernel(x: &[Float4], y: &mut [f32], cols4: u32, items: u32, scale: f32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    // One output per CTA: the thread dimension is clamped to 1, so only thread 0
    // has a valid slot and the map still proves disjointness.
    let mut out = chunk_mut(
        y,
        reshape_map!([1] | [(ROW_BLOCK, 1), grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut acc = 0.0f32;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            acc += v[0] + v[1] + v[2] + v[3];
        }
        k += 1;
    }
    let total = block_reduce_sum(acc);
    if tid == 0 {
        out[0] = total * scale;
    }
}

/// `max` along the last dimension.
#[gpu::cuda_kernel]
pub fn max_dim_kernel(x: &[Float4], y: &mut [f32], cols4: u32, items: u32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out = chunk_mut(
        y,
        reshape_map!([1] | [(ROW_BLOCK, 1), grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut acc = f32::MIN;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            acc = acc.max(v[0]).max(v[1]).max(v[2]).max(v[3]);
        }
        k += 1;
    }
    let total = block_reduce_max(acc);
    if tid == 0 {
        out[0] = total;
    }
}

/// `argmax` along the last dimension, returning the *smallest* index attaining
/// the maximum (PyTorch's tie-breaking rule).
///
/// Two block reductions: a float max, then an `i32` min over the indices that
/// match it, which maps onto the hardware `redux.sync.min.s32`.
#[gpu::cuda_kernel]
pub fn argmax_dim_kernel(x: &[Float4], y: &mut [i32], cols4: u32, items: u32, cols: u32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out = chunk_mut(
        y,
        reshape_map!([1] | [(ROW_BLOCK, 1), grid_dim::<DimX>()] => layout: [i0, t0, t1]),
    );
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut acc = f32::MIN;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            acc = acc.max(v[0]).max(v[1]).max(v[2]).max(v[3]);
        }
        k += 1;
    }
    let row_max = block_reduce_max(acc);

    let mut best = i32::MAX;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            unroll! {
                for j in 0..4 {
                    let idx = col * 4 + (j as u32);
                    if v[j] == row_max && idx < cols && (idx as i32) < best {
                        best = idx as i32;
                    }
                }
            }
        }
        k += 1;
    }
    let arg = block_reduce_min_i32(best);
    if tid == 0 {
        out[0] = arg;
    }
}


/// Inclusive cumulative sum along the last dimension.
///
/// Each thread owns four contiguous elements per tile (one `Float4`), scans them
/// in registers, and the per-thread totals are combined with a block-wide
/// exclusive scan. Tiles are carried by a running offset that thread 0 keeps in
/// its accumulator, so the row is processed in a single left-to-right sweep.
#[gpu::cuda_kernel]
pub fn cumsum_kernel(x: &[Float4], y: &mut [Float4], cols4: u32, items: u32) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out =
        chunk_mut(y, reshape_map!([items] | [ROW_BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]));
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let mut carry = 0.0f32;
    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        // Threads past the end of the row contribute zero but must still take
        // part in the scan, which is a block-wide (non-divergent) operation.
        let v = if col < cols4 { x[base + col as usize] } else { Float4::default() };
        let p0 = v[0];
        let p1 = p0 + v[1];
        let p2 = p1 + v[2];
        let p3 = p2 + v[3];
        let (prefix, total) = block_exclusive_scan(p3);
        let off = carry + prefix;
        if col < cols4 {
            out[k] = Float4::new([p0 + off, p1 + off, p2 + off, p3 + off]);
        }
        carry += total;
        k += 1;
    }
}

// ---------------------------------------------------------------------------
// Host drivers
// ---------------------------------------------------------------------------

/// Padded row stride and grid/tile counts for a `[rows, cols]` input.
pub struct RowPlan {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub items: u32,
    pub cols4: u32,
}

impl RowPlan {
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        assert!(cols % 4 == 0, "the vectorised row kernels need cols % 4 == 0");
        let stride = row_stride(cols);
        Self {
            rows,
            cols,
            stride,
            items: (stride / ROW_TILE) as u32,
            cols4: (cols / 4) as u32,
        }
    }
}

macro_rules! row_to_row_op {
    ($name:ident, $kernel:ident, $fill:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            let p = RowPlan::new(rows, cols);
            let padded = pad_rows(x, rows, cols, p.stride, $fill);
            let h4 = to_float4_padded(&padded, padded.len());
            let grid = rows as u32;
            let out = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
                let zeros = vec![Float4::default(); h4.len()];
                let mut d_y = ctx.new_tensor_view::<[Float4]>(&zeros).unwrap();
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                $kernel::launch(cfg, ctx, m, &d_x, &mut d_y, p.cols4, p.items).unwrap();
                let mut h_y = vec![Float4::default(); h4.len()];
                d_y.copy_to_host(&mut h_y).unwrap();
                from_float4(&h_y, h4.len() * 4)
            });
            unpad_rows(&out, rows, cols, p.stride)
        }
    };
}

row_to_row_op!(softmax, softmax_kernel, 0.0, "Row-wise softmax.");
row_to_row_op!(log_softmax, log_softmax_kernel, 0.0, "Row-wise log-softmax.");
row_to_row_op!(cumsum, cumsum_kernel, 0.0, "Row-wise inclusive cumulative sum.");

fn sum_dim_scaled(x: &[f32], rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    let p = RowPlan::new(rows, cols);
    let padded = pad_rows(x, rows, cols, p.stride, 0.0);
    let h4 = to_float4_padded(&padded, padded.len());
    let grid = rows as u32;
    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
        let zeros = vec![0.0f32; rows];
        let mut d_y = ctx.new_tensor_view::<[f32]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
        sum_dim_kernel::launch(cfg, ctx, m, &d_x, &mut d_y, p.cols4, p.items, scale).unwrap();
        let mut h_y = vec![0.0f32; rows];
        d_y.copy_to_host(&mut h_y).unwrap();
        h_y
    })
}

/// Sum along the last dimension: `[rows, cols] -> [rows]`.
pub fn sum_dim(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    sum_dim_scaled(x, rows, cols, 1.0)
}

/// Mean along the last dimension: `[rows, cols] -> [rows]`.
pub fn mean_dim(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    sum_dim_scaled(x, rows, cols, 1.0 / cols as f32)
}

/// Max along the last dimension: `[rows, cols] -> [rows]`.
pub fn max_dim(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let p = RowPlan::new(rows, cols);
    let padded = pad_rows(x, rows, cols, p.stride, f32::MIN);
    let h4 = to_float4_padded(&padded, padded.len());
    let grid = rows as u32;
    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
        let zeros = vec![0.0f32; rows];
        let mut d_y = ctx.new_tensor_view::<[f32]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
        max_dim_kernel::launch(cfg, ctx, m, &d_x, &mut d_y, p.cols4, p.items).unwrap();
        let mut h_y = vec![0.0f32; rows];
        d_y.copy_to_host(&mut h_y).unwrap();
        h_y
    })
}

/// Argmax along the last dimension: `[rows, cols] -> [rows]` (indices).
pub fn argmax_dim(x: &[f32], rows: usize, cols: usize) -> Vec<i32> {
    let p = RowPlan::new(rows, cols);
    let padded = pad_rows(x, rows, cols, p.stride, f32::MIN);
    let h4 = to_float4_padded(&padded, padded.len());
    let grid = rows as u32;
    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
        let zeros = vec![0i32; rows];
        let mut d_y = ctx.new_tensor_view::<[i32]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
        argmax_dim_kernel::launch(cfg, ctx, m, &d_x, &mut d_y, p.cols4, p.items, cols as u32)
            .unwrap();
        let mut h_y = vec![0i32; rows];
        d_y.copy_to_host(&mut h_y).unwrap();
        h_y
    })
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

pub fn softmax_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - m).exp() / s;
        }
    }
    out
}

pub fn log_softmax_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
        let shift = m + s.ln();
        for c in 0..cols {
            out[r * cols + c] = row[c] - shift;
        }
    }
    out
}

/// Accumulates in `f64`: the GPU scan sums in a different (tree) order, so the
/// reference has to be more accurate than either of them to be a fair oracle.
pub fn cumsum_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += x[r * cols + c] as f64;
            out[r * cols + c] = acc as f32;
        }
    }
    out
}

pub fn sum_dim_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| x[r * cols..(r + 1) * cols].iter().sum()).collect()
}

pub fn mean_dim_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    sum_dim_cpu(x, rows, cols).into_iter().map(|v| v / cols as f32).collect()
}

pub fn max_dim_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| x[r * cols..(r + 1) * cols].iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect()
}

pub fn argmax_dim_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| {
            let row = &x[r * cols..(r + 1) * cols];
            let mut best = 0usize;
            for c in 1..cols {
                if row[c] > row[best] {
                    best = c;
                }
            }
            best as i32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, sample};

    const SHAPES: &[(usize, usize)] = &[(1, 1024), (7, 1024), (33, 512), (5, 3000), (128, 4096)];

    #[test]
    fn softmax_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 3);
            assert_close(&softmax(&x, r, c), &softmax_cpu(&x, r, c), 1e-5, &format!("softmax {r}x{c}"));
        }
    }

    #[test]
    fn log_softmax_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 5);
            assert_close(
                &log_softmax(&x, r, c),
                &log_softmax_cpu(&x, r, c),
                1e-5,
                &format!("log_softmax {r}x{c}"),
            );
        }
    }

    #[test]
    fn sum_and_mean_match_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 9);
            assert_close(&sum_dim(&x, r, c), &sum_dim_cpu(&x, r, c), 1e-4, &format!("sum {r}x{c}"));
            assert_close(
                &mean_dim(&x, r, c),
                &mean_dim_cpu(&x, r, c),
                1e-4,
                &format!("mean {r}x{c}"),
            );
        }
    }

    #[test]
    fn max_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 13);
            assert_close(&max_dim(&x, r, c), &max_dim_cpu(&x, r, c), 1e-6, &format!("max {r}x{c}"));
        }
    }

    #[test]
    fn argmax_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 17);
            assert_eq!(argmax_dim(&x, r, c), argmax_dim_cpu(&x, r, c), "argmax {r}x{c}");
        }
    }

    #[test]
    fn cumsum_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 21);
            let got = cumsum(&x, r, c);
            let want = cumsum_cpu(&x, r, c);
            // A prefix sum cannot be more accurate than the magnitude of the
            // terms it accumulates, so the bound is relative to the row's L1
            // norm (1e-5 is about 80 f32 ulps of it).
            for r0 in 0..r {
                let l1: f32 = x[r0 * c..(r0 + 1) * c].iter().map(|v| v.abs()).sum();
                for c0 in 0..c {
                    let i = r0 * c + c0;
                    assert!(
                        (got[i] - want[i]).abs() <= 1e-5 * l1,
                        "cumsum {r}x{c} at {i}: gpu {} vs cpu {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }
}
