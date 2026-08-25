//! Row-wise normalisations: layer norm, RMS norm, L1 and L2 normalisation.
//!
//! Same shape as [`crate::reduce`]: one CTA of [`ROW_BLOCK`] threads per row,
//! `Float4` accesses, and warp-shuffle block reductions. `layer_norm` makes two
//! statistics passes over the row (mean, then variance about that mean) rather
//! than accumulating `sum` and `sum of squares` in one pass, because the
//! `E[x^2] - E[x]^2` form loses most of its significant digits when the row mean
//! is large compared with its standard deviation. Rows are at most a few KiB, so
//! the second pass is an L2 hit.

use gpu::*;

use crate::activation::drsqrt;
use crate::device::block_reduce_sum;
use crate::util::{ROW_BLOCK, from_float4, pad_rows, to_float4_padded, unpad_rows};

macro_rules! row_stat_loop {
    ($x:ident, $base:ident, $items:ident, $cols4:ident, $tid:ident, $acc:ident, $v:ident, $upd:expr) => {{
        let mut $acc = 0.0f32;
        let mut k = 0u32;
        while k < $items {
            let col = k * Config::BDIM_X + $tid;
            if col < $cols4 {
                let $v = $x[$base + col as usize];
                $upd;
            }
            k += 1;
        }
        block_reduce_sum($acc)
    }};
}

/// `layer_norm(x) = (x - mean) * rstd * weight + bias`, per row.
#[gpu::cuda_kernel]
pub fn layer_norm_kernel(
    x: &[Float4],
    weight: &[Float4],
    bias: &[Float4],
    y: &mut [Float4],
    cols4: u32,
    items: u32,
    inv_n: f32,
    eps: f32,
) {
    assert!(Config::BDIM_X == ROW_BLOCK);
    let tid = thread_id::<DimX>();
    let mut out =
        chunk_mut(y, reshape_map!([items] | [ROW_BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]));
    let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

    let sum = row_stat_loop!(x, base, items, cols4, tid, acc, v, {
        acc += v[0] + v[1] + v[2] + v[3]
    });
    let mean = sum * inv_n;
    let var = row_stat_loop!(x, base, items, cols4, tid, acc, v, {
        let d0 = v[0] - mean;
        let d1 = v[1] - mean;
        let d2 = v[2] - mean;
        let d3 = v[3] - mean;
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
    }) * inv_n;
    let rstd = drsqrt(var + eps);

    let mut k = 0u32;
    while k < items {
        let col = k * Config::BDIM_X + tid;
        if col < cols4 {
            let v = x[base + col as usize];
            let w = weight[col as usize];
            let b = bias[col as usize];
            out[k] = Float4::new([
                (v[0] - mean) * rstd * w[0] + b[0],
                (v[1] - mean) * rstd * w[1] + b[1],
                (v[2] - mean) * rstd * w[2] + b[2],
                (v[3] - mean) * rstd * w[3] + b[3],
            ]);
        }
        k += 1;
    }
}

/// A one-statistic-pass row scaling: `y = x * f(stat)`.
///
/// `rms_norm`, `l1_norm` and `l2_norm` differ only in what is accumulated and
/// how the accumulator becomes a scale, so they share one kernel body.
macro_rules! scale_kernel {
    ($kernel:ident, $v:ident, $acc:ident, $upd:expr, $total:ident, $inv_n:ident, $eps:ident, $scale:expr, $doc:literal) => {
        #[doc = $doc]
        #[gpu::cuda_kernel]
        pub fn $kernel(x: &[Float4], y: &mut [Float4], cols4: u32, items: u32, $inv_n: f32, $eps: f32) {
            assert!(Config::BDIM_X == ROW_BLOCK);
            let tid = thread_id::<DimX>();
            let mut out = chunk_mut(
                y,
                reshape_map!([items] | [ROW_BLOCK, grid_dim::<DimX>()] => layout: [t0, i0, t1]),
            );
            let base = (block_id::<DimX>() * items * Config::BDIM_X) as usize;

            let $total = row_stat_loop!(x, base, items, cols4, tid, $acc, $v, $upd);
            let scale = $scale;

            let mut k = 0u32;
            while k < items {
                let col = k * Config::BDIM_X + tid;
                if col < cols4 {
                    let v = x[base + col as usize];
                    out[k] =
                        Float4::new([v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale]);
                }
                k += 1;
            }
        }
    };
}

scale_kernel!(
    rms_norm_kernel, v, acc,
    { acc += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] },
    total, inv_n, eps,
    drsqrt(total * inv_n + eps),
    "`rms_norm(x) = x / sqrt(mean(x^2) + eps)`, per row."
);

scale_kernel!(
    l1_norm_kernel, v, acc,
    { acc += v[0].max(-v[0]) + v[1].max(-v[1]) + v[2].max(-v[2]) + v[3].max(-v[3]) },
    total, inv_n, eps,
    inv_n / (total + eps),
    "`l1_norm(x) = x / sum(|x|)`, per row (`inv_n` is 1 here)."
);

scale_kernel!(
    l2_norm_kernel, v, acc,
    { acc += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] },
    total, inv_n, eps,
    inv_n * drsqrt(total + eps),
    "`l2_norm(x) = x / sqrt(sum(x^2))`, per row (`inv_n` is 1 here)."
);

// ---------------------------------------------------------------------------
// Host drivers
// ---------------------------------------------------------------------------

/// `layer_norm` over the last dimension with per-column `weight` and `bias`.
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let p = crate::reduce::RowPlan::new(rows, cols);
    assert_eq!(weight.len(), cols);
    assert_eq!(bias.len(), cols);
    let padded = pad_rows(x, rows, cols, p.stride, 0.0);
    let h4 = to_float4_padded(&padded, padded.len());
    let hw = to_float4_padded(weight, p.stride);
    let hb = to_float4_padded(bias, p.stride);
    let grid = rows as u32;
    let inv_n = 1.0 / cols as f32;
    let out = gpu_host::cuda_ctx(0, |ctx, m| {
        let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
        let d_w = ctx.new_tensor_view::<[Float4]>(&hw).unwrap();
        let d_b = ctx.new_tensor_view::<[Float4]>(&hb).unwrap();
        let zeros = vec![Float4::default(); h4.len()];
        let mut d_y = ctx.new_tensor_view::<[Float4]>(&zeros).unwrap();
        let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
        layer_norm_kernel::launch(cfg, ctx, m, &d_x, &d_w, &d_b, &mut d_y, p.cols4, p.items, inv_n, eps)
            .unwrap();
        let mut h_y = vec![Float4::default(); h4.len()];
        d_y.copy_to_host(&mut h_y).unwrap();
        from_float4(&h_y, h4.len() * 4)
    });
    unpad_rows(&out, rows, cols, p.stride)
}

macro_rules! scale_host {
    ($name:ident, $kernel:ident, $inv_n:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
            let p = crate::reduce::RowPlan::new(rows, cols);
            let padded = pad_rows(x, rows, cols, p.stride, 0.0);
            let h4 = to_float4_padded(&padded, padded.len());
            let grid = rows as u32;
            #[allow(clippy::redundant_closure_call)]
            let inv_n: f32 = ($inv_n)(cols);
            let out = gpu_host::cuda_ctx(0, |ctx, m| {
                let d_x = ctx.new_tensor_view::<[Float4]>(&h4).unwrap();
                let zeros = vec![Float4::default(); h4.len()];
                let mut d_y = ctx.new_tensor_view::<[Float4]>(&zeros).unwrap();
                let cfg = gpu_host::gpu_config!(grid, 1, 1, @const ROW_BLOCK, 1, 1, 0);
                $kernel::launch(cfg, ctx, m, &d_x, &mut d_y, p.cols4, p.items, inv_n, eps).unwrap();
                let mut h_y = vec![Float4::default(); h4.len()];
                d_y.copy_to_host(&mut h_y).unwrap();
                from_float4(&h_y, h4.len() * 4)
            });
            unpad_rows(&out, rows, cols, p.stride)
        }
    };
}

scale_host!(rms_norm, rms_norm_kernel, |c: usize| 1.0 / c as f32, "`x / sqrt(mean(x^2) + eps)` per row.");
scale_host!(l1_norm, l1_norm_kernel, |_c: usize| 1.0f32, "`x / sum(|x|)` per row.");
scale_host!(l2_norm, l2_norm_kernel, |_c: usize| 1.0f32, "`x / sqrt(sum(x^2))` per row.");

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

pub fn layer_norm_cpu(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var = row.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>()
            / cols as f64;
        let rstd = 1.0 / (var + eps as f64).sqrt();
        for c in 0..cols {
            out[r * cols + c] =
                (((row[c] as f64 - mean) * rstd) as f32) * weight[c] + bias[c];
        }
    }
    out
}

pub fn rms_norm_cpu(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let ms = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cols as f64;
        let scale = 1.0 / (ms + eps as f64).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] as f64 * scale) as f32;
        }
    }
    out
}

pub fn l1_norm_cpu(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let s = row.iter().map(|v| v.abs() as f64).sum::<f64>() + eps as f64;
        for c in 0..cols {
            out[r * cols + c] = (row[c] as f64 / s) as f32;
        }
    }
    out
}

pub fn l2_norm_cpu(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let s = (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() + eps as f64).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] as f64 / s) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, sample};

    const SHAPES: &[(usize, usize)] = &[(1, 1024), (7, 1024), (33, 512), (5, 3000), (128, 4096)];
    const EPS: f32 = 1e-5;

    #[test]
    fn layer_norm_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 31);
            let w = sample(c, 32);
            let b = sample(c, 33);
            assert_close(
                &layer_norm(&x, r, c, &w, &b, EPS),
                &layer_norm_cpu(&x, r, c, &w, &b, EPS),
                1e-4,
                &format!("layer_norm {r}x{c}"),
            );
        }
    }

    #[test]
    fn layer_norm_handles_large_offsets() {
        // The two-pass formulation exists for exactly this case: a row whose
        // mean dwarfs its spread.
        let (r, c) = (4usize, 1024usize);
        let x: Vec<f32> = sample(r * c, 34).iter().map(|v| v * 0.01 + 1000.0).collect();
        let w = vec![1.0f32; c];
        let b = vec![0.0f32; c];
        assert_close(
            &layer_norm(&x, r, c, &w, &b, EPS),
            &layer_norm_cpu(&x, r, c, &w, &b, EPS),
            2e-2,
            "layer_norm offset",
        );
    }

    #[test]
    fn rms_norm_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 41);
            assert_close(
                &rms_norm(&x, r, c, EPS),
                &rms_norm_cpu(&x, r, c, EPS),
                1e-4,
                &format!("rms_norm {r}x{c}"),
            );
        }
    }

    #[test]
    fn l1_norm_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 43);
            assert_close(
                &l1_norm(&x, r, c, 0.0),
                &l1_norm_cpu(&x, r, c, 0.0),
                1e-4,
                &format!("l1_norm {r}x{c}"),
            );
        }
    }

    #[test]
    fn l2_norm_matches_cpu() {
        for &(r, c) in SHAPES {
            let x = sample(r * c, 47);
            assert_close(
                &l2_norm(&x, r, c, 0.0),
                &l2_norm_cpu(&x, r, c, 0.0),
                1e-4,
                &format!("l2_norm {r}x{c}"),
            );
        }
    }
}
