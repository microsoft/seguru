//! Host-side helpers shared by every operator module.
//!
//! All conversions between `f32` and [`Float4`] copy element by element, never
//! reinterpret a pointer, so the crate stays free of `unsafe`.

use gpu::Float4;

/// Threads per block used by the elementwise kernels.
pub const EW_BLOCK: u32 = 256;
/// `Float4`s handled by one thread in the elementwise kernels.
pub const VEC_PER_THREAD: u32 = 4;
/// `f32` elements consumed by one elementwise CTA.
pub const ELEMS_PER_CTA: usize = (EW_BLOCK * VEC_PER_THREAD * 4) as usize;

/// Threads per block used by the row-reduction kernels.
pub const ROW_BLOCK: u32 = 256;
/// `f32` elements a row-reduction CTA covers in one pass over a row.
pub const ROW_TILE: usize = (ROW_BLOCK * 4) as usize;

pub fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// Pack `x` into `Float4`s, zero-padding to `padded` elements.
pub fn to_float4_padded(x: &[f32], padded: usize) -> Vec<Float4> {
    assert!(padded % 4 == 0 && padded >= x.len());
    let mut out = Vec::with_capacity(padded / 4);
    let full = x.len() / 4;
    for c in x[..full * 4].chunks_exact(4) {
        out.push(Float4::new([c[0], c[1], c[2], c[3]]));
    }
    if x.len() % 4 != 0 {
        let mut tail = [0.0f32; 4];
        for (i, v) in x[full * 4..].iter().enumerate() {
            tail[i] = *v;
        }
        out.push(Float4::new(tail));
    }
    out.resize(padded / 4, Float4::default());
    out
}

/// Unpack `Float4`s back to `f32`, truncating to `n` elements.
pub fn from_float4(v: &[Float4], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for e in v {
        for k in 0..4 {
            out.push(e[k]);
        }
    }
    out.truncate(n);
    out
}

/// Lay `rows x cols` out with every row padded to `stride` columns using `fill`.
pub fn pad_rows(x: &[f32], rows: usize, cols: usize, stride: usize, fill: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    let mut out = vec![fill; rows * stride];
    for r in 0..rows {
        out[r * stride..r * stride + cols].copy_from_slice(&x[r * cols..(r + 1) * cols]);
    }
    out
}

/// Inverse of [`pad_rows`].
pub fn unpad_rows(x: &[f32], rows: usize, cols: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend_from_slice(&x[r * stride..r * stride + cols]);
    }
    out
}

/// Row stride used by the reduction kernels: a whole number of CTA tiles, so
/// the `reshape_map!` covering the output maps exactly onto the buffer.
pub fn row_stride(cols: usize) -> usize {
    round_up(cols, ROW_TILE).max(ROW_TILE)
}
