//! Host-side helpers shared by every kernel module: deterministic data
//! generation, padding to the tile geometry the kernels assume, and the
//! relative-error comparison used by the tests.

/// Round `x` up to the next multiple of `m`.
pub fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// Deterministic pseudo-random values in `[-1, 1)`.
///
/// A tiny xorshift keeps the test data reproducible without pulling in a
/// dependency, and keeps the magnitudes uniform so that a relative-error
/// comparison against the CPU reference is meaningful.
pub fn seq(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}

/// Copy a `rows x cols` row-major matrix into a `prows x pcols` buffer, zero
/// filling the padding.
pub fn pad2(src: &[f32], rows: usize, cols: usize, prows: usize, pcols: usize) -> Vec<f32> {
    assert!(prows >= rows && pcols >= cols);
    let mut out = vec![0.0f32; prows * pcols];
    for r in 0..rows {
        out[r * pcols..r * pcols + cols].copy_from_slice(&src[r * cols..r * cols + cols]);
    }
    out
}

/// Inverse of [`pad2`].
pub fn unpad2(src: &[f32], rows: usize, cols: usize, pcols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        out[r * cols..r * cols + cols].copy_from_slice(&src[r * pcols..r * pcols + cols]);
    }
    out
}

/// Pad a 1-D buffer with zeros.
pub fn pad1(src: &[f32], plen: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; plen];
    out[..src.len()].copy_from_slice(src);
    out
}

/// Compare two buffers with a relative tolerance, falling back to an absolute
/// tolerance for values near zero (f32 accumulation order differs between the
/// GPU and the scalar CPU reference, so exact equality is not attainable).
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite value {g} at {i}");
        let err = (g - w).abs() / w.abs().max(1.0);
        if err > worst {
            worst = err;
            worst_at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: relative error {worst} at index {worst_at} (gpu={}, cpu={}) exceeds {tol}",
        got[worst_at],
        want[worst_at]
    );
}

/// Pack a length-multiple-of-4 buffer into `Float4`s for 128-bit device loads.
pub fn to_float4(v: &[f32]) -> Vec<gpu::Float4> {
    assert!(v.len() % 4 == 0);
    v.chunks_exact(4).map(|c| gpu::Float4::new([c[0], c[1], c[2], c[3]])).collect()
}
