//! Test-only helpers: deterministic input generation and float comparison.

/// Deterministic pseudo-random values in roughly `[-3, 3]`, so the transcendental
/// activations are exercised over their interesting range without ever
/// overflowing `f32`.
pub fn sample(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) * 6.0 - 3.0
        })
        .collect()
}

/// Positive samples in `(0.1, 3.1]`, for operators that divide by a norm.
pub fn sample_positive(n: usize, seed: u64) -> Vec<f32> {
    sample(n, seed).into_iter().map(|v| v.abs() * 0.5 + 0.1).collect()
}

/// Compare two buffers with a mixed relative/absolute tolerance.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol * denom,
            "{ctx}: index {i}: gpu {g} vs cpu {w} (tol {tol})"
        );
    }
}
