//! 64-bit modular arithmetic modulo a word-size prime.
//!
//! Two multiplication strategies are provided, both of which are what a real
//! RNS/NTT homomorphic-encryption backend uses:
//!
//! * [`mul_mod`] — Barrett reduction with the precomputed constant
//!   `mu = floor(2^(2*bit+1) / q)` (the formulation used by HEonGPU's
//!   `OPERATOR_GPU_64`). Needed when both operands are runtime values.
//! * [`mul_mod_shoup`] — Shoup multiplication by a *precomputed* operand `w`
//!   carrying `w' = floor(w * 2^64 / q)`. Two `mul` / `mulhi` pairs and one
//!   conditional subtraction; this is what the NTT butterflies use, because the
//!   twiddle factor is known ahead of time.
//!
//! Both need a 64x64 -> 128 multiply. SeGuRu lowers Rust `u128` operations to
//! PTX `mul.hi.u64` / `mul.lo.u64`, so the natural `(a as u128) * (b as u128)`
//! spelling works in device code; `mul_hi` below is the only place that detail
//! is relied upon.


/// A prime modulus together with its Barrett constant.
///
/// `q` must be at least 2 and smaller than `2^62` so that every Barrett
/// intermediate fits in 128 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Modulus {
    /// The modulus itself.
    pub q: u64,
    /// `floor(log2(q)) + 1`.
    pub bit: u32,
    /// `floor(2^(2*bit+1) / q)`.
    pub mu: u64,
}

impl Modulus {
    pub fn new(q: u64) -> Self {
        assert!(q >= 2, "modulus must be >= 2");
        assert!(q < (1u64 << 62), "modulus must be < 2^62 for Barrett reduction");
        let bit = 64 - q.leading_zeros();
        let mu = ((1u128 << (2 * bit + 1)) / q as u128) as u64;
        Self { q, bit, mu }
    }

    /// `floor(w * 2^64 / q)`, the precomputed constant for [`mul_mod_shoup`].
    pub fn shoup(&self, w: u64) -> u64 {
        (((w as u128) << 64) / self.q as u128) as u64
    }

    /// `base^exp mod q` (host side; used to build twiddle tables).
    pub fn pow(&self, base: u64, mut exp: u64) -> u64 {
        let mut acc = 1u64;
        let mut b = base % self.q;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = (((acc as u128) * (b as u128)) % self.q as u128) as u64;
            }
            b = (((b as u128) * (b as u128)) % self.q as u128) as u64;
            exp >>= 1;
        }
        acc
    }

    /// Modular inverse via Fermat's little theorem (`q` prime).
    pub fn inv(&self, a: u64) -> u64 {
        assert!(a % self.q != 0, "zero has no inverse");
        self.pow(a, self.q - 2)
    }
}

/// The default 59-bit NTT-friendly prime used throughout the crate.
///
/// `q - 1` is divisible by `2^17`, so a primitive `2N`-th root of unity exists
/// for every supported ring size up to `N = 65536`.
pub const DEFAULT_Q: u64 = 576460752300015617;

/// High 64 bits of the 128-bit product `a * b`.
#[gpu::device]
#[inline(always)]
pub fn mul_hi(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 64) as u64
}

/// `(a + b) mod q`, for `a, b < q`.
#[gpu::device]
#[inline(always)]
pub fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q { s - q } else { s }
}

/// `(a - b) mod q`, for `a, b < q`.
#[gpu::device]
#[inline(always)]
pub fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

/// `(-a) mod q`, for `a < q`.
#[gpu::device]
#[inline(always)]
pub fn neg_mod(a: u64, q: u64) -> u64 {
    if a == 0 { 0 } else { q - a }
}

/// `(a * b) mod q` by Barrett reduction, for `a, b < q < 2^62`.
#[gpu::device]
#[inline(always)]
pub fn mul_mod(a: u64, b: u64, q: u64, mu: u64, bit: u32) -> u64 {
    let z = (a as u128) * (b as u128);
    let w = z >> (bit - 2);
    let w = (w * (mu as u128)) >> (bit + 3);
    let r = (z - w * (q as u128)) as u64;
    if r >= q { r - q } else { r }
}

/// `(a * w) mod q` where `w_shoup = floor(w * 2^64 / q)`, for `a < q < 2^63`.
#[gpu::device]
#[inline(always)]
pub fn mul_mod_shoup(a: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
    let hi = mul_hi(a, w_shoup);
    let r = a.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q));
    if r >= q { r - q } else { r }
}
