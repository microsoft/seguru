/// Barrett-reduction modular arithmetic mirroring HEonGPU's `OPERATOR_GPU_64`.
///
/// Every BFV kernel operates modulo one or more 64-bit primes. The [`Modulus64`]
/// type pre-computes the Barrett constant `mu` so that multiplication can avoid
/// expensive hardware division.

/// A 64-bit modulus with pre-computed Barrett reduction parameters.
///
/// Layout matches HEonGPU's `Modulus64`:
/// - `value`: the prime modulus
/// - `bit`:   floor(log2(value)) + 1
/// - `mu`:    floor(2^(2*bit+1) / value) — the Barrett constant
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Modulus64 {
    pub value: u64,
    pub bit: u64,
    pub mu: u64,
}

impl Modulus64 {
    /// Create a new modulus, computing `bit` and `mu` from `value`.
    ///
    /// # Panics
    /// Panics if `value < 2`.
    pub fn new(value: u64) -> Self {
        assert!(value >= 2, "modulus must be >= 2");
        let bit = (64 - value.leading_zeros()) as u64; // floor(log2(value)) + 1
        // mu = floor(2^(2*bit+1) / value)
        // Use u128 to avoid overflow.
        let numerator: u128 = 1u128 << (2 * bit + 1);
        let mu = (numerator / value as u128) as u64;
        Self { value, bit, mu }
    }
}

/// Barrett addition: `(a + b) mod modulus.value`.
///
/// Both `a` and `b` must already be in `[0, modulus.value)`.
#[inline]
pub fn mod_add(a: u64, b: u64, modulus: &Modulus64) -> u64 {
    let sum = a + b;
    if sum >= modulus.value { sum - modulus.value } else { sum }
}

/// Barrett subtraction: `(a - b) mod modulus.value`.
///
/// Both `a` and `b` must already be in `[0, modulus.value)`.
#[inline]
pub fn mod_sub(a: u64, b: u64, modulus: &Modulus64) -> u64 {
    let dif = a + modulus.value - b;
    if dif >= modulus.value { dif - modulus.value } else { dif }
}

/// Barrett multiplication: `(a * b) mod modulus.value`.
///
/// Uses 128-bit intermediates to avoid overflow.
/// Both `a` and `b` must already be in `[0, modulus.value)`.
#[inline]
pub fn mod_mul(a: u64, b: u64, modulus: &Modulus64) -> u64 {
    let z = (a as u128) * (b as u128);
    let bit = modulus.bit;
    let w = z >> (bit - 2);
    let w = (w * (modulus.mu as u128)) >> (bit + 3);
    let w = w * (modulus.value as u128);
    let mut r = (z - w) as u64;
    if r >= modulus.value {
        r -= modulus.value;
    }
    r
}

/// Barrett reduction of a single `u64` value.
///
/// `a` should be less than `2 * modulus.value` for a single-step reduction.
#[inline]
pub fn mod_reduce(a: u64, modulus: &Modulus64) -> u64 {
    let z = a as u128;
    let bit = modulus.bit;
    let w = z >> (bit - 2);
    let w = (w * (modulus.mu as u128)) >> (bit + 3);
    let w = w * (modulus.value as u128);
    let mut result = (z - w) as u64;
    if result >= modulus.value {
        result -= modulus.value;
    }
    result
}

/// Forced reduction: repeatedly reduce until `result < modulus.value`.
///
/// Works for any input value.
#[inline]
pub fn mod_reduce_forced(mut a: u64, modulus: &Modulus64) -> u64 {
    while a >= modulus.value {
        a = mod_reduce(a, modulus);
    }
    a
}

/// Centered reduction: map `a` in `[0, p)` to `[-p/2, p/2)`.
///
/// Values in `[0, (p-1)/2]` stay non-negative; values in `((p-1)/2, p)` become
/// negative.
#[inline]
pub fn centered_reduction(a: u64, modulus: &Modulus64) -> i64 {
    let half = (modulus.value - 1) / 2;
    if a <= half { a as i64 } else { a as i64 - modulus.value as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulus64_new() {
        // 17 = 0b10001 → bit_length = 5
        let m = Modulus64::new(17);
        assert_eq!(m.value, 17);
        assert_eq!(m.bit, 5); // floor(log2(17)) + 1 = 5
        // mu = floor(2^(2*5+1) / 17) = floor(2^11 / 17) = floor(2048/17) = 120
        assert_eq!(m.mu, 120);

        // Power of two edge: 16 = 0b10000 → bit = 5
        let m16 = Modulus64::new(16);
        assert_eq!(m16.bit, 5);
        assert_eq!(m16.mu, (1u128 << 11) as u64 / 16); // 2048/16 = 128

        // Large prime close to 2^60
        let p = 1152921504606846883u64;
        let m_large = Modulus64::new(p);
        assert_eq!(m_large.bit, 60);
        let expected_mu = ((1u128 << 121) / p as u128) as u64;
        assert_eq!(m_large.mu, expected_mu);
    }

    #[test]
    fn test_mod_add_basic() {
        let m = Modulus64::new(17);
        assert_eq!(mod_add(3, 4, &m), 7);
        assert_eq!(mod_add(15, 5, &m), 3); // 20 mod 17 = 3
        assert_eq!(mod_add(0, 0, &m), 0);
        assert_eq!(mod_add(16, 0, &m), 16);
        assert_eq!(mod_add(16, 1, &m), 0); // 17 mod 17 = 0
    }

    #[test]
    fn test_mod_sub_basic() {
        let m = Modulus64::new(17);
        assert_eq!(mod_sub(10, 3, &m), 7);
        assert_eq!(mod_sub(3, 10, &m), 10); // 3 - 10 ≡ -7 ≡ 10 (mod 17)
        assert_eq!(mod_sub(0, 0, &m), 0);
        assert_eq!(mod_sub(0, 1, &m), 16); // -1 mod 17 = 16
    }

    #[test]
    fn test_mod_mul_basic() {
        let m = Modulus64::new(17);
        assert_eq!(mod_mul(3, 4, &m), 12);
        assert_eq!(mod_mul(5, 5, &m), 8); // 25 mod 17 = 8
        assert_eq!(mod_mul(0, 12, &m), 0);
        assert_eq!(mod_mul(1, 16, &m), 16);
        assert_eq!(mod_mul(16, 16, &m), 1); // (-1)^2 mod 17
    }

    #[test]
    fn test_mod_mul_60bit_primes() {
        // Prime close to 2^60
        let p = 1152921504606846883u64;
        let m = Modulus64::new(p);

        // Verify a few hand-picked multiplications against u128 reference
        let pairs: [(u64, u64); 5] = [
            (1, p - 1),
            (p - 1, p - 1),
            (p / 2, 2),
            (123456789012345, 987654321098765),
            (p - 2, p - 3),
        ];
        for (a, b) in pairs {
            let expected = ((a as u128) * (b as u128) % (p as u128)) as u64;
            assert_eq!(mod_mul(a, b, &m), expected, "failed for a={a}, b={b}");
        }

        // Randomized test with deterministic seed via simple LCG
        let mut rng_state: u64 = 0xdeadbeef;
        for _ in 0..1000 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = rng_state % p;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = rng_state % p;
            let expected = ((a as u128) * (b as u128) % (p as u128)) as u64;
            assert_eq!(mod_mul(a, b, &m), expected, "failed for a={a}, b={b}");
        }
    }

    #[test]
    fn test_mod_reduce() {
        let m = Modulus64::new(17);
        assert_eq!(mod_reduce(22, &m), 5); // 17 + 5
        assert_eq!(mod_reduce(17, &m), 0);
        assert_eq!(mod_reduce(0, &m), 0);
        assert_eq!(mod_reduce(5, &m), 5);
    }

    #[test]
    fn test_mod_reduce_forced() {
        let m = Modulus64::new(17);
        assert_eq!(mod_reduce_forced(0, &m), 0);
        assert_eq!(mod_reduce_forced(5, &m), 5);
        assert_eq!(mod_reduce_forced(34, &m), 0); // 2 * 17
        assert_eq!(mod_reduce_forced(100, &m), 100 % 17);

        // Large value
        let p = 1152921504606846883u64;
        let m_large = Modulus64::new(p);
        assert_eq!(mod_reduce_forced(p + 5, &m_large), 5);
        assert_eq!(mod_reduce_forced(p * 2 + 7, &m_large), 7);
    }

    #[test]
    fn test_centered_reduction() {
        let m = Modulus64::new(17);
        assert_eq!(centered_reduction(0, &m), 0);
        assert_eq!(centered_reduction(8, &m), 8); // (17-1)/2 = 8 → stays positive
        assert_eq!(centered_reduction(16, &m), -1); // p-1 → -1
        assert_eq!(centered_reduction(9, &m), -8); // 9 > 8, so 9 - 17 = -8
        assert_eq!(centered_reduction(1, &m), 1);
    }
}
