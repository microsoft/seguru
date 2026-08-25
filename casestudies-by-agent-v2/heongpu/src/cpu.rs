//! Independent CPU reference implementations used as the correctness oracle.
//!
//! Nothing here shares code with the device path: modular reduction is done
//! with plain `u128` remainders rather than Barrett/Shoup, and the NTT is a
//! textbook O(N log N) loop. If the two agree, both the algebra and the GPU
//! index arithmetic are right.

/// `(a * b) mod q` with no precomputation.
pub fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    (((a as u128) * (b as u128)) % (q as u128)) as u64
}

pub fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 + b as u128) % q as u128) as u64
}

pub fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 + q as u128 - (b % q) as u128) % q as u128) as u64
}

pub fn neg_mod(a: u64, q: u64) -> u64 {
    ((q as u128 - (a % q) as u128) % q as u128) as u64
}

pub fn pow_mod(base: u64, mut e: u64, q: u64) -> u64 {
    let mut acc = 1u64;
    let mut b = base % q;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, b, q);
        }
        b = mul_mod(b, b, q);
        e >>= 1;
    }
    acc
}

pub fn poly_add(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    a.iter().zip(b).map(|(&x, &y)| add_mod(x, y, q)).collect()
}

pub fn poly_sub(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    a.iter().zip(b).map(|(&x, &y)| sub_mod(x, y, q)).collect()
}

pub fn poly_neg(a: &[u64], q: u64) -> Vec<u64> {
    a.iter().map(|&x| neg_mod(x, q)).collect()
}

pub fn poly_mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    a.iter().zip(b).map(|(&x, &y)| mul_mod(x, y, q)).collect()
}

pub fn poly_mul_scalar(a: &[u64], s: u64, q: u64) -> Vec<u64> {
    a.iter().map(|&x| mul_mod(x, s, q)).collect()
}

/// Ciphertext (two polynomials of length `n`) times one plaintext polynomial,
/// coefficient-wise in the NTT domain.
pub fn cipher_plain_mul(c: &[u64], p: &[u64], q: u64) -> Vec<u64> {
    let n = p.len();
    c.iter().enumerate().map(|(i, &x)| mul_mod(x, p[i % n], q)).collect()
}

/// Bit-reverse `x` within `bits` bits.
pub fn bit_reverse(x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for i in 0..bits {
        if x & (1 << i) != 0 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

/// Table of `psi^bitrev(i)` for `i in 0..n`, the twiddles used by the
/// Cooley-Tukey negacyclic forward transform.
pub fn psi_table(psi: u64, n: usize, q: u64) -> Vec<u64> {
    let bits = n.trailing_zeros();
    (0..n).map(|i| pow_mod(psi, bit_reverse(i, bits) as u64, q)).collect()
}

/// Forward negacyclic NTT: Cooley-Tukey decimation-in-time, natural-order
/// input, bit-reversed-order output. `psi_rev` is [`psi_table`].
pub fn ntt_forward(a: &[u64], psi_rev: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut a = a.to_vec();
    let mut t = n;
    let mut m = 1usize;
    while m < n {
        t /= 2;
        for i in 0..m {
            let j1 = 2 * i * t;
            let s = psi_rev[m + i];
            for j in j1..j1 + t {
                let u = a[j];
                let v = mul_mod(a[j + t], s, q);
                a[j] = add_mod(u, v, q);
                a[j + t] = sub_mod(u, v, q);
            }
        }
        m *= 2;
    }
    a
}

/// Inverse negacyclic NTT: Gentleman-Sande, bit-reversed-order input,
/// natural-order output, including the final `n^-1` scaling. `psi_inv_rev` is
/// [`psi_table`] built from `psi^-1`.
pub fn ntt_inverse(a: &[u64], psi_inv_rev: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut a = a.to_vec();
    let mut t = 1usize;
    let mut m = n;
    while m > 1 {
        let mut j1 = 0usize;
        let h = m / 2;
        for i in 0..h {
            let s = psi_inv_rev[h + i];
            for j in j1..j1 + t {
                let u = a[j];
                let v = a[j + t];
                a[j] = add_mod(u, v, q);
                a[j + t] = mul_mod(sub_mod(u, v, q), s, q);
            }
            j1 += 2 * t;
        }
        t *= 2;
        m /= 2;
    }
    let n_inv = pow_mod(n as u64, q - 2, q);
    a.iter().map(|&x| mul_mod(x, n_inv, q)).collect()
}

/// Schoolbook negacyclic convolution: `a * b mod (x^n + 1, q)`.
pub fn negacyclic_mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut out = vec![0u64; n];
    for i in 0..n {
        if a[i] == 0 {
            continue;
        }
        for j in 0..n {
            let p = mul_mod(a[i], b[j], q);
            let k = i + j;
            if k < n {
                out[k] = add_mod(out[k], p, q);
            } else {
                out[k - n] = sub_mod(out[k - n], p, q);
            }
        }
    }
    out
}
