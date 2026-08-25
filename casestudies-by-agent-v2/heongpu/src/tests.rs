use crate::arith::{BinOp, run_binary, run_cipher_plain_mul, run_neg, run_scalar_mul};
use crate::modular::{DEFAULT_Q, Modulus};
use crate::ntt::{NttTables, forward, inverse};
use crate::{cpu, modular};

/// xorshift stream, so tests are deterministic without a dependency.
fn sample(n: usize, seed: u64, q: u64) -> Vec<u64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % q
        })
        .collect()
}

const SIZES: [usize; 5] = [4096, 8192, 16384, 32768, 65536];

#[test]
fn modulus_constants_are_consistent() {
    let m = Modulus::new(DEFAULT_Q);
    assert_eq!(m.bit, 59);
    assert_eq!(m.mu, ((1u128 << (2 * 59 + 1)) / m.q as u128) as u64);
    assert_eq!(m.shoup(1), ((1u128 << 64) / m.q as u128) as u64);
    assert_eq!(cpu::mul_mod(m.inv(12345), 12345, m.q), 1);
    assert!((m.q - 1) % (1 << 17) == 0, "q must support N up to 65536");
}

#[test]
fn barrett_and_shoup_match_reference() {
    let m = Modulus::new(DEFAULT_Q);
    let a = sample(2000, 12345, m.q);
    let b = sample(2000, 999, m.q);
    for i in 0..a.len() {
        assert_eq!(modular::mul_mod(a[i], b[i], m.q, m.mu, m.bit), cpu::mul_mod(a[i], b[i], m.q));
        assert_eq!(
            modular::mul_mod_shoup(a[i], b[i], m.shoup(b[i]), m.q),
            cpu::mul_mod(a[i], b[i], m.q)
        );
        assert_eq!(modular::add_mod(a[i], b[i], m.q), cpu::add_mod(a[i], b[i], m.q));
        assert_eq!(modular::sub_mod(a[i], b[i], m.q), cpu::sub_mod(a[i], b[i], m.q));
        assert_eq!(modular::neg_mod(a[i], m.q), cpu::neg_mod(a[i], m.q));
        assert_eq!(modular::mul_hi(a[i], b[i]), (((a[i] as u128) * b[i] as u128) >> 64) as u64);
    }
    // Edge cases at the ends of the range.
    for &x in &[0u64, 1, m.q - 1] {
        for &y in &[0u64, 1, m.q - 1] {
            assert_eq!(modular::mul_mod(x, y, m.q, m.mu, m.bit), cpu::mul_mod(x, y, m.q));
            assert_eq!(modular::mul_mod_shoup(x, y, m.shoup(y), m.q), cpu::mul_mod(x, y, m.q));
            assert_eq!(modular::add_mod(x, y, m.q), cpu::add_mod(x, y, m.q));
            assert_eq!(modular::sub_mod(x, y, m.q), cpu::sub_mod(x, y, m.q));
        }
    }
}

#[test]
fn gpu_elementwise_matches_cpu() {
    let m = Modulus::new(DEFAULT_Q);
    // A sub-tile length, an exact tile and a ragged length exercising padding.
    for &n in &[1024usize, 4096, 3 * 4096 + 17] {
        let a = sample(n, 7, m.q);
        let b = sample(n, 11, m.q);
        assert_eq!(run_binary(&a, &b, BinOp::Add, &m), cpu::poly_add(&a, &b, m.q), "add {n}");
        assert_eq!(run_binary(&a, &b, BinOp::Sub, &m), cpu::poly_sub(&a, &b, m.q), "sub {n}");
        assert_eq!(run_binary(&a, &b, BinOp::Mul, &m), cpu::poly_mul(&a, &b, m.q), "mul {n}");
        assert_eq!(run_neg(&a, &m), cpu::poly_neg(&a, m.q), "neg {n}");
        let s = a[0].max(1);
        assert_eq!(run_scalar_mul(&a, s, &m), cpu::poly_mul_scalar(&a, s, m.q), "scalar {n}");
    }
}

#[test]
fn gpu_cipher_plain_mul_matches_cpu() {
    let m = Modulus::new(DEFAULT_Q);
    let n = 4096;
    let c = sample(2 * n, 5, m.q);
    let p = sample(n, 6, m.q);
    assert_eq!(run_cipher_plain_mul(&c, &p, &m), cpu::cipher_plain_mul(&c, &p, m.q));
}

#[test]
fn gpu_ciphertext_add_sub_negate() {
    let m = Modulus::new(DEFAULT_Q);
    let n = 8192;
    let c0 = sample(2 * n, 21, m.q);
    let c1 = sample(2 * n, 22, m.q);
    let sum = run_binary(&c0, &c1, BinOp::Add, &m);
    let back = run_binary(&sum, &c1, BinOp::Sub, &m);
    assert_eq!(back, c0);
    let neg = run_neg(&c0, &m);
    assert_eq!(run_binary(&c0, &neg, BinOp::Add, &m), vec![0u64; 2 * n]);
}

#[test]
fn gpu_forward_ntt_matches_cpu() {
    let m = Modulus::new(DEFAULT_Q);
    for &n in &SIZES {
        let t = NttTables::new(n, m);
        let a = sample(n, n as u64 + 3, m.q);
        assert_eq!(forward(&a, &t, 1), cpu::ntt_forward(&a, &t.w_fwd, m.q), "forward N={n}");
    }
}

#[test]
fn gpu_inverse_ntt_matches_cpu() {
    let m = Modulus::new(DEFAULT_Q);
    for &n in &SIZES {
        let t = NttTables::new(n, m);
        let a = sample(n, n as u64 + 9, m.q);
        assert_eq!(inverse(&a, &t, 1), cpu::ntt_inverse(&a, &t.w_inv, m.q), "inverse N={n}");
    }
}

#[test]
fn ntt_roundtrip_is_identity() {
    let m = Modulus::new(DEFAULT_Q);
    for &n in &SIZES {
        let t = NttTables::new(n, m);
        let a = sample(n, 1234 + n as u64, m.q);
        let f = forward(&a, &t, 1);
        assert_eq!(inverse(&f, &t, 1), a, "roundtrip N={n}");
    }
}

#[test]
fn ntt_batched_matches_single() {
    let m = Modulus::new(DEFAULT_Q);
    let n = 8192;
    let batch = 5;
    let t = NttTables::new(n, m);
    let all = sample(n * batch, 77, m.q);
    let got = forward(&all, &t, batch);
    for b in 0..batch {
        let one = forward(&all[b * n..(b + 1) * n], &t, 1);
        assert_eq!(&got[b * n..(b + 1) * n], one.as_slice(), "batch slot {b}");
    }
    assert_eq!(inverse(&got, &t, batch), all);
}

#[test]
fn negacyclic_convolution_matches_schoolbook() {
    let m = Modulus::new(DEFAULT_Q);
    let n = 4096;
    let t = NttTables::new(n, m);
    // Sparse-ish operands keep the O(N^2) oracle fast while still covering the
    // wrap-around sign flip of x^N = -1.
    let mut a = sample(n, 31, m.q);
    let mut b = sample(n, 41, m.q);
    for i in 0..n {
        if i % 8 != 0 {
            a[i] = 0;
        }
        if i % 4 != 0 {
            b[i] = 0;
        }
    }
    let fa = forward(&a, &t, 1);
    let fb = forward(&b, &t, 1);
    let prod = run_binary(&fa, &fb, BinOp::Mul, &m);
    let got = inverse(&prod, &t, 1);
    assert_eq!(got, cpu::negacyclic_mul(&a, &b, m.q));
}

#[test]
fn cpu_reference_ntt_roundtrips() {
    let m = Modulus::new(DEFAULT_Q);
    let t = NttTables::new(4096, m);
    let a = sample(t.n, 8, m.q);
    assert_eq!(cpu::ntt_inverse(&cpu::ntt_forward(&a, &t.w_fwd, m.q), &t.w_inv, m.q), a);
    // psi is a primitive 2N-th root of unity.
    assert_eq!(cpu::pow_mod(t.psi, t.n as u64, m.q), m.q - 1);
}
