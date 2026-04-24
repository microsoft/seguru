/// Key generation kernels ported from HEonGPU's `keygeneration.cu`.
///
/// This module implements:
/// - Secret key generation (ternary distribution, RNS conversion)
/// - Public key generation (pk = -(sk*a + e), a)
/// - Relin key / Galois key / Switch key generation
/// - Multi-party key generation variants
/// - TFHE-specific key generation (bootstrapping keys, switch keys)
///
/// # GPU kernel availability
///
/// Only simple 1-output-per-thread kernels are ported as `#[gpu::cuda_kernel]`.
/// Kernels that write multiple outputs per thread, use curand, NTT, or shared
/// memory are provided as CPU reference implementations or stubs.

use crate::modular::{mod_add, mod_mul, mod_sub, mod_reduce_forced, Modulus64};
use gpu::prelude::*;

// ===========================================================================
// Helper functions (CPU)
// ===========================================================================

/// Convert a rotation step to the corresponding Galois element.
///
/// Mirrors the CUDA `__host__` function `steps_to_galois_elt`.
/// `group_order` is typically `2 * coeff_count`.
pub fn steps_to_galois_elt(steps: i32, coeff_count: i32, group_order: i32) -> i32 {
    if steps == 0 {
        return group_order - 1; // column rotation element
    }
    let s = if steps < 0 {
        // Negative rotation: use inverse element
        let s = (-steps) % (coeff_count / 2);
        (coeff_count / 2) - s
    } else {
        steps % (coeff_count / 2)
    };
    // Galois element = 3^s mod group_order
    let mut elt: i64 = 1;
    let m = group_order as i64;
    for _ in 0..s {
        elt = (elt * 3) % m;
    }
    elt as i32
}

/// Bit-reverse an index of `n_power` bits.
///
/// Mirrors the CUDA `__device__` function `bitreverse`.
pub fn bitreverse(index: i32, n_power: i32) -> i32 {
    let mut result = 0i32;
    let mut idx = index;
    for _ in 0..n_power {
        result = (result << 1) | (idx & 1);
        idx >>= 1;
    }
    result
}

/// Compute the permutation index for Galois automorphism.
///
/// Given a coefficient index, the Galois element, and the ring parameters,
/// returns the destination index after applying the automorphism
/// `x -> x^{galois_elt}` on Z[x]/(x^N+1).
///
/// Mirrors the CUDA `__device__` function `permutation`.
pub fn permutation(index: i32, galois_elt: i32, coeff_count: i32, n_power: i32) -> i32 {
    let rev = bitreverse(index, n_power);
    let mapped = (((rev as i64) * (galois_elt as i64)) % ((2 * coeff_count) as i64)) as i32;
    bitreverse(
        if mapped < 0 { mapped + 2 * coeff_count } else { mapped },
        n_power,
    )
}

// ===========================================================================
// CPU reference: secretkey_gen_kernel_v2
// ===========================================================================

/// Generate a ternary secret key from sparse positions and values.
///
/// `sk` is zeroed, then for each `i in 0..hw`, `sk[positions[i]] = values[i]`.
/// In the CUDA kernel each thread handles one index in `[0, 1<<n_power)` and
/// loops over hw to check if its index matches a position.
pub fn secretkey_gen_v2_cpu(
    sk: &mut [i32],
    positions: &[i32],
    values: &[i32],
    hw: usize,
    n_power: u32,
) {
    let n = 1usize << n_power;
    for v in sk[..n].iter_mut() {
        *v = 0;
    }
    for i in 0..hw {
        let pos = positions[i] as usize;
        if pos < n {
            sk[pos] = values[i];
        }
    }
}

// ===========================================================================
// GPU kernel: secretkey_gen_v2_kernel
// ===========================================================================

/// GPU kernel: ternary secret key generation.
///
/// Each thread handles one coefficient index. It scans the sparse (positions,
/// values) arrays of length `hw` and writes the matching value (or 0).
///
/// Launch: 1D grid over `1 << n_power` elements.
#[gpu::cuda_kernel]
pub fn secretkey_gen_v2_kernel(
    positions: &[i32],
    values: &[i32],
    sk: &mut [i32],
    hw: u32,
    n_power: u32,
) {
    let gid = (block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>()) as usize;
    let n = 1usize << (n_power as usize);
    if gid >= n {
        return;
    }

    let mut val = 0i32;
    let mut i = 0u32;
    while i < hw {
        if positions[i as usize] as usize == gid {
            val = values[i as usize];
        }
        i += 1;
    }

    let mut out = chunk_mut(sk, MapLinear::new(1));
    out[0] = val;
}

// ===========================================================================
// CPU reference: secretkey_rns_kernel
// ===========================================================================

/// Convert integer secret key to RNS representation.
///
/// For each RNS modulus `j` and coefficient `i`, compute
/// `output[i + (j << n_power)] = sk[i] mod modulus[j]`.
/// Negative coefficients are mapped to their positive representative.
pub fn secretkey_rns_cpu(
    input: &[i32],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_mod_count: usize,
) {
    let n = 1usize << n_power;
    for j in 0..rns_mod_count {
        let m = &moduli[j];
        for i in 0..n {
            let sk_val = input[i];
            let reduced = if sk_val < 0 {
                // Negative: add modulus
                mod_reduce_forced((m.value as i64 + sk_val as i64) as u64, m)
            } else {
                mod_reduce_forced(sk_val as u64, m)
            };
            output[i + (j << n_power)] = reduced;
        }
    }
}

// ===========================================================================
// CPU reference: publickey_gen_kernel
// ===========================================================================

/// Public key generation: pk0 = -(sk * a + e) mod q, pk1 = a.
///
/// `pk` layout: ring_size * rns_count * 2 (pk0 then pk1).
/// `sk`, `error`, `a` layout: ring_size * rns_count.
pub fn publickey_gen_cpu(
    pk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for j in 0..rns_count {
        let m = &moduli[j];
        for i in 0..ring_size {
            let loc = i + (j << n_power);
            // pk0 = -(sk * a + e) mod q
            let sa = mod_mul(sk[loc], a[loc], m);
            let sa_e = mod_add(sa, error[loc], m);
            pk[loc] = mod_sub(0, sa_e, m);
            // pk1 = a
            pk[loc + rns_chunk] = a[loc];
        }
    }
}

// ===========================================================================
// CPU reference: threshold_pk_addition
// ===========================================================================

/// Threshold public key addition for multi-party.
///
/// `pkout[i] = pk1[i] + pk2[i] mod q` for the first half (pk0 component).
/// If `first` is true, the second half (pk1 component) is copied from pk1.
/// Otherwise the second half is added element-wise.
pub fn threshold_pk_addition_cpu(
    pk1: &[u64],
    pk2: &[u64],
    pkout: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    first: bool,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for j in 0..rns_count {
        let m = &moduli[j];
        for i in 0..ring_size {
            let loc = i + (j << n_power);
            pkout[loc] = mod_add(pk1[loc], pk2[loc], m);
            if first {
                pkout[loc + rns_chunk] = pk1[loc + rns_chunk];
            } else {
                pkout[loc + rns_chunk] = mod_add(
                    pk1[loc + rns_chunk],
                    pk2[loc + rns_chunk],
                    m,
                );
            }
        }
    }
}

// ===========================================================================
// CPU reference: relinkey_gen_kernel
// ===========================================================================

/// Relinearization key generation (Method I).
///
/// For each decomposition level `k` in `0..d-1` and each RNS modulus `j`:
///   rk0[k][j][i] = -(sk * a[k][j][i] + e[k][j][i]) + P_k * sk^2[j][i]
///   rk1[k][j][i] = a[k][j][i]
///
/// `rk` layout: d_tilda * 2 * rns_count * ring_size.
/// `sk_sk` (sk^2) layout: rns_count * ring_size.
/// `a`, `e` layout: d_tilda * rns_count * ring_size.
/// `factor` layout: d_tilda * rns_count (the P_k factors).
pub fn relinkey_gen_cpu(
    rk: &mut [u64],
    sk: &[u64],
    sk_sk: &[u64],
    error: &[u64],
    a: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let a_loc = i + (j << n_power) + k * rns_chunk;
                let sk_loc = i + (j << n_power);
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let rk1_loc = rk0_loc + rns_chunk;

                let sa = mod_mul(sk[sk_loc], a[a_loc], m);
                let sa_e = mod_add(sa, error[a_loc], m);
                let neg = mod_sub(0, sa_e, m);
                let f = factor[k * rns_count + j];
                let scaled = mod_mul(f, sk_sk[sk_loc], m);
                rk[rk0_loc] = mod_add(neg, scaled, m);
                rk[rk1_loc] = a[a_loc];
            }
        }
    }
}

// ===========================================================================
// CPU reference: relinkey_gen_II_kernel
// ===========================================================================

/// Relinearization key generation (Method II).
///
/// Similar to Method I but without the extra P_k * sk^2 term.
/// rk0 = -(sk * a + e), rk1 = a, for each decomposition level.
pub fn relinkey_gen_ii_cpu(
    rk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let a_loc = i + (j << n_power) + k * rns_chunk;
                let sk_loc = i + (j << n_power);
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let rk1_loc = rk0_loc + rns_chunk;

                let sa = mod_mul(sk[sk_loc], a[a_loc], m);
                let sa_e = mod_add(sa, error[a_loc], m);
                rk[rk0_loc] = mod_sub(0, sa_e, m);
                rk[rk1_loc] = a[a_loc];
            }
        }
    }
}

/// Relinearization key generation (Method II, leveled variant).
///
/// Same as `relinkey_gen_ii_cpu` but operates at a specific level
/// with `current_rns_count` moduli.
pub fn relinkey_gen_ii_leveled_cpu(
    rk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    current_rns_count: usize,
    d_tilda: usize,
) {
    relinkey_gen_ii_cpu(rk, sk, error, a, moduli, n_power, current_rns_count, d_tilda);
}

// ===========================================================================
// CPU reference: relinkey_DtoB_kernel
// ===========================================================================

/// Relinearization key base conversion (D-to-B).
///
/// Converts relin key components from the decomposition base D to the
/// auxiliary base B using f64 intermediate accumulation. This is a complex
/// multi-output kernel involving floating-point base conversion.
///
/// Each thread reads across `d` decomposition levels and writes across
/// `d_tilda` output levels.
pub fn relinkey_dtob_cpu(
    rk: &mut [u64],
    base_change_matrix: &[u64],
    mi_inv: &[u64],
    moduli_d: &[Modulus64],
    moduli_b: &[Modulus64],
    n_power: u32,
    d: usize,
    d_tilda: usize,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    let _ = (rk, base_change_matrix, mi_inv, moduli_d, moduli_b, ring_size, d, d_tilda, rns_count);
    // TODO: full f64-based base conversion (complex multi-output kernel)
    // Each thread accumulates d contributions using f64 rounding,
    // then writes d_tilda outputs. Requires careful porting of the
    // CUDA double-precision accumulation logic.
}

/// Leveled variant of D-to-B base conversion.
pub fn relinkey_dtob_leveled_cpu(
    rk: &mut [u64],
    base_change_matrix: &[u64],
    mi_inv: &[u64],
    moduli_d: &[Modulus64],
    moduli_b: &[Modulus64],
    n_power: u32,
    d: usize,
    d_tilda: usize,
    current_rns_count: usize,
) {
    let _ = (rk, base_change_matrix, mi_inv, moduli_d, moduli_b, n_power, d, d_tilda, current_rns_count);
    // TODO: leveled variant of f64-based base conversion
}

// ===========================================================================
// CPU reference: multi_party_relinkey_piece_method_I_stage_I_kernel
// ===========================================================================

/// Multi-party relin key piece generation (Method I, Stage I).
///
/// For each decomposition level `k` and RNS modulus `j`:
///   rk0[k][j][i] = -(sk * a + e) + P_k * sk * u
///   rk1[k][j][i] = a
pub fn multi_party_relinkey_piece_method_i_stage_i_cpu(
    rk: &mut [u64],
    sk: &[u64],
    u: &[u64],
    error: &[u64],
    a: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let a_loc = i + (j << n_power) + k * rns_chunk;
                let sk_loc = i + (j << n_power);
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let rk1_loc = rk0_loc + rns_chunk;

                let sa = mod_mul(sk[sk_loc], a[a_loc], m);
                let sa_e = mod_add(sa, error[a_loc], m);
                let neg = mod_sub(0, sa_e, m);

                let f = factor[k * rns_count + j];
                let sku = mod_mul(sk[sk_loc], u[sk_loc], m);
                let scaled = mod_mul(f, sku, m);

                rk[rk0_loc] = mod_add(neg, scaled, m);
                rk[rk1_loc] = a[a_loc];
            }
        }
    }
}

// ===========================================================================
// CPU reference: multi_party_relinkey_piece_method_II_stage_I_kernel
// ===========================================================================

/// Multi-party relin key piece generation (Method II, Stage I).
///
/// rk0 = -(sk * a + e), rk1 = a.
pub fn multi_party_relinkey_piece_method_ii_stage_i_cpu(
    rk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    relinkey_gen_ii_cpu(rk, sk, error, a, moduli, n_power, rns_count, d_tilda);
}

// ===========================================================================
// CPU reference: multi_party_relinkey_piece_method_I_II_stage_II_kernel
// ===========================================================================

/// Multi-party relin key piece aggregation (Stage II, Methods I & II).
///
/// Accumulates relin key pieces from multiple parties:
///   rk0_out += piece_rk0
///   rk1_out += piece_rk1 (or copies on first)
pub fn multi_party_relinkey_piece_method_i_ii_stage_ii_cpu(
    rk_out: &mut [u64],
    rk_piece: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
    first: bool,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let rk1_loc = rk0_loc + rns_chunk;

                rk_out[rk0_loc] = mod_add(rk_out[rk0_loc], rk_piece[rk0_loc], m);
                if first {
                    rk_out[rk1_loc] = rk_piece[rk1_loc];
                } else {
                    rk_out[rk1_loc] = mod_add(rk_out[rk1_loc], rk_piece[rk1_loc], m);
                }
            }
        }
    }
}

// ===========================================================================
// CPU reference: multi_party_relinkey_method_I_stage_I_kernel (2 overloads)
// ===========================================================================

/// Multi-party relin key (Method I, Stage I) — adds sk^2 contribution.
///
/// rk0 += P_k * sk^2 for each decomposition level.
pub fn multi_party_relinkey_method_i_stage_i_cpu(
    rk: &mut [u64],
    sk_sk: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let sk_loc = i + (j << n_power);

                let f = factor[k * rns_count + j];
                let scaled = mod_mul(f, sk_sk[sk_loc], m);
                rk[rk0_loc] = mod_add(rk[rk0_loc], scaled, m);
            }
        }
    }
}

/// Multi-party relin key (Method I, Stage I, overload 2) — subtracts sk*u.
///
/// rk0 -= P_k * sk * u for each decomposition level.
pub fn multi_party_relinkey_method_i_stage_i_v2_cpu(
    rk: &mut [u64],
    sk: &[u64],
    u: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let sk_loc = i + (j << n_power);

                let f = factor[k * rns_count + j];
                let sku = mod_mul(sk[sk_loc], u[sk_loc], m);
                let scaled = mod_mul(f, sku, m);
                rk[rk0_loc] = mod_sub(rk[rk0_loc], scaled, m);
            }
        }
    }
}

// ===========================================================================
// CPU reference: multi_party_relinkey_method_I_stage_II_kernel (2 overloads)
// ===========================================================================

/// Multi-party relin key aggregation (Method I, Stage II).
///
/// rk0_out += piece_rk0, rk1_out += piece_rk1.
pub fn multi_party_relinkey_method_i_stage_ii_cpu(
    rk_out: &mut [u64],
    rk_piece: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let rk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let rk1_loc = rk0_loc + rns_chunk;
                rk_out[rk0_loc] = mod_add(rk_out[rk0_loc], rk_piece[rk0_loc], m);
                rk_out[rk1_loc] = mod_add(rk_out[rk1_loc], rk_piece[rk1_loc], m);
            }
        }
    }
}

/// Overload 2: with conditional first-copy behavior on rk1.
pub fn multi_party_relinkey_method_i_stage_ii_v2_cpu(
    rk_out: &mut [u64],
    rk_piece: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
    first: bool,
) {
    multi_party_relinkey_piece_method_i_ii_stage_ii_cpu(
        rk_out, rk_piece, moduli, n_power, rns_count, d_tilda, first,
    );
}

// ===========================================================================
// CPU reference: galoiskey_gen_kernel
// ===========================================================================

/// Galois key generation (Method I).
///
/// For each decomposition level `k` and RNS modulus `j`:
///   gk0[k][j][i] = -(sk * a + e) + P_k * permute(sk, galois_elt)
///   gk1[k][j][i] = a
///
/// Uses the `permutation` helper function for the Galois automorphism.
pub fn galoiskey_gen_cpu(
    gk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
    galois_elt: i32,
) {
    let ring_size = 1usize << n_power;
    let coeff_count = ring_size as i32;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let a_loc = i + (j << n_power) + k * rns_chunk;
                let sk_loc = i + (j << n_power);
                let gk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let gk1_loc = gk0_loc + rns_chunk;

                let sa = mod_mul(sk[sk_loc], a[a_loc], m);
                let sa_e = mod_add(sa, error[a_loc], m);
                let neg = mod_sub(0, sa_e, m);

                // Apply Galois automorphism to sk
                let perm_idx = permutation(i as i32, galois_elt, coeff_count, n_power as i32);
                let sk_perm_loc = perm_idx as usize + (j << n_power);
                let f = factor[k * rns_count + j];
                let scaled = mod_mul(f, sk[sk_perm_loc], m);

                gk[gk0_loc] = mod_add(neg, scaled, m);
                gk[gk1_loc] = a[a_loc];
            }
        }
    }
}

// ===========================================================================
// CPU reference: galoiskey_gen_II_kernel
// ===========================================================================

/// Galois key generation (Method II).
///
/// gk0 = -(sk * a + e), gk1 = a. Same structure as relinkey_gen_ii
/// but operates in the Galois key context.
pub fn galoiskey_gen_ii_cpu(
    gk: &mut [u64],
    sk: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    relinkey_gen_ii_cpu(gk, sk, error, a, moduli, n_power, rns_count, d_tilda);
}

// ===========================================================================
// CPU reference: multi_party_galoiskey_gen_method_I_II_kernel
// ===========================================================================

/// Multi-party Galois key generation (Methods I & II).
///
/// Accumulates Galois key pieces from multiple parties, same structure
/// as the relin key aggregation.
pub fn multi_party_galoiskey_gen_method_i_ii_cpu(
    gk_out: &mut [u64],
    gk_piece: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
    first: bool,
) {
    multi_party_relinkey_piece_method_i_ii_stage_ii_cpu(
        gk_out, gk_piece, moduli, n_power, rns_count, d_tilda, first,
    );
}

// ===========================================================================
// CPU reference: switchkey_gen_kernel
// ===========================================================================

/// Switch key generation (Method I).
///
/// For each decomposition level `k` and RNS modulus `j`:
///   swk0[k][j][i] = -(sk_to * a + e) + P_k * sk_from
///   swk1[k][j][i] = a
pub fn switchkey_gen_cpu(
    swk: &mut [u64],
    sk_to: &[u64],
    sk_from: &[u64],
    error: &[u64],
    a: &[u64],
    factor: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for k in 0..d_tilda {
        for j in 0..rns_count {
            let m = &moduli[j];
            for i in 0..ring_size {
                let a_loc = i + (j << n_power) + k * rns_chunk;
                let sk_loc = i + (j << n_power);
                let swk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let swk1_loc = swk0_loc + rns_chunk;

                let sa = mod_mul(sk_to[sk_loc], a[a_loc], m);
                let sa_e = mod_add(sa, error[a_loc], m);
                let neg = mod_sub(0, sa_e, m);

                let f = factor[k * rns_count + j];
                let scaled = mod_mul(f, sk_from[sk_loc], m);

                swk[swk0_loc] = mod_add(neg, scaled, m);
                swk[swk1_loc] = a[a_loc];
            }
        }
    }
}

// ===========================================================================
// CPU reference: switchkey_gen_II_kernel
// ===========================================================================

/// Switch key generation (Method II).
///
/// swk0 = -(sk_to * a + e), swk1 = a.
pub fn switchkey_gen_ii_cpu(
    swk: &mut [u64],
    sk_to: &[u64],
    error: &[u64],
    a: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d_tilda: usize,
) {
    relinkey_gen_ii_cpu(swk, sk_to, error, a, moduli, n_power, rns_count, d_tilda);
}

// ===========================================================================
// CPU reference: switchkey_kernel
// ===========================================================================

/// Switch key application (key-switching step).
///
/// Applies the switch key to an input polynomial, producing two output
/// polynomials (c0, c1). Each thread reads across `d` levels and accumulates.
pub fn switchkey_cpu(
    output: &mut [u64],
    input: &[u64],
    swk: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    d: usize,
) {
    let ring_size = 1usize << n_power;
    let rns_chunk = rns_count * ring_size;
    for j in 0..rns_count {
        let m = &moduli[j];
        for i in 0..ring_size {
            let out0_loc = i + (j << n_power);
            let out1_loc = out0_loc + rns_chunk;
            let mut acc0: u64 = 0;
            let mut acc1: u64 = 0;
            for k in 0..d {
                let in_loc = i + (j << n_power) + k * rns_chunk;
                let swk0_loc = i + (j << n_power) + k * 2 * rns_chunk;
                let swk1_loc = swk0_loc + rns_chunk;

                let contrib0 = mod_mul(input[in_loc], swk[swk0_loc], m);
                let contrib1 = mod_mul(input[in_loc], swk[swk1_loc], m);
                acc0 = mod_add(acc0, contrib0, m);
                acc1 = mod_add(acc1, contrib1, m);
            }
            output[out0_loc] = acc0;
            output[out1_loc] = acc1;
        }
    }
}

// ===========================================================================
// TFHE stubs (curand / NTT / shared memory dependent)
// ===========================================================================

/// TFHE secret key generation.
///
/// **Stub**: requires curand for uniform random number generation.
/// The CUDA kernel uses `curand_uniform` to generate binary secret key
/// coefficients in {0, 1}.
pub fn tfhe_secretkey_gen_stub(
    _sk: &mut [i32],
    _n: usize,
) {
    unimplemented!(
        "tfhe_secretkey_gen requires curand (GPU random number generation). \
         Provide pre-generated random data or use a CPU RNG."
    );
}

/// TFHE noise generation.
///
/// **Stub**: requires curand for Gaussian noise sampling.
/// The CUDA kernel uses `curand_normal_double` to generate discrete
/// Gaussian noise with the specified standard deviation.
pub fn tfhe_generate_noise_stub(
    _noise: &mut [f64],
    _n: usize,
    _std_dev: f64,
) {
    unimplemented!(
        "tfhe_generate_noise requires curand (GPU Gaussian sampling). \
         Provide pre-generated noise or use a CPU RNG."
    );
}

/// TFHE uniform random number generation.
///
/// **Stub**: requires curand for uniform random u64 generation.
/// Used to sample the `a` polynomial in RLWE encryption.
pub fn tfhe_generate_uniform_random_stub(
    _output: &mut [u64],
    _n: usize,
) {
    unimplemented!(
        "tfhe_generate_uniform_random requires curand. \
         Provide pre-generated random data or use a CPU RNG."
    );
}

/// TFHE switch key generation.
///
/// **Stub**: requires curand + warp reduction + shared memory.
/// Generates a key-switching key from an LWE secret key to a smaller
/// LWE secret key, using RLWE encryption of key bits with Gaussian noise.
pub fn tfhe_generate_switchkey_stub(
    _swk: &mut [u64],
    _sk_in: &[i32],
    _sk_out: &[i32],
    _n_in: usize,
    _n_out: usize,
    _base_log: u32,
    _level: u32,
) {
    unimplemented!(
        "tfhe_generate_switchkey requires curand + warp reduction + shared memory. \
         Port requires a multi-pass CPU implementation."
    );
}

/// TFHE bootstrapping key random number generation.
///
/// **Stub**: requires curand for generating random polynomials used
/// in RGSW encryption of the bootstrapping key bits.
pub fn tfhe_generate_bootkey_random_numbers_stub(
    _randoms: &mut [u64],
    _total_elements: usize,
) {
    unimplemented!(
        "tfhe_generate_bootkey_random_numbers requires curand. \
         Provide pre-generated random data or use a CPU RNG."
    );
}

/// TFHE RLWE key conversion to NTT domain.
///
/// **Stub**: requires SmallForwardNTT + shared memory.
/// Converts RLWE key polynomials into NTT representation for efficient
/// polynomial multiplication during bootstrapping.
pub fn tfhe_convert_rlwekey_ntt_domain_stub(
    _key: &mut [u64],
    _n: usize,
    _modulus: u64,
) {
    unimplemented!(
        "tfhe_convert_rlwekey_ntt_domain requires SmallForwardNTT + shared memory. \
         Use an external NTT library or port the NTT kernels separately."
    );
}

/// TFHE bootstrapping key conversion to NTT domain.
///
/// **Stub**: requires SmallForwardNTT + shared memory.
/// Converts all RGSW ciphertexts in the bootstrapping key to NTT
/// representation.
pub fn tfhe_convert_bootkey_ntt_domain_stub(
    _bootkey: &mut [u64],
    _n: usize,
    _n_lwe: usize,
    _level: u32,
    _modulus: u64,
) {
    unimplemented!(
        "tfhe_convert_bootkey_ntt_domain requires SmallForwardNTT + shared memory. \
         Use an external NTT library or port the NTT kernels separately."
    );
}

/// TFHE bootstrapping key generation.
///
/// **Stub**: requires curand + SmallForwardNTT/InverseNTT + shared memory.
/// Encrypts each bit of the LWE secret key as an RGSW ciphertext,
/// forming the bootstrapping key used in gate bootstrapping.
pub fn tfhe_generate_bootkey_stub(
    _bootkey: &mut [u64],
    _sk_lwe: &[i32],
    _sk_rlwe: &[i32],
    _n_lwe: usize,
    _n_rlwe: usize,
    _level: u32,
    _base_log: u32,
    _modulus: u64,
    _std_dev: f64,
) {
    unimplemented!(
        "tfhe_generate_bootkey requires curand + NTT + shared memory. \
         This is the most complex TFHE kernel — port requires a full \
         multi-pass CPU implementation with external NTT and RNG."
    );
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitreverse() {
        // 3 bits: 0b101 -> 0b101 = 5
        assert_eq!(bitreverse(5, 3), 5);
        // 3 bits: 0b001 -> 0b100 = 4
        assert_eq!(bitreverse(1, 3), 4);
        // 3 bits: 0b110 -> 0b011 = 3
        assert_eq!(bitreverse(6, 3), 3);
        // 4 bits: 0b0001 -> 0b1000 = 8
        assert_eq!(bitreverse(1, 4), 8);
        assert_eq!(bitreverse(0, 4), 0);
    }

    #[test]
    fn test_steps_to_galois_elt() {
        let n = 8; // coeff_count
        let group_order = 2 * n;

        // step=0 gives column rotation element
        assert_eq!(steps_to_galois_elt(0, n, group_order), group_order - 1);

        // step=1: 3^1 mod 16 = 3
        assert_eq!(steps_to_galois_elt(1, n, group_order), 3);

        // step=2: 3^2 mod 16 = 9
        assert_eq!(steps_to_galois_elt(2, n, group_order), 9);
    }

    #[test]
    fn test_permutation_identity() {
        // galois_elt = 1 should be identity permutation
        let n_power = 3;
        let n = 1 << n_power;
        for i in 0..n {
            assert_eq!(permutation(i, 1, n, n_power), i);
        }
    }

    #[test]
    fn test_secretkey_gen_v2() {
        let n_power = 3u32;
        let n = 1usize << n_power;
        let mut sk = vec![0i32; n];
        let positions = vec![1, 3, 5];
        let values = vec![1, -1, 1];

        secretkey_gen_v2_cpu(&mut sk, &positions, &values, 3, n_power);

        assert_eq!(sk[0], 0);
        assert_eq!(sk[1], 1);
        assert_eq!(sk[2], 0);
        assert_eq!(sk[3], -1);
        assert_eq!(sk[4], 0);
        assert_eq!(sk[5], 1);
        assert_eq!(sk[6], 0);
        assert_eq!(sk[7], 0);
    }

    #[test]
    fn test_secretkey_rns() {
        let n_power = 2u32;
        let n = 1usize << n_power;
        let input = vec![1i32, -1, 0, 2];
        let moduli = vec![Modulus64::new(17), Modulus64::new(13)];
        let rns_count = 2;
        let mut output = vec![0u64; n * rns_count];

        secretkey_rns_cpu(&input, &mut output, &moduli, n_power, rns_count);

        // mod 17: [1, 16, 0, 2]
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 16); // -1 mod 17
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 2);
        // mod 13: [1, 12, 0, 2]
        assert_eq!(output[4], 1);
        assert_eq!(output[5], 12); // -1 mod 13
        assert_eq!(output[6], 0);
        assert_eq!(output[7], 2);
    }

    #[test]
    fn test_publickey_gen() {
        let n_power = 2u32;
        let n = 1usize << n_power;
        let rns_count = 1;
        let moduli = vec![Modulus64::new(17)];

        let sk = vec![1u64, 2, 3, 4];
        let a = vec![5u64, 6, 7, 8];
        let error = vec![0u64; n];
        let mut pk = vec![0u64; n * rns_count * 2];

        publickey_gen_cpu(&mut pk, &sk, &error, &a, &moduli, n_power, rns_count);

        // pk0 = -(sk*a) mod 17
        for i in 0..n {
            let expected = mod_sub(0, mod_mul(sk[i], a[i], &moduli[0]), &moduli[0]);
            assert_eq!(pk[i], expected);
        }
        // pk1 = a
        for i in 0..n {
            assert_eq!(pk[n + i], a[i]);
        }
    }

    #[test]
    fn test_threshold_pk_addition() {
        let n_power = 2u32;
        let n = 1usize << n_power;
        let rns_count = 1;
        let moduli = vec![Modulus64::new(17)];

        let pk1 = vec![1u64, 2, 3, 4, 10, 11, 12, 13];
        let pk2 = vec![5u64, 6, 7, 8, 1, 2, 3, 4];
        let mut pkout = vec![0u64; n * rns_count * 2];

        // first=true: pk0 added, pk1 copied from pk1
        threshold_pk_addition_cpu(&pk1, &pk2, &mut pkout, &moduli, n_power, rns_count, true);
        assert_eq!(pkout[0], mod_add(1, 5, &moduli[0]));
        assert_eq!(pkout[n], 10); // copied, not added
    }
}
