/// BFV decryption kernels: secret-key multiplication and full decrypt.
///
/// Data layout follows HEonGPU conventions:
/// - Ciphertext component `c0`, `c1`: flat arrays of length `ring_size * rns_count`
/// - Secret key `sk`: flat array of length `ring_size * rns_count`

use crate::modular::{mod_add, mod_mul, mod_reduce_forced, mod_sub, Modulus64};
use gpu::prelude::*;

// ---------------------------------------------------------------------------
// CPU reference functions
// ---------------------------------------------------------------------------

/// Decrypt core: output[i] = c1[i] * sk[i] mod q, for each RNS level.
pub fn sk_multiplication_cpu(
    ct1: &[u64],
    sk: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..rns_count {
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            output[index] = mod_mul(ct1[index], sk[index], &moduli[block_y]);
        }
    }
}

/// Full decryption: result[i] = c0[i] + c1[i]*sk[i] mod q.
pub fn decrypt_cpu(
    c0: &[u64],
    c1: &[u64],
    sk: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..rns_count {
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let prod = mod_mul(c1[index], sk[index], &moduli[block_y]);
            output[index] = mod_add(c0[index], prod, &moduli[block_y]);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

/// GPU kernel: sk_multiplication — Barrett multiply c1[i] * sk[i] mod q.
///
/// Launch with 1D grid over `ring_size * rns_count` elements.
#[gpu::cuda_kernel]
pub fn sk_multiplication_kernel(
    ct1: &[u64],
    sk: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gid = (bid * bdim + tid) as usize;

    let idy = (gid >> (n_power as usize)) % (rns_count as usize);
    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let a = ct1[gid];
    let b = sk[gid];

    // Inline Barrett multiplication
    let z = (a as u128) * (b as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut r = (z - w) as u64;
    if r >= mod_val {
        r -= mod_val;
    }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// GPU kernel: sk_multiplicationx3 — 3-component Barrett multiply
// ---------------------------------------------------------------------------

/// GPU kernel: sk_multiplicationx3 — multiplies ct1 by sk and ct2 by sk^2.
///
/// Reads from `ct_in` (length = ring_size * decomp_mod_count * 2):
///   - First half: ct1 coefficients
///   - Second half: ct2 coefficients
///
/// Writes to `ct_out` (same layout):
///   - ct_out[index] = ct_in[index] * sk[index] mod q
///   - ct_out[index + offset] = ct_in[index + offset] * sk^2[index] mod q
///
/// Launch with 1D grid over `ring_size * decomp_mod_count` elements.
/// Caller can pass the same buffer for ct_in and ct_out for in-place operation
/// on the host side (copy to ct_out before launch if needed).
#[gpu::cuda_kernel]
pub fn sk_multiplicationx3_kernel(
    ct_in: &[u64],
    sk: &[u64],
    ct_out1: &mut [u64],
    ct_out2: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    decomp_mod_count: u32,
) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gid = (bid * bdim + tid) as usize;

    let idy = (gid >> (n_power as usize)) % (decomp_mod_count as usize);
    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let ct_1 = ct_in[gid];
    let sk_ = sk[gid];

    // ct1 * sk
    let z = (ct_1 as u128) * (sk_ as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut r = (z - w) as u64;
    if r >= mod_val {
        r -= mod_val;
    }

    let offset = (decomp_mod_count as usize) << (n_power as usize);

    // sk^2
    let z2 = (sk_ as u128) * (sk_ as u128);
    let w2 = z2 >> (bit as u32 - 2);
    let w2 = (w2 * (mu as u128)) >> (bit as u32 + 3);
    let w2 = w2 * (mod_val as u128);
    let mut sk2 = (z2 - w2) as u64;
    if sk2 >= mod_val {
        sk2 -= mod_val;
    }

    // ct2 * sk^2
    let ct_2 = ct_in[gid + offset];
    let z3 = (ct_2 as u128) * (sk2 as u128);
    let w3 = z3 >> (bit as u32 - 2);
    let w3 = (w3 * (mu as u128)) >> (bit as u32 + 3);
    let w3 = w3 * (mod_val as u128);
    let mut r2 = (z3 - w3) as u64;
    if r2 >= mod_val {
        r2 -= mod_val;
    }

    let mut out1 = chunk_mut(ct_out1, MapLinear::new(1));
    out1[0] = r;
    let mut out2 = chunk_mut(ct_out2, MapLinear::new(1));
    out2[0] = r2;
}

/// CPU reference: sk_multiplicationx3 — 3-component secret-key multiplication.
pub fn sk_multiplicationx3_cpu(
    ct1: &mut [u64],
    sk: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..decomp_mod_count {
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let ct_1 = ct1[index];
            let sk_ = sk[index];
            ct1[index] = mod_mul(ct_1, sk_, &moduli[block_y]);

            let offset = decomp_mod_count << n_power;
            let ct_2 = ct1[index + offset];
            let sk2 = mod_mul(sk_, sk_, &moduli[block_y]);
            ct1[index + offset] = mod_mul(ct_2, sk2, &moduli[block_y]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: decryption_kernel — RNS CRT decryption
// ---------------------------------------------------------------------------

/// CPU reference: decryption_kernel — full BFV decryption using RNS CRT reconstruction.
///
/// Implements the Bajard et al. fast base conversion for BFV decryption:
///   For each ring coefficient idx:
///     1. For each RNS level i:
///        - Compute mt = (c0[i] + c1[i]) * t * γ * Qi_inverse[i] mod q_i
///        - Reduce mt into the plaintext modulus t → accumulate sum_t
///        - Reduce mt into auxiliary modulus γ → accumulate sum_gamma
///     2. Multiply sums by mulq_inv_t and mulq_inv_gamma
///     3. Apply γ-correction: if sum_gamma > γ/2, adjust for negative range
///     4. Multiply by inv_gamma to get final plaintext
///
/// # Table parameters
/// - `qi_t`: CRT basis conversion coefficients for t, Qi_t[i] = (Q/q_i) mod t
/// - `qi_gamma`: CRT basis conversion coefficients for γ, Qi_gamma[i] = (Q/q_i) mod γ
/// - `qi_inverse`: Qi_inverse[i] = (Q/q_i)^{-1} mod q_i
/// - `mulq_inv_t`: (-Q)^{-1} mod t
/// - `mulq_inv_gamma`: (-Q)^{-1} mod γ
/// - `inv_gamma`: γ^{-1} mod t
/// - `gamma`: auxiliary modulus for fast base conversion
pub fn decryption_kernel_cpu(
    c0: &[u64],
    c1: &[u64],
    plain: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    gamma: &Modulus64,
    qi_t: &[u64],
    qi_gamma: &[u64],
    qi_inverse: &[u64],
    mulq_inv_t: u64,
    mulq_inv_gamma: u64,
    inv_gamma: u64,
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let mut sum_t: u64 = 0;
        let mut sum_gamma: u64 = 0;

        for i in 0..decomp_mod_count {
            let location = idx + (i << n_power);

            let mut mt = mod_add(c0[location], c1[location], &moduli[i]);

            let gamma_reduced = mod_reduce_forced(gamma.value, &moduli[i]);

            mt = mod_mul(mt, plain_mod.value, &moduli[i]);
            mt = mod_mul(mt, gamma_reduced, &moduli[i]);
            mt = mod_mul(mt, qi_inverse[i], &moduli[i]);

            let mt_in_t = mod_reduce_forced(mt, plain_mod);
            let mt_in_gamma = mod_reduce_forced(mt, gamma);

            let mt_in_t = mod_mul(mt_in_t, qi_t[i], plain_mod);
            let mt_in_gamma = mod_mul(mt_in_gamma, qi_gamma[i], gamma);

            sum_t = mod_add(sum_t, mt_in_t, plain_mod);
            sum_gamma = mod_add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = mod_mul(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = mod_mul(sum_gamma, mulq_inv_gamma, gamma);

        let gamma_2 = gamma.value >> 1;

        if sum_gamma > gamma_2 {
            let gamma_in_t = mod_reduce_forced(gamma.value, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(gamma_in_t, sum_gamma_in_t, plain_mod);
            result = mod_add(sum_t, result, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        } else {
            let sum_t_r = mod_reduce_forced(sum_t, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(sum_t_r, sum_gamma_in_t, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: decryption_kernelx3 — 3-component RNS CRT decryption
// ---------------------------------------------------------------------------

/// CPU reference: decryption_kernelx3 — 3-component variant of `decryption_kernel`.
///
/// Same algorithm as `decryption_kernel_cpu` but the initial accumulation
/// sums three ciphertext components: mt = c0[i] + c1[i] + c2[i].
/// Used after relinearization of degree-2 ciphertexts.
pub fn decryption_kernelx3_cpu(
    c0: &[u64],
    c1: &[u64],
    c2: &[u64],
    plain: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    gamma: &Modulus64,
    qi_t: &[u64],
    qi_gamma: &[u64],
    qi_inverse: &[u64],
    mulq_inv_t: u64,
    mulq_inv_gamma: u64,
    inv_gamma: u64,
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let mut sum_t: u64 = 0;
        let mut sum_gamma: u64 = 0;

        for i in 0..decomp_mod_count {
            let location = idx + (i << n_power);

            let mut mt = mod_add(c0[location], c1[location], &moduli[i]);
            mt = mod_add(mt, c2[location], &moduli[i]);

            let gamma_reduced = mod_reduce_forced(gamma.value, &moduli[i]);

            mt = mod_mul(mt, plain_mod.value, &moduli[i]);
            mt = mod_mul(mt, gamma_reduced, &moduli[i]);
            mt = mod_mul(mt, qi_inverse[i], &moduli[i]);

            let mt_in_t = mod_reduce_forced(mt, plain_mod);
            let mt_in_gamma = mod_reduce_forced(mt, gamma);

            let mt_in_t = mod_mul(mt_in_t, qi_t[i], plain_mod);
            let mt_in_gamma = mod_mul(mt_in_gamma, qi_gamma[i], gamma);

            sum_t = mod_add(sum_t, mt_in_t, plain_mod);
            sum_gamma = mod_add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = mod_mul(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = mod_mul(sum_gamma, mulq_inv_gamma, gamma);

        let gamma_2 = gamma.value >> 1;

        if sum_gamma > gamma_2 {
            let gamma_in_t = mod_reduce_forced(gamma.value, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(gamma_in_t, sum_gamma_in_t, plain_mod);
            result = mod_add(sum_t, result, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        } else {
            let sum_t_r = mod_reduce_forced(sum_t, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(sum_t_r, sum_gamma_in_t, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: coeff_multadd
// ---------------------------------------------------------------------------

/// GPU kernel: coeff_multadd — output = (input1 + input2) * plain_mod_value mod q.
///
/// Launch with 1D grid over `ring_size * decomp_mod_count` elements.
#[gpu::cuda_kernel]
pub fn coeff_multadd_kernel(
    input1: &[u64],
    input2: &[u64],
    output: &mut [u64],
    plain_mod_value: u64,
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    decomp_mod_count: u32,
) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gid = (bid * bdim + tid) as usize;

    let idy = (gid >> (n_power as usize)) % (decomp_mod_count as usize);
    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    // add input1 + input2
    let ct_0 = input1[gid];
    let ct_1 = input2[gid];
    let mut sum = ct_1 + ct_0;
    if sum >= mod_val {
        sum -= mod_val;
    }

    // multiply by plain_mod_value
    let z = (sum as u128) * (plain_mod_value as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut r = (z - w) as u64;
    if r >= mod_val {
        r -= mod_val;
    }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

/// CPU reference: coeff_multadd — output = (input1 + input2) * plain_mod mod q.
pub fn coeff_multadd_cpu(
    input1: &[u64],
    input2: &[u64],
    output: &mut [u64],
    plain_mod: &Modulus64,
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..decomp_mod_count {
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let mut ct_0 = mod_add(input2[index], input1[index], &moduli[block_y]);
            ct_0 = mod_mul(ct_0, plain_mod.value, &moduli[block_y]);
            output[index] = ct_0;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: compose_kernel — CRT composition with big integer arithmetic
// ---------------------------------------------------------------------------

/// CPU reference: compose_kernel — RNS to positional (big integer) conversion.
///
/// For each ring coefficient, this performs CRT composition:
///   1. For each RNS level i: compute base * Mi_inv[i] mod q_i
///   2. Multiply by the big-integer Mi[i] (the CRT lifting factor)
///   3. Accumulate into a big-integer result
///   4. If result >= decryption_modulus, subtract it
///   5. Scatter the big-integer limbs back to interleaved layout
///
/// # Parameters
/// - `input`: RNS representation, length = ring_size * coeff_modulus_count
/// - `output`: positional representation, same length (limbs interleaved by ring_size)
/// - `moduli`: RNS moduli array
/// - `mi_inv`: Mi_inv[i] = (M/q_i)^{-1} mod q_i, length = coeff_modulus_count
/// - `mi`: big-integer CRT lifting factors, mi[i*coeff_modulus_count .. (i+1)*coeff_modulus_count]
/// - `decryption_modulus`: big-integer product of all moduli, length = coeff_modulus_count limbs
pub fn compose_kernel_cpu(
    input: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    mi_inv: &[u64],
    mi: &[u64],
    decryption_modulus: &[u64],
    coeff_modulus_count: usize,
    n_power: u32,
) {
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let mut compose_result = vec![0u64; coeff_modulus_count];

        for i in 0..coeff_modulus_count {
            let base = input[idx + (i << n_power)];
            let temp = mod_mul(base, mi_inv[i], &moduli[i]);

            // Multiply Mi[i] (big integer) by temp (scalar) → big_integer_result
            let mi_slice = &mi[i * coeff_modulus_count..(i + 1) * coeff_modulus_count];
            let mut big_integer_result = vec![0u64; coeff_modulus_count];
            let mut carry: u128 = 0;
            for k in 0..coeff_modulus_count {
                let prod = (mi_slice[k] as u128) * (temp as u128) + carry;
                big_integer_result[k] = prod as u64;
                carry = prod >> 64;
            }

            // Add big_integer_result into compose_result
            let mut c: u128 = 0;
            for k in 0..coeff_modulus_count {
                let sum = (compose_result[k] as u128) + (big_integer_result[k] as u128) + c;
                compose_result[k] = sum as u64;
                c = sum >> 64;
            }

            // If compose_result >= decryption_modulus, subtract
            if bigint_gte(&compose_result, decryption_modulus) {
                let mut borrow: u128 = 0;
                for k in 0..coeff_modulus_count {
                    let diff = (compose_result[k] as u128)
                        .wrapping_sub(decryption_modulus[k] as u128)
                        .wrapping_sub(borrow);
                    compose_result[k] = diff as u64;
                    borrow = if diff > (compose_result[k] as u128) { 1 } else { 0 };
                    // More precise borrow detection
                    borrow = ((compose_result[k] as u128) + (decryption_modulus[k] as u128)
                        + borrow
                        != (diff & 0xFFFFFFFFFFFFFFFF) as u128 + (diff >> 64) * 0)
                        as u128;
                }
                // Simpler approach: recompute
                let mut borrow: i128 = 0;
                let saved = compose_result.clone();
                for k in 0..coeff_modulus_count {
                    let diff = (saved[k] as i128) - (decryption_modulus[k] as i128) - borrow;
                    compose_result[k] = diff as u64;
                    borrow = if diff < 0 { 1 } else { 0 };
                }
            }
        }

        // Scatter to output
        for i in 0..coeff_modulus_count {
            output[idx + (i << n_power)] = compose_result[i];
        }
    }
}

/// Helper: big-integer greater-than-or-equal comparison (little-endian limbs).
fn bigint_gte(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true // equal
}

// ---------------------------------------------------------------------------
// CPU: find_max_norm — find max |centered_reduction(val)|
// ---------------------------------------------------------------------------

/// CPU reference: find_max_norm_kernel — finds the maximum norm of
/// centered reductions across all ring coefficients.
///
/// For each coefficient idx:
///   1. Assemble the big-integer representation from interleaved limbs
///   2. If value >= upper_half_threshold, compute decryption_modulus - value
///      (this is the centered reduction for negative values)
///   3. Track the maximum such value
///
/// The CUDA version uses shared memory reduction across threads in a block.
/// This CPU version simply iterates over all coefficients.
///
/// # Parameters
/// - `input`: positional representation (from compose_kernel), interleaved limbs
/// - `output`: result big-integer of length coeff_modulus_count (max norm)
/// - `upper_half_threshold`: threshold for centered reduction, length = coeff_modulus_count
/// - `decryption_modulus`: product of all RNS moduli, length = coeff_modulus_count
pub fn find_max_norm_cpu(
    input: &[u64],
    output: &mut [u64],
    upper_half_threshold: &[u64],
    decryption_modulus: &[u64],
    coeff_modulus_count: usize,
    n_power: u32,
) {
    let ring_size = 1usize << n_power;
    let mut max_val = vec![0u64; coeff_modulus_count];

    for idx in 0..ring_size {
        // Assemble big-integer from interleaved layout
        let mut big_val = vec![0u64; coeff_modulus_count];
        for j in 0..coeff_modulus_count {
            big_val[j] = input[idx + (j << n_power)];
        }

        // Centered reduction: if val >= threshold, negate
        if bigint_gte(&big_val, upper_half_threshold) {
            // big_val = decryption_modulus - big_val
            let mut borrow: i128 = 0;
            for k in 0..coeff_modulus_count {
                let diff = (decryption_modulus[k] as i128) - (big_val[k] as i128) - borrow;
                big_val[k] = diff as u64;
                borrow = if diff < 0 { 1 } else { 0 };
            }
        }

        // Track maximum
        if bigint_gt(&big_val, &max_val) {
            max_val = big_val;
        }
    }

    output[..coeff_modulus_count].copy_from_slice(&max_val);
}

/// Helper: big-integer strict greater-than comparison (little-endian limbs).
fn bigint_gt(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    false // equal → not strictly greater
}

// ---------------------------------------------------------------------------
// GPU kernel: sk_multiplication_ckks
// ---------------------------------------------------------------------------

/// GPU kernel: sk_multiplication_ckks — CKKS decryption multiply.
///
/// Computes plaintext[i] = ciphertext[i] + ciphertext[i + decomp_mod_count * ring_size] * sk[i] mod q.
/// This fuses the c0 + c1*sk operation for CKKS where the ciphertext is stored
/// as [c0 || c1] in a single contiguous buffer.
///
/// Launch with 1D grid over `ring_size * decomp_mod_count` elements.
#[gpu::cuda_kernel]
pub fn sk_multiplication_ckks_kernel(
    ciphertext: &[u64],
    plaintext: &mut [u64],
    sk: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    decomp_mod_count: u32,
) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gid = (bid * bdim + tid) as usize;

    let idy = (gid >> (n_power as usize)) % (decomp_mod_count as usize);
    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let offset = (decomp_mod_count as usize) << (n_power as usize);
    let ct_0 = ciphertext[gid];
    let ct_1 = ciphertext[gid + offset];
    let sk_ = sk[gid];

    // ct_1 * sk
    let z = (ct_1 as u128) * (sk_ as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut prod = (z - w) as u64;
    if prod >= mod_val {
        prod -= mod_val;
    }

    // ct_0 + prod
    let mut result = prod + ct_0;
    if result >= mod_val {
        result -= mod_val;
    }

    let mut out = chunk_mut(plaintext, MapLinear::new(1));
    out[0] = result;
}

/// CPU reference: sk_multiplication_ckks.
pub fn sk_multiplication_ckks_cpu(
    ciphertext: &[u64],
    plaintext: &mut [u64],
    sk: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..decomp_mod_count {
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let offset = decomp_mod_count << n_power;

            let ct_0 = ciphertext[index];
            let ct_1 = ciphertext[index + offset];
            let sk_ = sk[index];

            let prod = mod_mul(ct_1, sk_, &moduli[block_y]);
            plaintext[index] = mod_add(prod, ct_0, &moduli[block_y]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: decryption_fusion_bfv_kernel — fused RNS + decomp decryption
// ---------------------------------------------------------------------------

/// CPU reference: decryption_fusion_bfv — fused BFV decryption.
///
/// Same algorithm as `decryption_kernel_cpu` but operates on a single
/// ciphertext buffer `ct` where the c0+c1*sk multiplication has already
/// been accumulated. The input is just the fused polynomial (not split
/// into c0 and c1).
///
/// This is used when the sk multiplication and addition have been fused
/// into a single buffer before calling decryption.
pub fn decryption_fusion_bfv_cpu(
    ct: &[u64],
    plain: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    gamma: &Modulus64,
    qi_t: &[u64],
    qi_gamma: &[u64],
    qi_inverse: &[u64],
    mulq_inv_t: u64,
    mulq_inv_gamma: u64,
    inv_gamma: u64,
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let mut sum_t: u64 = 0;
        let mut sum_gamma: u64 = 0;

        for i in 0..decomp_mod_count {
            let location = idx + (i << n_power);
            let mut mt = ct[location];

            let gamma_reduced = mod_reduce_forced(gamma.value, &moduli[i]);

            mt = mod_mul(mt, plain_mod.value, &moduli[i]);
            mt = mod_mul(mt, gamma_reduced, &moduli[i]);
            mt = mod_mul(mt, qi_inverse[i], &moduli[i]);

            let mt_in_t = mod_reduce_forced(mt, plain_mod);
            let mt_in_gamma = mod_reduce_forced(mt, gamma);

            let mt_in_t = mod_mul(mt_in_t, qi_t[i], plain_mod);
            let mt_in_gamma = mod_mul(mt_in_gamma, qi_gamma[i], gamma);

            sum_t = mod_add(sum_t, mt_in_t, plain_mod);
            sum_gamma = mod_add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = mod_mul(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = mod_mul(sum_gamma, mulq_inv_gamma, gamma);

        let gamma_2 = gamma.value >> 1;

        if sum_gamma > gamma_2 {
            let gamma_in_t = mod_reduce_forced(gamma.value, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(gamma_in_t, sum_gamma_in_t, plain_mod);
            result = mod_add(sum_t, result, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        } else {
            let sum_t_r = mod_reduce_forced(sum_t, plain_mod);
            let sum_gamma_in_t = mod_reduce_forced(sum_gamma, plain_mod);

            let mut result = mod_sub(sum_t_r, sum_gamma_in_t, plain_mod);
            result = mod_mul(result, inv_gamma, plain_mod);
            plain[idx] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: decrypt_lwe — LWE inner product decryption
// ---------------------------------------------------------------------------

/// CPU reference: decrypt_lwe — LWE decryption via inner product.
///
/// For each segment seg in [0, k):
///   output[seg] = input_b[seg] - sum_{i=0}^{n-1} (input_a[seg*n + i] * sk[i])
///
/// All arithmetic is wrapping u32/i32 (modular 2^32), matching the CUDA kernel
/// which uses `uint32_t` accumulation and `int32_t` result.
pub fn decrypt_lwe_cpu(
    sk: &[i32],
    input_a: &[i32],
    input_b: &[i32],
    output: &mut [i32],
    n: usize,
    k: usize,
) {
    for seg in 0..k {
        let base = seg * n;
        let mut sum: u32 = 0;
        for i in 0..n {
            let secret_key = sk[i] as u32;
            let r = input_a[base + i] as u32;
            sum = sum.wrapping_add(r.wrapping_mul(secret_key));
        }
        output[seg] = (input_b[seg] as u32).wrapping_sub(sum) as i32;
    }
}

// ---------------------------------------------------------------------------
// CPU: col_boot_dec_mul_with_sk — bootstrapping sk multiply
// ---------------------------------------------------------------------------

/// CPU reference: col_boot_dec_mul_with_sk — multiply with secret key for
/// collaborative bootstrapping decryption share.
///
/// For block_z=0: output[idx] = ct1[idx] * sk[idx] mod q  (c1 * sk)
/// For block_z=1: output[idx] = -(a[idx] * sk[idx]) mod q  (negated a * sk)
///
/// Output layout: ring_size * decomp_mod_count * 2
pub fn col_boot_dec_mul_with_sk_cpu(
    ct1: &[u64],
    a: &[u64],
    sk: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for block_z in 0..2usize {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let in_index = idx + (block_y << n_power);
                let out_index = in_index + ((decomp_mod_count * block_z) << n_power);

                let sk_ = sk[in_index];

                let result = if block_z == 0 {
                    // c1 * sk mod q
                    mod_mul(ct1[in_index], sk_, &moduli[block_y])
                } else {
                    // -(a * sk) mod q
                    let prod = mod_mul(a[in_index], sk_, &moduli[block_y]);
                    mod_sub(0, prod, &moduli[block_y])
                };

                output[out_index] = result;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: col_boot_add_random_and_errors
// ---------------------------------------------------------------------------

/// CPU reference: col_boot_add_random_and_errors — add error terms and
/// scaled random plaintext during collaborative bootstrapping.
///
/// For each element:
///   1. Compute ΔM = random_plain[idx] * coeffdiv_plain[block_y] + fix (mod q)
///      where fix = floor((random_plain[idx] * Q_mod_t + upper_threshold) / t)
///   2. ct[index] += error[index]
///   3. For block_z=0: ct[index] -= ΔM
///      For block_z=1: ct[index] += ΔM
pub fn col_boot_add_random_and_errors_cpu(
    ct: &mut [u64],
    errors: &[u64],
    random_plain: &[u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for block_z in 0..2usize {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let in_index = idx + (block_y << n_power)
                    + ((decomp_mod_count * block_z) << n_power);

                let random_message = random_plain[idx];
                let fix = random_message * q_mod_t;
                let fix = fix + upper_threshold;
                let fix = fix / plain_mod.value;

                let mut delta_m =
                    mod_mul(random_message, coeffdiv_plain[block_y], &moduli[block_y]);
                delta_m = mod_add(delta_m, fix, &moduli[block_y]);

                let mut ct_ = ct[in_index];
                let error = errors[in_index];
                ct_ = mod_add(ct_, error, &moduli[block_y]);

                if block_z == 0 {
                    ct_ = mod_sub(ct_, delta_m, &moduli[block_y]);
                } else {
                    ct_ = mod_add(ct_, delta_m, &moduli[block_y]);
                }

                ct[in_index] = ct_;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: col_boot_enc — collaborative bootstrapping encryption
// ---------------------------------------------------------------------------

/// CPU reference: col_boot_enc — collaborative bootstrapping encryption step.
///
/// Computes ct[index] = h[index] + ΔM (mod q) where
/// ΔM is the scaled random plaintext, same formula as in col_boot_add_random_and_errors.
///
/// This is the encryption contribution from a single party during collaborative
/// bootstrapping.
///
/// # Parameters
/// - `ct`: output ciphertext buffer, length = ring_size * decomp_mod_count
/// - `h`: input polynomial (e.g., public key share), same length
/// - `random_plain`: random plaintext polynomial, length = ring_size
/// - Other params: same as col_boot_add_random_and_errors
pub fn col_boot_enc_cpu(
    ct: &mut [u64],
    h: &[u64],
    random_plain: &[u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;

    for block_y in 0..decomp_mod_count {
        for idx in 0..ring_size {
            let in_index = idx + (block_y << n_power);

            let h_ = h[in_index];
            let random_message = random_plain[idx];
            let fix = random_message * q_mod_t;
            let fix = fix + upper_threshold;
            let fix = fix / plain_mod.value;

            let mut delta_m =
                mod_mul(random_message, coeffdiv_plain[block_y], &moduli[block_y]);
            delta_m = mod_add(delta_m, fix, &moduli[block_y]);

            ct[in_index] = mod_add(h_, delta_m, &moduli[block_y]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: col_boot_dec_mul_with_sk_ckks — CKKS variant
// ---------------------------------------------------------------------------

/// CPU reference: col_boot_dec_mul_with_sk_ckks — CKKS variant of collaborative
/// bootstrapping secret-key multiplication.
///
/// For block_y < current_decomp_mod_count:
///   output[index] = ct1[index] * sk[index] mod q[block_y]  (c1 * sk)
/// For block_y >= current_decomp_mod_count:
///   Maps to a different modulus offset: offset_block = block_y - current_decomp_mod_count
///   output[index] = -(a[offset_index] * sk[offset_index]) mod q[offset_block]
///
/// The total block_y range is current_decomp_mod_count + decomp_mod_count.
pub fn col_boot_dec_mul_with_sk_ckks_cpu(
    ct1: &[u64],
    a: &[u64],
    sk: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
    current_decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    let total_y = current_decomp_mod_count + decomp_mod_count;

    for block_y in 0..total_y {
        for idx in 0..ring_size {
            let in_index = idx + (block_y << n_power);

            let result = if block_y < current_decomp_mod_count {
                let sk_ = sk[in_index];
                mod_mul(ct1[in_index], sk_, &moduli[block_y])
            } else {
                let offset_block = block_y - current_decomp_mod_count;
                let m_in_index = idx + (offset_block << n_power);
                let sk_ = sk[m_in_index];
                let prod = mod_mul(a[m_in_index], sk_, &moduli[offset_block]);
                mod_sub(0, prod, &moduli[offset_block])
            };

            output[in_index] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: col_boot_add_random_and_errors_ckks — CKKS variant
// ---------------------------------------------------------------------------

/// CPU reference: col_boot_add_random_and_errors_ckks — CKKS variant of
/// collaborative bootstrapping error/random addition.
///
/// For block_y < current_decomp_mod_count:
///   ct[index] += error0[index] - random_plain[index]  (mod q[block_y])
/// For block_y >= current_decomp_mod_count:
///   Uses offset_block = block_y - current_decomp_mod_count
///   ct[index] += error1[offset_index] + random_plain[offset_index]  (mod q[offset_block])
///
/// Unlike the BFV variant, CKKS does not scale the random message by Δ.
pub fn col_boot_add_random_and_errors_ckks_cpu(
    ct: &mut [u64],
    error0: &[u64],
    error1: &[u64],
    random_plain: &[u64],
    moduli: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
    current_decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    let total_y = current_decomp_mod_count + decomp_mod_count;

    for block_y in 0..total_y {
        for idx in 0..ring_size {
            let in_index = idx + (block_y << n_power);

            if block_y < current_decomp_mod_count {
                let mut ct_ = ct[in_index];
                let error = error0[in_index];
                let random_message = random_plain[in_index];

                ct_ = mod_add(ct_, error, &moduli[block_y]);
                ct_ = mod_sub(ct_, random_message, &moduli[block_y]);

                ct[in_index] = ct_;
            } else {
                let offset_block = block_y - current_decomp_mod_count;
                let m_in_index = idx + (offset_block << n_power);

                let mut ct_ = ct[in_index];
                let error = error1[m_in_index];
                let random_message = random_plain[m_in_index];

                ct_ = mod_add(ct_, error, &moduli[offset_block]);
                ct_ = mod_add(ct_, random_message, &moduli[offset_block]);

                ct[in_index] = ct_;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{decode_bfv_cpu, encode_bfv_cpu};
    use crate::modular::Modulus64;
    use gpu_host::cuda_ctx;
    use rand::Rng;

    const P0: u64 = 1152921504606846883;
    const P1: u64 = 1152921504606830593;

    fn make_moduli() -> Vec<Modulus64> {
        vec![Modulus64::new(P0), Modulus64::new(P1)]
    }

    fn mod_values(moduli: &[Modulus64]) -> Vec<u64> {
        moduli.iter().map(|m| m.value).collect()
    }

    fn mod_bits(moduli: &[Modulus64]) -> Vec<u64> {
        moduli.iter().map(|m| m.bit).collect()
    }

    fn mod_mus(moduli: &[Modulus64]) -> Vec<u64> {
        moduli.iter().map(|m| m.mu).collect()
    }

    fn random_rns_poly(ring_size: usize, moduli: &[Modulus64], n_power: u32) -> Vec<u64> {
        let rns_count = moduli.len();
        let mut rng = rand::rng();
        (0..ring_size * rns_count)
            .map(|i| {
                let idy = (i >> n_power) % rns_count;
                rng.random::<u64>() % moduli[idy].value
            })
            .collect()
    }

    #[test]
    fn test_decrypt_cpu_trivial() {
        // c1=0, so decrypt(c0, c1, sk) = c0
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();

        let c0 = random_rns_poly(ring_size, &moduli, n_power);
        let c1 = vec![0u64; ring_size * rns_count];
        let sk = random_rns_poly(ring_size, &moduli, n_power);
        let mut output = vec![0u64; ring_size * rns_count];

        decrypt_cpu(&c0, &c1, &sk, &mut output, &moduli, n_power, rns_count);

        assert_eq!(output, c0, "trivial decrypt with c1=0 should return c0");
    }

    #[test]
    fn test_sk_multiplication_cpu_basic() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();

        let ct1 = random_rns_poly(ring_size, &moduli, n_power);
        let sk = random_rns_poly(ring_size, &moduli, n_power);
        let mut output = vec![0u64; ring_size * rns_count];

        sk_multiplication_cpu(&ct1, &sk, &mut output, &moduli, n_power, rns_count);

        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let index = idx + (block_y << n_power);
                let expected = mod_mul(ct1[index], sk[index], &moduli[block_y]);
                assert_eq!(output[index], expected, "mismatch at y={block_y} x={idx}");
            }
        }
    }

    #[test]
    fn test_sk_multiplication_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();
        let total = ring_size * rns_count;

        let ct1 = random_rns_poly(ring_size, &moduli, n_power);
        let sk = random_rns_poly(ring_size, &moduli, n_power);

        // CPU reference
        let mut cpu_out = vec![0u64; total];
        sk_multiplication_cpu(&ct1, &sk, &mut cpu_out, &moduli, n_power, rns_count);

        // GPU
        let mv = mod_values(&moduli);
        let mb = mod_bits(&moduli);
        let mm = mod_mus(&moduli);
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_ct1 = ctx.new_tensor_view(ct1.as_slice()).expect("alloc");
            let d_sk = ctx.new_tensor_view(sk.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let d_mb = ctx.new_tensor_view(mb.as_slice()).expect("alloc");
            let d_mm = ctx.new_tensor_view(mm.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            sk_multiplication_kernel::launch(
                config,
                ctx,
                m,
                &d_ct1,
                &d_sk,
                &mut d_out,
                &d_mv,
                &d_mb,
                &d_mm,
                n_power,
                rns_count as u32,
            )
            .expect("kernel launch");
            d_out.copy_to_host(&mut gpu_out).expect("copy");
        });

        assert_eq!(gpu_out, cpu_out, "GPU sk_multiplication mismatch");
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip_cpu() {
        // Trivial encryption: c0 = encoded_message, c1 = 0
        // decrypt(c0, c1=0, sk) = c0
        // decode(c0) = original message
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let plain_modulus = 65537u64;
        let moduli = make_moduli();
        let rns_count = moduli.len();

        // Original plaintext
        let messages: Vec<u64> = (0..ring_size as u64).map(|i| i % plain_modulus).collect();

        // Encode
        let mut encoded = vec![0u64; ring_size * rns_count];
        encode_bfv_cpu(&messages, &mut encoded, &moduli, plain_modulus, n_power);

        // Trivial encrypt: c0 = encoded, c1 = 0
        let c0 = encoded.clone();
        let c1 = vec![0u64; ring_size * rns_count];
        let sk = random_rns_poly(ring_size, &moduli, n_power);

        // Decrypt
        let mut decrypted = vec![0u64; ring_size * rns_count];
        decrypt_cpu(&c0, &c1, &sk, &mut decrypted, &moduli, n_power, rns_count);

        // Decode
        let mut result = vec![0u64; ring_size];
        decode_bfv_cpu(&decrypted, &mut result, &moduli[0], plain_modulus, n_power);

        assert_eq!(result, messages, "encode → trivial encrypt → decrypt → decode roundtrip failed");
    }
}
