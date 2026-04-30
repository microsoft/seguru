/// Homomorphic multiplication kernels for BFV ciphertexts.
///
/// Cross-multiplication takes two ciphertexts ct1=(c0,c1) and ct2=(d0,d1) and
/// produces three components: (c0*d0, c0*d1+c1*d0, c1*d1).
///
/// Cipher-plain multiplication element-wise multiplies each ciphertext
/// component by a plaintext polynomial.

use crate::modular::{mod_add, mod_mul, mod_reduce_forced, mod_sub, Modulus64};
use gpu::prelude::*;

// ---------------------------------------------------------------------------
// CPU reference functions
// ---------------------------------------------------------------------------

/// Cross-multiply two ciphertexts producing 3 output components.
///
/// ct1 layout: `[c0_rns0..c0_rnsN, c1_rns0..c1_rnsN]` — length ring_size * rns_count * 2
/// ct2 same layout.
/// output: 3 components — length ring_size * rns_count * 3
///   out0 = c0*d0, out1 = c0*d1 + c1*d0, out2 = c1*d1
pub fn cross_multiplication_cpu(
    ct1: &[u64],
    ct2: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    let ct_offset = rns_count << n_power; // offset between c0 and c1
    for idy in 0..rns_count {
        for idx in 0..ring_size {
            let loc = idx + (idy << n_power);

            let a = ct1[loc];             // c0
            let b = ct1[loc + ct_offset]; // c1
            let c = ct2[loc];             // d0
            let d = ct2[loc + ct_offset]; // d1

            let out0 = mod_mul(a, c, &moduli[idy]);
            let ad = mod_mul(a, d, &moduli[idy]);
            let bc = mod_mul(b, c, &moduli[idy]);
            let out1 = mod_add(ad, bc, &moduli[idy]);
            let out2 = mod_mul(b, d, &moduli[idy]);

            output[loc] = out0;
            output[loc + ct_offset] = out1;
            output[loc + 2 * ct_offset] = out2;
        }
    }
}

/// Element-wise multiply of ciphertext by plaintext polynomial.
///
/// cipher layout: ring_size * rns_count * cipher_count
/// plain layout:  ring_size * rns_count (one polynomial, shared across cipher components)
pub fn cipher_plain_mul_cpu(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_z in 0..cipher_count {
        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let ct_loc = idx + (block_y << n_power) + ((rns_count << n_power) * block_z);
                let pt_loc = idx + (block_y << n_power);
                output[ct_loc] = mod_mul(cipher[ct_loc], plain[pt_loc], &moduli[block_y]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

/// GPU kernel: cipher-plain element-wise Barrett modular multiplication.
///
/// Each thread multiplies one ciphertext element by the corresponding plaintext
/// element. The plaintext wraps around every `rns_count * ring_size` elements.
///
/// Launch with total_elements = ring_size * rns_count * cipher_count threads.
#[gpu::cuda_kernel]
pub fn cipher_plain_mul_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let pt_size = rns_count << n_power; // elements per plaintext
    let pt_loc = gid % pt_size;
    let idy = ((gid >> n_power) % rns_count) as usize;

    let a = cipher[gid as usize];
    let b = plain[pt_loc as usize];

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

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
// GPU kernel: cross_multiplication
// ---------------------------------------------------------------------------

/// GPU kernel: cross-multiply two ciphertexts, producing component c0*d0.
///
/// Launch with ring_size * rns_count threads.
/// ct1/ct2 layout: `[c0_rns0..c0_rnsN, c1_rns0..c1_rnsN]`
#[gpu::cuda_kernel]
pub fn cross_multiplication_out0_kernel(
    ct1: &[u64],
    ct2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let loc = gid as usize;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let a = ct1[loc];
    let c = ct2[loc];

    let z = (a as u128) * (c as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

/// GPU kernel: cross-multiply two ciphertexts, producing component c0*d1 + c1*d0.
///
/// Launch with ring_size * rns_count threads.
#[gpu::cuda_kernel]
pub fn cross_multiplication_out1_kernel(
    ct1: &[u64],
    ct2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let loc = gid as usize;
    let ct_offset = (rns_count << n_power) as usize;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let a = ct1[loc];
    let b = ct1[loc + ct_offset];
    let c = ct2[loc];
    let d = ct2[loc + ct_offset];

    // ad = a * d
    let z = (a as u128) * (d as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut ad = (z - w * (mod_val as u128)) as u64;
    if ad >= mod_val { ad -= mod_val; }

    // bc = b * c
    let z = (b as u128) * (c as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut bc = (z - w * (mod_val as u128)) as u64;
    if bc >= mod_val { bc -= mod_val; }

    // out1 = ad + bc
    let mut r = ad + bc;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

/// GPU kernel: cross-multiply two ciphertexts, producing component c1*d1.
///
/// Launch with ring_size * rns_count threads.
#[gpu::cuda_kernel]
pub fn cross_multiplication_out2_kernel(
    ct1: &[u64],
    ct2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let loc = gid as usize;
    let ct_offset = (rns_count << n_power) as usize;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let b = ct1[loc + ct_offset];
    let d = ct2[loc + ct_offset];

    let z = (b as u128) * (d as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// CPU-only: fast_convertion (complex base conversion with inner loops)
// ---------------------------------------------------------------------------

/// CPU-only reference for HEonGPU's `fast_convertion` kernel.
///
/// Performs RNS base conversion from `ibase` to `obase` using the
/// Behz-style lifting through an auxiliary modulus `m_tilde`.
///
/// For each coefficient position and cipher half (determined by `idy`):
///   1. Multiply input by `m_tilde` and `inv_punctured_prod` in each ibase limb.
///   2. Base-change to obase via matrix multiply.
///   3. Base-change to m_tilde for correction term.
///   4. Apply Shenoy-Kumaresan correction.
///   5. Write [ibase_original | obase_converted] to output.
pub fn fast_convertion_cpu(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    ibase: &[Modulus64],
    obase: &[Modulus64],
    m_tilde: &Modulus64,
    inv_prod_q_mod_m_tilde: u64,
    inv_m_tilde_mod_bsk: &[u64],
    prod_q_mod_bsk: &[u64],
    base_change_matrix_bsk: &[u64],
    base_change_matrix_m_tilde: &[u64],
    inv_punctured_prod_mod_base_array: &[u64],
    n_power: u32,
    ibase_size: usize,
    obase_size: usize,
    cipher_count_times_2: usize,
) {
    let ring_size = 1usize << n_power;
    for idy in 0..cipher_count_times_2 {
        let input = if (idy >> 1) == 0 { in1 } else { in2 };
        let base_loc = (idy % 2) * ibase_size;
        for idx in 0..ring_size {
            let location = idx + (base_loc << n_power);

            // Step 1: read input, multiply by m_tilde and inv_punctured_prod
            let mut temp_orig = vec![0u64; ibase_size];
            let mut temp = vec![0u64; ibase_size];
            for i in 0..ibase_size {
                temp_orig[i] = input[location + (i << n_power)];
                temp[i] = mod_mul(temp_orig[i], m_tilde.value, &ibase[i]);
                temp[i] = mod_mul(temp[i], inv_punctured_prod_mod_base_array[i], &ibase[i]);
            }

            // Step 2: base change to obase (Bsk)
            let mut temp2 = vec![0u64; obase_size + 1];
            for i in 0..obase_size {
                temp2[i] = 0;
                for j in 0..ibase_size {
                    let mult = mod_mul(
                        temp[j],
                        base_change_matrix_bsk[j + i * ibase_size],
                        &obase[i],
                    );
                    temp2[i] = mod_add(temp2[i], mult, &obase[i]);
                }
            }

            // Step 3: base change to m_tilde
            temp2[obase_size] = 0;
            for j in 0..ibase_size {
                let temp_in = mod_reduce_forced(temp[j], m_tilde);
                let mult = mod_mul(temp_in, base_change_matrix_m_tilde[j], m_tilde);
                temp2[obase_size] = mod_add(temp2[obase_size], mult, m_tilde);
            }

            // Step 4: Shenoy-Kumaresan correction
            let m_tilde_div_2 = m_tilde.value >> 1;
            let mut r_m_tilde = mod_mul(temp2[obase_size], inv_prod_q_mod_m_tilde, m_tilde);
            r_m_tilde = m_tilde.value - r_m_tilde;

            for i in 0..obase_size {
                let mut temp3 = r_m_tilde;
                if temp3 >= m_tilde_div_2 {
                    temp3 = obase[i].value - m_tilde.value;
                    temp3 = mod_add(temp3, r_m_tilde, &obase[i]);
                }
                temp3 = mod_mul(temp3, prod_q_mod_bsk[i], &obase[i]);
                temp3 = mod_add(temp2[i], temp3, &obase[i]);
                temp2[i] = mod_mul(temp3, inv_m_tilde_mod_bsk[i], &obase[i]);
            }

            // Step 5: write output [ibase_original | obase_converted]
            let location2 = idx + (idy * (obase_size + ibase_size)) * ring_size;
            for i in 0..ibase_size {
                output[location2 + (i << n_power)] = temp_orig[i];
            }
            for i in 0..obase_size {
                output[location2 + ((i + ibase_size) << n_power)] = temp2[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-only: fast_floor (complex base conversion with multiple loops)
// ---------------------------------------------------------------------------

/// CPU-only reference for HEonGPU's `fast_floor` kernel.
///
/// Performs the "floor" step of BFV decryption/rescaling: converts from
/// combined q∪Bsk representation, applies plain_modulus scaling, and
/// performs Shenoy-Kumaresan-style base conversion back to q-base.
pub fn fast_floor_cpu(
    in_baseq_bsk: &[u64],
    output: &mut [u64],
    ibase: &[Modulus64],
    obase: &[Modulus64],
    plain_modulus: &Modulus64,
    inv_punctured_prod_mod_base_array: &[u64],
    base_change_matrix_bsk: &[u64],
    inv_prod_q_mod_bsk: &[u64],
    inv_punctured_prod_mod_b_array: &[u64],
    base_change_matrix_q: &[u64],
    base_change_matrix_msk: &[u64],
    inv_prod_b_mod_m_sk: u64,
    prod_b_mod_q: &[u64],
    n_power: u32,
    ibase_size: usize,
    obase_size: usize,
    component_count: usize,
) {
    let ring_size = 1usize << n_power;
    let combined = ibase_size + obase_size;
    let msk = &obase[obase_size - 1]; // last obase entry is m_sk

    for idy in 0..component_count {
        for idx in 0..ring_size {
            let location_q = idx + (idy * combined) * ring_size;
            let location_bsk = location_q + (ibase_size << n_power);

            // Step 1: multiply q-part by plain_modulus, then by inv_punctured_prod
            let mut reg_q = vec![0u64; ibase_size];
            for i in 0..ibase_size {
                reg_q[i] = mod_mul(
                    in_baseq_bsk[location_q + (i << n_power)],
                    plain_modulus.value,
                    &ibase[i],
                );
                reg_q[i] = mod_mul(
                    reg_q[i],
                    inv_punctured_prod_mod_base_array[i],
                    &ibase[i],
                );
            }

            // Step 2: multiply Bsk-part by plain_modulus
            let mut reg_bsk = vec![0u64; obase_size];
            for i in 0..obase_size {
                reg_bsk[i] = mod_mul(
                    in_baseq_bsk[location_bsk + (i << n_power)],
                    plain_modulus.value,
                    &obase[i],
                );
            }

            // Step 3: base change q -> Bsk
            let mut temp = vec![0u64; obase_size];
            for i in 0..obase_size {
                temp[i] = 0;
                for j in 0..ibase_size {
                    let mult = mod_mul(
                        reg_q[j],
                        base_change_matrix_bsk[j + i * ibase_size],
                        &obase[i],
                    );
                    temp[i] = mod_add(temp[i], mult, &obase[i]);
                }
            }

            // Step 4: subtract and multiply by inv_prod_q_mod_Bsk
            for i in 0..obase_size {
                let temp2 = mod_sub(obase[i].value, temp[i], &obase[i]);
                let temp2 = mod_add(temp2, reg_bsk[i], &obase[i]);
                reg_bsk[i] = mod_mul(temp2, inv_prod_q_mod_bsk[i], &obase[i]);
            }

            // Step 5: multiply B-part by inv_punctured_prod_mod_B
            let mut temp3 = vec![0u64; obase_size - 1];
            for i in 0..(obase_size - 1) {
                temp3[i] = mod_mul(
                    reg_bsk[i],
                    inv_punctured_prod_mod_b_array[i],
                    &obase[i],
                );
            }

            // Step 6: base change B -> q
            let mut temp4 = vec![0u64; ibase_size + 1];
            for i in 0..ibase_size {
                temp4[i] = 0;
                for j in 0..(obase_size - 1) {
                    let t = mod_reduce_forced(temp3[j], &ibase[i]);
                    let mult = mod_mul(
                        t,
                        base_change_matrix_q[j + i * (obase_size - 1)],
                        &ibase[i],
                    );
                    let mult = mod_reduce_forced(mult, &ibase[i]);
                    temp4[i] = mod_add(temp4[i], mult, &ibase[i]);
                }
            }

            // Step 7: base change B -> m_sk
            temp4[ibase_size] = 0;
            for j in 0..(obase_size - 1) {
                let mult = mod_mul(temp3[j], base_change_matrix_msk[j], msk);
                temp4[ibase_size] = mod_add(temp4[ibase_size], mult, msk);
            }

            // Step 8: compute alpha_sk correction
            let alpha_sk = mod_sub(msk.value, reg_bsk[obase_size - 1], msk);
            let alpha_sk = mod_add(alpha_sk, temp4[ibase_size], msk);
            let alpha_sk = mod_mul(alpha_sk, inv_prod_b_mod_m_sk, msk);

            let m_sk_div_2 = msk.value >> 1;

            // Step 9: apply correction to each q-limb
            for i in 0..ibase_size {
                let obase_r = mod_reduce_forced(msk.value, &ibase[i]);
                let temp4_r = mod_reduce_forced(temp4[i], &ibase[i]);
                let alpha_sk_r = mod_reduce_forced(alpha_sk, &ibase[i]);
                if alpha_sk > m_sk_div_2 {
                    let inner = mod_sub(obase_r, alpha_sk_r, &ibase[i]);
                    let inner = mod_mul(inner, prod_b_mod_q[i], &ibase[i]);
                    temp4[i] = mod_add(temp4_r, inner, &ibase[i]);
                } else {
                    let inner = mod_sub(ibase[i].value, prod_b_mod_q[i], &ibase[i]);
                    let inner = mod_mul(inner, alpha_sk_r, &ibase[i]);
                    temp4[i] = mod_add(temp4_r, inner, &ibase[i]);
                }
            }

            // Step 10: write output
            let location_out = idx + (idy * ibase_size) * ring_size;
            for i in 0..ibase_size {
                output[location_out + (i << n_power)] = temp4[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: threshold_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: if plain value >= threshold, add upper_half_increment; else copy.
///
/// Launch with ring_size * decomp_size threads.
/// plain_in has ring_size elements (shared across all RNS limbs).
#[gpu::cuda_kernel]
pub fn threshold_kernel(
    plain_in: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    plain_upper_half_increment: &[u64],
    plain_upper_half_threshold: u64,
    n_power: u32,
    _decomp_size: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = (gid >> n_power) as usize;

    let plain_reg = plain_in[idx as usize];

    let mod_val = mod_values[block_y];
    let result = if plain_reg >= plain_upper_half_threshold {
        let sum = plain_reg + plain_upper_half_increment[block_y];
        if sum >= mod_val { sum - mod_val } else { sum }
    } else {
        plain_reg
    };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipherplain_kernel (3D grid variant)
// ---------------------------------------------------------------------------

/// GPU kernel: cipher-plain multiplication with 3D grid (rns × cipher components).
///
/// Launch with ring_size * decomp_size * cipher_size threads.
#[gpu::cuda_kernel]
pub fn cipherplain_kernel(
    cipher: &[u64],
    plain_in: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let pt_size = rns_count << n_power;
    let pt_loc = gid % pt_size;
    let idy = ((gid >> n_power) % rns_count) as usize;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let a = cipher[gid as usize];
    let b = plain_in[pt_loc as usize];

    let z = (a as u128) * (b as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// CPU-only: cipher_constant_plain_multiplication (f64 → bigint → multiply)
// ---------------------------------------------------------------------------

/// CPU-only: multiply each ciphertext coefficient by a double-precision constant.
///
/// Converts `message` to a 128-bit integer, reduces mod each RNS prime,
/// then multiplies element-wise. Mirrors HEonGPU's
/// `cipher_constant_plain_multiplication_kernel`.
pub fn cipher_constant_plain_multiplication_cpu(
    input: &[u64],
    message: f64,
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    let two_pow_64: f64 = (1u128 << 64) as f64;

    let coeff_double = message.round();
    let is_negative = coeff_double.is_sign_negative();
    let coeff_abs = coeff_double.abs();

    let coeff_lo = (coeff_abs % two_pow_64) as u64;
    let coeff_hi = (coeff_abs / two_pow_64) as u64;
    let coeff_128 = (coeff_hi as u128) << 64 | (coeff_lo as u128);

    for bz in 0..cipher_count {
        for by in 0..rns_count {
            // Reduce 128-bit coeff mod this prime
            let pt_reduced = (coeff_128 % (moduli[by].value as u128)) as u64;
            let pt = if is_negative {
                mod_sub(moduli[by].value, pt_reduced, &moduli[by])
            } else {
                pt_reduced
            };

            for idx in 0..ring_size {
                let loc = idx + (by << n_power) + ((rns_count << n_power) * bz);
                output[loc] = mod_mul(input[loc], pt, &moduli[by]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: cipherplain_multiply_accumulate_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: inner product of cipher × plain across iterations.
///
/// For each output element, computes:
///   sum = Σ_i cipher[loc + i*ct_stride] * plain[loc_pt + i*pt_stride] mod p
///
/// Launch with ring_size * rns_count * cipher_count threads.
#[gpu::cuda_kernel]
pub fn cipherplain_multiply_accumulate_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
    iteration_count: u32,
    current_decomp_count: u32,
    first_decomp_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;
    let block_z = (gid >> n_power) / rns_count;

    let location_ct =
        (idx + ((block_y as u32) << n_power) + ((rns_count * block_z) << n_power)) as usize;
    let location_pt = (idx + ((block_y as u32) << n_power)) as usize;

    let offset_ct = (current_decomp_count << (n_power + 1)) as usize;
    let offset_pt = (first_decomp_count << n_power) as usize;

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let mut sum: u64 = 0;
    let mut i = 0u32;
    while i < iteration_count {
        let ct = cipher[location_ct + (i as usize) * offset_ct];
        let pt = plain[location_pt + (i as usize) * offset_pt];

        let z = (ct as u128) * (pt as u128);
        let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
        let mut mul_ctpt = (z - w * (mod_val as u128)) as u64;
        if mul_ctpt >= mod_val { mul_ctpt -= mod_val; }

        sum = sum + mul_ctpt;
        if sum >= mod_val { sum -= mod_val; }

        i += 1;
    }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = sum;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipherplain_multiply_accumulate_indexed_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: indexed inner product of cipher × plain.
///
/// Like `cipherplain_multiply_accumulate_kernel` but reads cipher data at
/// positions given by `ct_indices[i]` rather than sequential offsets.
#[gpu::cuda_kernel]
pub fn cipherplain_multiply_accumulate_indexed_kernel(
    baby_results: &[u64],
    plaintexts: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    ct_indices: &[u32],
    n_power: u32,
    rns_count: u32,
    iteration_count: u32,
    current_decomp_count: u32,
    first_decomp_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;
    let block_z = (gid >> n_power) / rns_count;

    let location_out =
        (idx + ((block_y as u32) << n_power) + ((rns_count * block_z) << n_power)) as usize;
    let location_pt = (idx + ((block_y as u32) << n_power)) as usize;

    let ct_stride = (current_decomp_count << (n_power + 1)) as usize;
    let offset_pt = (first_decomp_count << n_power) as usize;

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let mut sum: u64 = 0;
    let mut i = 0u32;
    while i < iteration_count {
        let ct_offset = (ct_indices[i as usize] as usize) * ct_stride;
        let ct = baby_results[location_out + ct_offset];
        let pt = plaintexts[location_pt + (i as usize) * offset_pt];

        let z = (ct as u128) * (pt as u128);
        let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
        let mut mul_ctpt = (z - w * (mod_val as u128)) as u64;
        if mul_ctpt >= mod_val { mul_ctpt -= mod_val; }

        sum = sum + mul_ctpt;
        if sum >= mod_val { sum -= mod_val; }

        i += 1;
    }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = sum;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipher_div_by_i_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: divide ciphertext by imaginary unit in NTT domain.
///
/// For idx < ring_size/2: multiply by -psi (negate psi first).
/// For idx >= ring_size/2: multiply by psi.
/// psi = ntt_table[1 + block_y * ring_size].
#[gpu::cuda_kernel]
pub fn cipher_div_by_i_kernel(
    input: &[u64],
    output: &mut [u64],
    ntt_table: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;

    let location_ct = gid as usize;
    let location_psi = 1 + (block_y << n_power);

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let ct = input[location_ct];
    let psi = ntt_table[location_psi];

    let half_ring = 1u32 << (n_power - 1);
    let effective_psi = if idx < half_ring {
        // neg_psi = mod_val - psi
        mod_val - psi
    } else {
        psi
    };

    let z = (ct as u128) * (effective_psi as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipher_mult_by_i_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: multiply ciphertext by imaginary unit in NTT domain.
///
/// For idx < ring_size/2: multiply by psi.
/// For idx >= ring_size/2: multiply by -psi.
#[gpu::cuda_kernel]
pub fn cipher_mult_by_i_kernel(
    input: &[u64],
    output: &mut [u64],
    ntt_table: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;

    let location_ct = gid as usize;
    let location_psi = 1 + (block_y << n_power);

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let ct = input[location_ct];
    let psi = ntt_table[location_psi];

    let half_ring = 1u32 << (n_power - 1);
    let effective_psi = if idx < half_ring {
        psi
    } else {
        mod_val - psi
    };

    let z = (ct as u128) * (effective_psi as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipher_mult_by_gaussian_integer_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: multiply ciphertext by gaussian integer (a + bi) in NTT domain.
///
/// const_imag = c_imag * psi mod p
/// For idx < ring_size/2: ct *= (c_real + const_imag) mod p
/// For idx >= ring_size/2: ct *= (c_real - const_imag) mod p
#[gpu::cuda_kernel]
pub fn cipher_mult_by_gaussian_integer_kernel(
    input: &[u64],
    real_rns: &[u64],
    imag_rns: &[u64],
    output: &mut [u64],
    ntt_table: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;

    let location_ct = gid as usize;
    let location_psi = 1 + (block_y << n_power);

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let ct = input[location_ct];
    let psi = ntt_table[location_psi];
    let c_real = real_rns[block_y];
    let c_imag = imag_rns[block_y];

    // const_imag = c_imag * psi mod p
    let z = (c_imag as u128) * (psi as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut const_imag = (z - w * (mod_val as u128)) as u64;
    if const_imag >= mod_val { const_imag -= mod_val; }

    let half_ring = 1u32 << (n_power - 1);
    let scaled_const = if idx < half_ring {
        // c_real + const_imag
        let s = c_real + const_imag;
        if s >= mod_val { s - mod_val } else { s }
    } else {
        // c_real - const_imag
        let s = c_real + mod_val - const_imag;
        if s >= mod_val { s - mod_val } else { s }
    };

    let z = (ct as u128) * (scaled_const as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut r = (z - w * (mod_val as u128)) as u64;
    if r >= mod_val { r -= mod_val; }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipher_add_by_gaussian_integer_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: add gaussian integer (a + bi) to the first ciphertext component.
///
/// Only adds to block_z == 0 (first cipher component); other components pass through.
#[gpu::cuda_kernel]
pub fn cipher_add_by_gaussian_integer_kernel(
    input: &[u64],
    real_rns: &[u64],
    imag_rns: &[u64],
    output: &mut [u64],
    ntt_table: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;
    let block_z = (gid >> n_power) / rns_count;

    let location_ct = gid as usize;
    let location_psi = 1 + (block_y << n_power);

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let mut ct = input[location_ct];
    let psi = ntt_table[location_psi];
    let c_real = real_rns[block_y];
    let c_imag = imag_rns[block_y];

    // const_imag = c_imag * psi mod p
    let z = (c_imag as u128) * (psi as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut const_imag = (z - w * (mod_val as u128)) as u64;
    if const_imag >= mod_val { const_imag -= mod_val; }

    if block_z == 0 {
        let half_ring = 1u32 << (n_power - 1);
        let scaled_const = if idx < half_ring {
            let s = c_real + const_imag;
            if s >= mod_val { s - mod_val } else { s }
        } else {
            let s = c_real + mod_val - const_imag;
            if s >= mod_val { s - mod_val } else { s }
        };

        ct = ct + scaled_const;
        if ct >= mod_val { ct -= mod_val; }
    }

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = ct;
}

// ---------------------------------------------------------------------------
// GPU kernel: cipher_mult_by_gaussian_integer_and_add_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: multiply ciphertext by gaussian integer and accumulate.
///
/// accumulator[loc] += ct * (c_real ± c_imag*psi) mod p
#[gpu::cuda_kernel]
pub fn cipher_mult_by_gaussian_integer_and_add_kernel(
    input: &[u64],
    real_rns: &[u64],
    imag_rns: &[u64],
    accumulator: &mut [u64],
    ntt_table: &[u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = gid & ((1u32 << n_power) - 1);
    let block_y = ((gid >> n_power) % rns_count) as usize;

    let location_ct = gid as usize;
    let location_psi = 1 + (block_y << n_power);

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let ct = input[location_ct];
    let psi = ntt_table[location_psi];
    let c_real = real_rns[block_y];
    let c_imag = imag_rns[block_y];

    // const_imag = c_imag * psi mod p
    let z = (c_imag as u128) * (psi as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut const_imag = (z - w * (mod_val as u128)) as u64;
    if const_imag >= mod_val { const_imag -= mod_val; }

    let half_ring = 1u32 << (n_power - 1);
    let scaled_const = if idx < half_ring {
        let s = c_real + const_imag;
        if s >= mod_val { s - mod_val } else { s }
    } else {
        let s = c_real + mod_val - const_imag;
        if s >= mod_val { s - mod_val } else { s }
    };

    // mult_result = ct * scaled_const mod p
    let z = (ct as u128) * (scaled_const as u128);
    let w = (z >> (bit as u32 - 2)) * (mu as u128) >> (bit as u32 + 3);
    let mut mult_result = (z - w * (mod_val as u128)) as u64;
    if mult_result >= mod_val { mult_result -= mod_val; }

    // accumulator += mult_result
    let mut acc = chunk_mut(accumulator, MapLinear::new(1));
    let mut result = acc[0] + mult_result;
    if result >= mod_val { result -= mod_val; }
    acc[0] = result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addition::{addition_kernel, multiply_elementwise_kernel};
    use crate::modular::Modulus64;
    use gpu_host::cuda_ctx;
    use rand::Rng;

    const P0: u64 = 1152921504606846883;
    const P1: u64 = 1152921504606830593;

    fn make_moduli() -> Vec<Modulus64> {
        vec![Modulus64::new(P0), Modulus64::new(P1)]
    }

    fn random_poly(len: usize, moduli: &[Modulus64], n_power: u32) -> Vec<u64> {
        let mut rng = rand::rng();
        let rns_count = moduli.len();
        (0..len)
            .map(|i| {
                let idy = (i >> n_power) % rns_count;
                rng.random::<u64>() % moduli[idy].value
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // CPU tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_multiplication_cpu() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let ct_offset = rns_count * ring_size;
        let moduli = make_moduli();

        let ct1 = random_poly(ring_size * rns_count * 2, &moduli, n_power);
        let ct2 = random_poly(ring_size * rns_count * 2, &moduli, n_power);
        let mut output = vec![0u64; ring_size * rns_count * 3];

        cross_multiplication_cpu(&ct1, &ct2, &mut output, &moduli, n_power, rns_count);

        // Verify each element
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power);
                let a = ct1[loc];
                let b = ct1[loc + ct_offset];
                let c = ct2[loc];
                let d = ct2[loc + ct_offset];

                let exp0 = mod_mul(a, c, &moduli[idy]);
                let exp1 = mod_add(
                    mod_mul(a, d, &moduli[idy]),
                    mod_mul(b, c, &moduli[idy]),
                    &moduli[idy],
                );
                let exp2 = mod_mul(b, d, &moduli[idy]);

                assert_eq!(output[loc], exp0, "out0 mismatch at ({idx},{idy})");
                assert_eq!(
                    output[loc + ct_offset],
                    exp1,
                    "out1 mismatch at ({idx},{idy})"
                );
                assert_eq!(
                    output[loc + 2 * ct_offset],
                    exp2,
                    "out2 mismatch at ({idx},{idy})"
                );
            }
        }
    }

    #[test]
    fn test_cipher_plain_mul_cpu() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 2usize;
        let moduli = make_moduli();

        let cipher = random_poly(ring_size * rns_count * cipher_count, &moduli, n_power);
        let plain = random_poly(ring_size * rns_count, &moduli, n_power);
        let mut output = vec![0u64; ring_size * rns_count * cipher_count];

        cipher_plain_mul_cpu(
            &cipher,
            &plain,
            &mut output,
            &moduli,
            n_power,
            rns_count,
            cipher_count,
        );

        for bz in 0..cipher_count {
            for by in 0..rns_count {
                for idx in 0..ring_size {
                    let ct_loc = idx + (by << n_power) + ((rns_count << n_power) * bz);
                    let pt_loc = idx + (by << n_power);
                    let expected = mod_mul(cipher[ct_loc], plain[pt_loc], &moduli[by]);
                    assert_eq!(output[ct_loc], expected, "mismatch at ({idx},{by},{bz})");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU vs CPU tests
    // -----------------------------------------------------------------------

    /// Test cross-multiplication on GPU by composing existing elementwise
    /// multiply and addition kernels, then comparing against CPU reference.
    #[test]
    fn test_cross_multiplication_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let ct_offset = rns_count * ring_size;
        let moduli = make_moduli();

        // ct1 = (c0, c1), ct2 = (d0, d1)
        let ct1 = random_poly(ring_size * rns_count * 2, &moduli, n_power);
        let ct2 = random_poly(ring_size * rns_count * 2, &moduli, n_power);

        // CPU reference
        let mut cpu_out = vec![0u64; ring_size * rns_count * 3];
        cross_multiplication_cpu(&ct1, &ct2, &mut cpu_out, &moduli, n_power, rns_count);

        // GPU: compose cross-multiply from elementwise kernels
        // Split ct1 into c0, c1 and ct2 into d0, d1
        let c0 = &ct1[..ct_offset];
        let c1 = &ct1[ct_offset..];
        let d0 = &ct2[..ct_offset];
        let d1 = &ct2[ct_offset..];

        let mv: Vec<u64> = moduli.iter().map(|m| m.value).collect();
        let mb: Vec<u64> = moduli.iter().map(|m| m.bit).collect();
        let mm: Vec<u64> = moduli.iter().map(|m| m.mu).collect();

        // out0 = c0*d0, temp_ad = c0*d1, temp_bc = c1*d0, out1 = ad+bc, out2 = c1*d1
        let mut gpu_out0 = vec![0u64; ct_offset];
        let mut gpu_temp_ad = vec![0u64; ct_offset];
        let mut gpu_temp_bc = vec![0u64; ct_offset];
        let mut gpu_out1 = vec![0u64; ct_offset];
        let mut gpu_out2 = vec![0u64; ct_offset];

        let block_size = 256u32;
        let grid_size = (ct_offset as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_c0 = ctx.new_tensor_view(c0).expect("alloc");
            let d_c1 = ctx.new_tensor_view(c1).expect("alloc");
            let d_d0 = ctx.new_tensor_view(d0).expect("alloc");
            let d_d1 = ctx.new_tensor_view(d1).expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let d_mb = ctx.new_tensor_view(mb.as_slice()).expect("alloc");
            let d_mm = ctx.new_tensor_view(mm.as_slice()).expect("alloc");

            // out0 = c0 * d0
            let mut d_out0 = ctx
                .new_tensor_view(gpu_out0.as_mut_slice())
                .expect("alloc");
            let cfg = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            multiply_elementwise_kernel::launch(
                cfg,
                ctx,
                m,
                &d_c0,
                &d_d0,
                &mut d_out0,
                &d_mv,
                &d_mb,
                &d_mm,
                n_power,
                rns_count as u32,
            )
            .expect("kernel");

            // temp_ad = c0 * d1
            let mut d_ad = ctx
                .new_tensor_view(gpu_temp_ad.as_mut_slice())
                .expect("alloc");
            let cfg = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            multiply_elementwise_kernel::launch(
                cfg,
                ctx,
                m,
                &d_c0,
                &d_d1,
                &mut d_ad,
                &d_mv,
                &d_mb,
                &d_mm,
                n_power,
                rns_count as u32,
            )
            .expect("kernel");

            // temp_bc = c1 * d0
            let mut d_bc = ctx
                .new_tensor_view(gpu_temp_bc.as_mut_slice())
                .expect("alloc");
            let cfg = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            multiply_elementwise_kernel::launch(
                cfg,
                ctx,
                m,
                &d_c1,
                &d_d0,
                &mut d_bc,
                &d_mv,
                &d_mb,
                &d_mm,
                n_power,
                rns_count as u32,
            )
            .expect("kernel");

            // out1 = temp_ad + temp_bc
            d_ad.copy_to_host(&mut gpu_temp_ad).expect("copy");
            d_bc.copy_to_host(&mut gpu_temp_bc).expect("copy");
            let d_ad_r = ctx
                .new_tensor_view(gpu_temp_ad.as_slice())
                .expect("alloc");
            let d_bc_r = ctx
                .new_tensor_view(gpu_temp_bc.as_slice())
                .expect("alloc");
            let mut d_out1 = ctx
                .new_tensor_view(gpu_out1.as_mut_slice())
                .expect("alloc");
            let cfg = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            addition_kernel::launch(
                cfg,
                ctx,
                m,
                &d_ad_r,
                &d_bc_r,
                &mut d_out1,
                &d_mv,
                n_power,
                rns_count as u32,
            )
            .expect("kernel");

            // out2 = c1 * d1
            let mut d_out2 = ctx
                .new_tensor_view(gpu_out2.as_mut_slice())
                .expect("alloc");
            let cfg = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            multiply_elementwise_kernel::launch(
                cfg,
                ctx,
                m,
                &d_c1,
                &d_d1,
                &mut d_out2,
                &d_mv,
                &d_mb,
                &d_mm,
                n_power,
                rns_count as u32,
            )
            .expect("kernel");

            // Copy results back
            d_out0.copy_to_host(&mut gpu_out0).expect("copy");
            d_out1.copy_to_host(&mut gpu_out1).expect("copy");
            d_out2.copy_to_host(&mut gpu_out2).expect("copy");
        });

        // Reassemble into contiguous output matching CPU layout
        let mut gpu_out = vec![0u64; ring_size * rns_count * 3];
        gpu_out[..ct_offset].copy_from_slice(&gpu_out0);
        gpu_out[ct_offset..2 * ct_offset].copy_from_slice(&gpu_out1);
        gpu_out[2 * ct_offset..3 * ct_offset].copy_from_slice(&gpu_out2);

        assert_eq!(gpu_out, cpu_out, "GPU cross-multiplication mismatch");
    }

    #[test]
    fn test_cipher_plain_mul_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 2usize;
        let total = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let cipher = random_poly(total, &moduli, n_power);
        let plain = random_poly(ring_size * rns_count, &moduli, n_power);

        // CPU reference
        let mut cpu_out = vec![0u64; total];
        cipher_plain_mul_cpu(
            &cipher,
            &plain,
            &mut cpu_out,
            &moduli,
            n_power,
            rns_count,
            cipher_count,
        );

        // GPU
        let mv: Vec<u64> = moduli.iter().map(|m| m.value).collect();
        let mb: Vec<u64> = moduli.iter().map(|m| m.bit).collect();
        let mm: Vec<u64> = moduli.iter().map(|m| m.mu).collect();
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_cipher = ctx.new_tensor_view(cipher.as_slice()).expect("alloc");
            let d_plain = ctx.new_tensor_view(plain.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let d_mb = ctx.new_tensor_view(mb.as_slice()).expect("alloc");
            let d_mm = ctx.new_tensor_view(mm.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            cipher_plain_mul_kernel::launch(
                config,
                ctx,
                m,
                &d_cipher,
                &d_plain,
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

        assert_eq!(gpu_out, cpu_out, "GPU cipher-plain mul mismatch");
    }
}
