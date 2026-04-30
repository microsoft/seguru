/// Key-switching (relinearization) kernels ported from HEonGPU's `switchkey.cu`.
///
/// This module contains all 37 kernels from the original CUDA file. Kernels that
/// produce a single output per thread are ported as `#[gpu::cuda_kernel]` GPU
/// kernels. Complex kernels (loops, multiple outputs, f64/f32 rounding, base
/// conversion) are provided as CPU reference implementations.
///
/// Data layout follows HEonGPU conventions:
/// - Ring elements are stored in flat arrays indexed by
///   `idx + (rns_level << n_power) + ((rns_count * cipher_component) << n_power)`
/// - `n_power = log2(ring_size)`

use crate::modular::{mod_add, mod_mul, mod_reduce_forced, mod_sub, Modulus64};

// ===========================================================================
// Helper: inline Barrett multiplication for CPU reference functions
// ===========================================================================

#[inline]
fn cpu_mod_reduce(a: u64, modulus: &Modulus64) -> u64 {
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

// ===========================================================================
// 1. cipher_broadcast_kernel — CPU reference (loop over rns_mod_count outputs)
// ===========================================================================

/// Broadcasts input coefficients across RNS limbs with modular reduction.
///
/// For each `(idx, block_y)`, copies `input[idx + (block_y << n_power)]` into
/// `rns_mod_count` consecutive output limbs, reducing by `modulus[i]`.
pub fn cipher_broadcast_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    rns_mod_count: usize,
    decomp_count: usize,
    ring_size: usize,
) {
    for block_y in 0..decomp_count {
        let location = (rns_mod_count * block_y) << n_power;
        for idx in 0..ring_size {
            let input_ = input[idx + (block_y << n_power)];
            for i in 0..rns_mod_count {
                output[idx + (i << n_power) + location] =
                    mod_mul(1, input_, &modulus[i]);
            }
        }
    }
}

// ===========================================================================
// 2. cipher_broadcast_leveled_kernel — CPU reference (loop + conditional mod)
// ===========================================================================

/// Leveled broadcast: distributes input across current RNS limbs with level-aware
/// modulus selection.
pub fn cipher_broadcast_leveled_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    first_rns_mod_count: usize,
    current_rns_mod_count: usize,
    n_power: u32,
    current_decomp_count: usize,
    ring_size: usize,
) {
    let level = first_rns_mod_count - current_rns_mod_count;
    for block_y in 0..current_decomp_count {
        let location = (current_rns_mod_count * block_y) << n_power;
        for idx in 0..ring_size {
            let input_ = input[idx + (block_y << n_power)];
            for i in 0..current_rns_mod_count {
                let mod_index = if i < current_decomp_count {
                    i
                } else {
                    i + level
                };
                let result = mod_reduce_forced(input_, &modulus[mod_index]);
                output[idx + (i << n_power) + location] = result;
            }
        }
    }
}

// ===========================================================================
// 3. keyswitch_multiply_accumulate_kernel — CPU reference (2 outputs, loops)
// ===========================================================================

/// Key-switch multiply-accumulate: inner product of decomposed ciphertext with
/// relinearization key, producing two output polynomials (ct0, ct1).
pub fn keyswitch_multiply_accumulate_cpu(
    input: &[u64],
    relinkey: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    q_tilda_size: usize,
    rns_mod_count: usize,
    ring_size: usize,
    iteration_count1: usize,
    iteration_count2: usize,
) {
    let key_offset1 = q_tilda_size << n_power;
    let key_offset2 = q_tilda_size << (n_power + 1);

    for block_y in 0..rns_mod_count {
        let modulus_reg = &modulus[block_y];
        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let mut ct_0_sum: u64 = 0;
            let mut ct_1_sum: u64 = 0;

            for i in 0..iteration_count1 {
                for k in 0..4usize {
                    let piece_idx = 4 * i + k;
                    let in_piece =
                        input[index + ((piece_idx * q_tilda_size) << n_power)];
                    let rk0 = relinkey[index + (key_offset2 * piece_idx)];
                    let rk1 =
                        relinkey[index + (key_offset2 * piece_idx) + key_offset1];

                    let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                    let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                    ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                    ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
                }
            }

            let loop_offset = iteration_count1 * 4;
            for i in loop_offset..loop_offset + iteration_count2 {
                let in_piece = input[index + ((i * q_tilda_size) << n_power)];
                let rk0 = relinkey[index + (key_offset2 * i)];
                let rk1 = relinkey[index + (key_offset2 * i) + key_offset1];

                let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
            }

            output[index] = ct_0_sum;
            output[index + key_offset1] = ct_1_sum;
        }
    }
}

// ===========================================================================
// 4. keyswitch_multiply_accumulate_leveled_kernel — CPU reference
// ===========================================================================

/// Leveled key-switch multiply-accumulate with level-aware key indexing.
pub fn keyswitch_multiply_accumulate_leveled_cpu(
    input: &[u64],
    relinkey: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    first_rns_mod_count: usize,
    current_decomp_mod_count: usize,
    iteration_count1: usize,
    iteration_count2: usize,
    n_power: u32,
    rns_mod_count_y: usize,
    ring_size: usize,
) {
    let current_decomp_mod_count_plus_one = current_decomp_mod_count + 1;
    let key_offset1 = first_rns_mod_count << n_power;
    let key_offset2 = first_rns_mod_count << (n_power + 1);

    for block_y in 0..rns_mod_count_y {
        let key_index = if block_y == current_decomp_mod_count {
            first_rns_mod_count - 1
        } else {
            block_y
        };
        let modulus_reg = &modulus[key_index];

        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let mut ct_0_sum: u64 = 0;
            let mut ct_1_sum: u64 = 0;

            for i in 0..iteration_count1 {
                for k in 0..4usize {
                    let piece_idx = 4 * i + k;
                    let in_piece = input
                        [index + ((piece_idx * current_decomp_mod_count_plus_one) << n_power)];
                    let rk0 = relinkey
                        [idx + (key_index << n_power) + (key_offset2 * piece_idx)];
                    let rk1 = relinkey[idx
                        + (key_index << n_power)
                        + (key_offset2 * piece_idx)
                        + key_offset1];

                    let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                    let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                    ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                    ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
                }
            }

            let loop_offset = iteration_count1 * 4;
            for i in loop_offset..loop_offset + iteration_count2 {
                let in_piece = input
                    [index + ((i * current_decomp_mod_count_plus_one) << n_power)];
                let rk0 =
                    relinkey[idx + (key_index << n_power) + (key_offset2 * i)];
                let rk1 = relinkey
                    [idx + (key_index << n_power) + (key_offset2 * i) + key_offset1];

                let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
            }

            output[index] = ct_0_sum;
            output[index + (current_decomp_mod_count_plus_one << n_power)] = ct_1_sum;
        }
    }
}

// ===========================================================================
// 5. keyswitch_multiply_accumulate_leveled_method_II_kernel — CPU reference
// ===========================================================================

/// Method-II leveled key-switch multiply-accumulate with level offset.
pub fn keyswitch_multiply_accumulate_leveled_method_ii_cpu(
    input: &[u64],
    relinkey: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    first_rns_mod_count: usize,
    current_decomp_mod_count: usize,
    current_rns_mod_count: usize,
    iteration_count1: usize,
    iteration_count2: usize,
    level: usize,
    n_power: u32,
    rns_mod_count_y: usize,
    ring_size: usize,
) {
    let key_offset1 = first_rns_mod_count << n_power;
    let key_offset2 = first_rns_mod_count << (n_power + 1);

    for block_y in 0..rns_mod_count_y {
        let key_index = if block_y < current_decomp_mod_count {
            block_y
        } else {
            block_y + level
        };
        let modulus_reg = &modulus[key_index];

        for idx in 0..ring_size {
            let index = idx + (block_y << n_power);
            let index2 = idx + (key_index << n_power);
            let mut ct_0_sum: u64 = 0;
            let mut ct_1_sum: u64 = 0;

            for i in 0..iteration_count1 {
                for k in 0..4usize {
                    let piece_idx = 4 * i + k;
                    let in_piece =
                        input[index + ((piece_idx * current_rns_mod_count) << n_power)];
                    let rk0 = relinkey[index2 + (key_offset2 * piece_idx)];
                    let rk1 =
                        relinkey[index2 + (key_offset2 * piece_idx) + key_offset1];

                    let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                    let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                    ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                    ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
                }
            }

            let loop_offset = iteration_count1 * 4;
            for i in loop_offset..loop_offset + iteration_count2 {
                let in_piece =
                    input[index + ((i * current_rns_mod_count) << n_power)];
                let rk0 = relinkey[index2 + (key_offset2 * i)];
                let rk1 = relinkey[index2 + (key_offset2 * i) + key_offset1];

                let mult0 = mod_mul(in_piece, rk0, modulus_reg);
                let mult1 = mod_mul(in_piece, rk1, modulus_reg);

                ct_0_sum = mod_add(ct_0_sum, mult0, modulus_reg);
                ct_1_sum = mod_add(ct_1_sum, mult1, modulus_reg);
            }

            output[index] = ct_0_sum;
            output[index + (current_rns_mod_count << n_power)] = ct_1_sum;
        }
    }
}

// ===========================================================================
// 6. divide_round_lastq_kernel — CPU reference
// ===========================================================================

/// Divide-and-round by last q modulus for standard relinearization.
pub fn divide_round_lastq_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let mut last_ct = input[idx
                    + (decomp_mod_count << n_power)
                    + (((decomp_mod_count + 1) << n_power) * block_z)];

                last_ct = mod_add(last_ct, half[0], &modulus[decomp_mod_count]);
                last_ct = mod_reduce_forced(last_ct, &modulus[block_y]);
                last_ct = mod_sub(last_ct, half_mod[block_y], &modulus[block_y]);

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + (((decomp_mod_count + 1) << n_power) * block_z)];

                input_ = mod_sub(input_, last_ct, &modulus[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &modulus[block_y]);

                let ct_in = ct[idx
                    + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z)];

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 7. divide_round_lastq_switchkey_kernel — CPU reference
// ===========================================================================

/// Divide-and-round by last q for switch-key (ct_in = 0 for block_z != 0).
pub fn divide_round_lastq_switchkey_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let mut last_ct = input[idx
                    + (decomp_mod_count << n_power)
                    + (((decomp_mod_count + 1) << n_power) * block_z)];

                last_ct = mod_add(last_ct, half[0], &modulus[decomp_mod_count]);
                last_ct = mod_reduce_forced(last_ct, &modulus[block_y]);
                last_ct = mod_sub(last_ct, half_mod[block_y], &modulus[block_y]);

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + (((decomp_mod_count + 1) << n_power) * block_z)];

                input_ = mod_sub(input_, last_ct, &modulus[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &modulus[block_y]);

                let ct_in = if block_z == 0 {
                    ct[idx
                        + (block_y << n_power)
                        + ((decomp_mod_count << n_power) * block_z)]
                } else {
                    0u64
                };

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 8. divide_round_lastq_extended_kernel — CPU reference (nested loops, array)
// ===========================================================================

/// Extended divide-and-round with multi-prime P base (up to 15 P primes).
pub fn divide_round_lastq_extended_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_ct_arr = [0u64; 15];
                for i in 0..p_size {
                    last_ct_arr[i] = input[idx
                        + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z)];
                }

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + ((q_prime_size << n_power) * block_z)];

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let mut last_ct_add_half_ = last_ct_arr[p_size - 1 - i];
                    last_ct_add_half_ = mod_add(
                        last_ct_add_half_,
                        half[i],
                        &modulus[q_prime_size - 1 - i],
                    );
                    for j in 0..(p_size - 1 - i) {
                        let mut temp1 = mod_reduce_forced(
                            last_ct_add_half_,
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(last_ct_arr[j], temp1, &modulus[q_size + j]);
                        last_ct_arr[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_ct_add_half_, &modulus[block_y]);
                    temp1 = mod_sub(
                        temp1,
                        half_mod[location_ + block_y],
                        &modulus[block_y],
                    );
                    temp1 = mod_sub(input_, temp1, &modulus[block_y]);
                    input_ = mod_mul(
                        temp1,
                        last_q_modinv[location_ + block_y],
                        &modulus[block_y],
                    );
                    location_ += q_prime_size - 1 - i;
                }

                let ct_in = ct[idx
                    + (block_y << n_power)
                    + ((q_size << n_power) * block_z)];

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((q_size << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 9. divide_round_lastq_extended_switchkey_kernel — CPU reference
// ===========================================================================

/// Extended divide-and-round for switch-key variant (ct_in = 0 when block_z != 0).
pub fn divide_round_lastq_extended_switchkey_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_ct_arr = [0u64; 15];
                for i in 0..p_size {
                    last_ct_arr[i] = input[idx
                        + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z)];
                }

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + ((q_prime_size << n_power) * block_z)];

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let mut last_ct_add_half_ = last_ct_arr[p_size - 1 - i];
                    last_ct_add_half_ = mod_add(
                        last_ct_add_half_,
                        half[i],
                        &modulus[q_prime_size - 1 - i],
                    );
                    for j in 0..(p_size - 1 - i) {
                        let mut temp1 = mod_reduce_forced(
                            last_ct_add_half_,
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(last_ct_arr[j], temp1, &modulus[q_size + j]);
                        last_ct_arr[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_ct_add_half_, &modulus[block_y]);
                    temp1 = mod_sub(
                        temp1,
                        half_mod[location_ + block_y],
                        &modulus[block_y],
                    );
                    temp1 = mod_sub(input_, temp1, &modulus[block_y]);
                    input_ = mod_mul(
                        temp1,
                        last_q_modinv[location_ + block_y],
                        &modulus[block_y],
                    );
                    location_ += q_prime_size - 1 - i;
                }

                let ct_in = if block_z == 0 {
                    ct[idx
                        + (block_y << n_power)
                        + ((q_size << n_power) * block_z)]
                } else {
                    0u64
                };

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((q_size << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 10. DivideRoundLastqNewP_leveled — CPU reference
// ===========================================================================

/// Leveled divide-and-round with new P base (same structure as extended, for
/// leveled HE context).
pub fn divide_round_lastq_new_p_leveled_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    // Identical logic to divide_round_lastq_extended_cpu
    divide_round_lastq_extended_cpu(
        input,
        ct,
        output,
        modulus,
        half,
        half_mod,
        last_q_modinv,
        n_power,
        q_prime_size,
        q_size,
        p_size,
        ring_size,
        cipher_size,
    );
}

// ===========================================================================
// 11. divide_round_lastq_leveled_stage_one_kernel — CPU reference (multi-output)
// ===========================================================================

/// Stage one of leveled divide-round: adds half and reduces last ciphertext
/// coefficient across all decomposition moduli.
pub fn divide_round_lastq_leveled_stage_one_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    n_power: u32,
    first_decomp_count: usize,
    current_decomp_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_y in 0..cipher_size {
        for idx in 0..ring_size {
            let mut last_ct = input[idx
                + (current_decomp_count << n_power)
                + (((current_decomp_count + 1) << n_power) * block_y)];

            last_ct = mod_add(last_ct, half[0], &modulus[first_decomp_count]);

            for i in 0..current_decomp_count {
                let mut last_ct_i = mod_reduce_forced(last_ct, &modulus[i]);
                last_ct_i = mod_sub(last_ct_i, half_mod[i], &modulus[i]);

                output[idx
                    + (i << n_power)
                    + ((current_decomp_count << n_power) * block_y)] = last_ct_i;
            }
        }
    }
}

// ===========================================================================
// 12. divide_round_lastq_leveled_stage_two_kernel — CPU reference
// ===========================================================================

/// Stage two of leveled divide-round: subtracts reduced last-q term, multiplies
/// by inverse, and adds to existing ciphertext.
pub fn divide_round_lastq_leveled_stage_two_cpu(
    input_last: &[u64],
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    last_q_modinv: &[u64],
    n_power: u32,
    current_decomp_count: usize,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let last_ct = input_last[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)];

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + (((current_decomp_count + 1) << n_power) * block_z)];

                input_ = mod_sub(input_, last_ct, &modulus[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &modulus[block_y]);

                let ct_in = ct[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)];

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 13. divide_round_lastq_leveled_stage_two_switchkey_kernel — CPU reference
// ===========================================================================

/// Stage two of leveled divide-round for switch-key (ct_in = 0 when block_z != 0).
pub fn divide_round_lastq_leveled_stage_two_switchkey_cpu(
    input_last: &[u64],
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    last_q_modinv: &[u64],
    n_power: u32,
    current_decomp_count: usize,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let last_ct = input_last[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)];

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + (((current_decomp_count + 1) << n_power) * block_z)];

                input_ = mod_sub(input_, last_ct, &modulus[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &modulus[block_y]);

                let ct_in = if block_z == 0 {
                    ct[idx
                        + (block_y << n_power)
                        + ((current_decomp_count << n_power) * block_z)]
                } else {
                    0u64
                };

                let result = mod_add(ct_in, input_, &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)] = result;
            }
        }
    }
}

// ===========================================================================
// 14. move_cipher_leveled_kernel — GPU kernel (simple 1:1 copy)
// ===========================================================================

/// GPU kernel: moves cipher coefficients for rescale operation.
///
/// Grid: `(ring_size/block_size, current_decomp_count-1, cipher_size)`.
#[gpu::cuda_kernel]
pub fn move_cipher_leveled_kernel(
    input: &[u64],
    output: &mut [u64],
    n_power: u32,
    current_decomp_count: u32,
) {
    let idx = (gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as usize;
    let block_y = gpu::block_id::<gpu::DimY>() as usize;
    let block_z = gpu::block_id::<gpu::DimZ>() as usize;

    let cdc = current_decomp_count as usize;
    let np = n_power as usize;

    let r_input = input[idx + (block_y << np) + (((cdc + 1) << np) * block_z)];

    let mut out = gpu::chunk_mut(output, gpu::MapLinear::new(1));
    out[0] = r_input;
}

/// CPU reference for move_cipher_leveled_kernel.
pub fn move_cipher_leveled_cpu(
    input: &[u64],
    output: &mut [u64],
    n_power: u32,
    current_decomp_count: usize,
    decomp_y: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_y {
            for idx in 0..ring_size {
                let loc = idx
                    + (block_y << n_power)
                    + (((current_decomp_count + 1) << n_power) * block_z);
                output[loc] = input[loc];
            }
        }
    }
}

// ===========================================================================
// 15. divide_round_lastq_rescale_kernel — CPU reference
// ===========================================================================

/// Divide-and-round for rescaling: subtracts reduced last-q, multiplies by
/// inverse. No existing ciphertext addition.
pub fn divide_round_lastq_rescale_cpu(
    input_last: &[u64],
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    last_q_modinv: &[u64],
    n_power: u32,
    current_decomp_count: usize,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let last_ct = input_last[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)];

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + (((current_decomp_count + 1) << n_power) * block_z)];

                input_ = mod_sub(input_, last_ct, &modulus[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &modulus[block_y]);

                output[idx
                    + (block_y << n_power)
                    + ((current_decomp_count << n_power) * block_z)] = input_;
            }
        }
    }
}

// ===========================================================================
// 16. base_conversion_DtoB_relin_kernel — CPU reference (f32 rounding, loops)
// ===========================================================================

/// Base conversion from decomposition basis D to auxiliary basis B for
/// relinearization. Uses f32 correction factor.
pub fn base_conversion_dtob_relin_cpu(
    ciphertext: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    b_base: &[Modulus64],
    base_change_matrix_d_to_b: &[u64],
    mi_inv_d_to_b: &[u64],
    prod_d_to_b: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    _l: usize,
    _d_tilda: usize,
    d: usize,
    r_prime: usize,
    ring_size: usize,
) {
    for block_y in 0..d {
        let ij = i_j[block_y] as usize;
        let iloc = i_location[block_y] as usize;
        let matrix_index = iloc * r_prime;

        for idx in 0..ring_size {
            let location = idx + (iloc << n_power);
            let location_out = idx + ((block_y * r_prime) << n_power);

            let mut partial = [0u64; 20];
            let mut r: f32 = 0.0;

            for i in 0..ij {
                let temp = ciphertext[location + (i << n_power)];
                partial[i] = mod_mul(temp, mi_inv_d_to_b[iloc + i], &modulus[iloc + i]);
                let div = partial[i] as f32;
                let modv = modulus[iloc + i].value as f32;
                r += div / modv;
            }

            r = r.round();
            let r_ = r as u64;

            for i in 0..r_prime {
                let mut temp = 0u64;
                for j in 0..ij {
                    let mult = mod_mul(
                        partial[j],
                        base_change_matrix_d_to_b[j + (i * ij) + matrix_index],
                        &b_base[i],
                    );
                    temp = mod_add(temp, mult, &b_base[i]);
                }
                let r_mul = mod_mul(r_, prod_d_to_b[i + (block_y * r_prime)], &b_base[i]);
                let result = mod_sub(temp, r_mul, &b_base[i]);
                output[location_out + (i << n_power)] = result;
            }
        }
    }
}

// ===========================================================================
// 17. base_conversion_DtoQtilde_relin_kernel — CPU reference
// ===========================================================================

/// Base conversion from D to Q-tilde basis for relinearization.
pub fn base_conversion_dtoqtilde_relin_cpu(
    ciphertext: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    base_change_matrix_d_to_qtilda: &[u64],
    mi_inv_d_to_qtilda: &[u64],
    prod_d_to_qtilda: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    _l: usize,
    q_tilda: usize,
    d: usize,
    ring_size: usize,
) {
    for block_y in 0..d {
        let ij = i_j[block_y] as usize;
        let iloc = i_location[block_y] as usize;
        let matrix_index = iloc * q_tilda;

        for idx in 0..ring_size {
            let location = idx + (iloc << n_power);
            let location_out = idx + ((block_y * q_tilda) << n_power);

            let mut partial = [0u64; 20];
            let mut r: f32 = 0.0;

            for i in 0..ij {
                let temp = ciphertext[location + (i << n_power)];
                partial[i] =
                    mod_mul(temp, mi_inv_d_to_qtilda[iloc + i], &modulus[iloc + i]);
                let div = partial[i] as f32;
                let modv = modulus[iloc + i].value as f32;
                r += div / modv;
            }

            r = r.round();
            let r_ = r as u64;

            for i in 0..q_tilda {
                let mut temp = 0u64;
                for j in 0..ij {
                    let mult = mod_mul(
                        partial[j],
                        base_change_matrix_d_to_qtilda[j + (i * ij) + matrix_index],
                        &modulus[i],
                    );
                    temp = mod_add(temp, mult, &modulus[i]);
                }
                let r_mul = mod_mul(
                    r_,
                    prod_d_to_qtilda[i + (block_y * q_tilda)],
                    &modulus[i],
                );
                let result = mod_sub(temp, r_mul, &modulus[i]);
                output[location_out + (i << n_power)] = result;
            }
        }
    }
}

// ===========================================================================
// 18. base_conversion_DtoB_relin_leveled_kernel — CPU reference
// ===========================================================================

/// Leveled base conversion D to B with indirect modulus indexing.
pub fn base_conversion_dtob_relin_leveled_cpu(
    ciphertext: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    b_base: &[Modulus64],
    base_change_matrix_d_to_b: &[u64],
    mi_inv_d_to_b: &[u64],
    prod_d_to_b: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    _d_tilda: usize,
    d: usize,
    r_prime: usize,
    mod_index: &[usize],
    ring_size: usize,
) {
    for block_y in 0..d {
        let ij = i_j[block_y] as usize;
        let iloc = i_location[block_y] as usize;
        let matrix_index = iloc * r_prime;

        for idx in 0..ring_size {
            let location = idx + (iloc << n_power);
            let location_out = idx + ((block_y * r_prime) << n_power);

            let mut partial = [0u64; 20];
            let mut r: f64 = 0.0;

            for i in 0..ij {
                let temp = ciphertext[location + (i << n_power)];
                partial[i] = mod_mul(
                    temp,
                    mi_inv_d_to_b[iloc + i],
                    &modulus[mod_index[iloc + i]],
                );
                let div = partial[i] as f64;
                let modv = modulus[mod_index[iloc + i]].value as f64;
                r += div / modv;
            }

            r = r.round();
            let r_ = r as u64;

            for i in 0..r_prime {
                let mut temp = 0u64;
                for j in 0..ij {
                    let mult = mod_mul(
                        partial[j],
                        base_change_matrix_d_to_b[j + (i * ij) + matrix_index],
                        &b_base[i],
                    );
                    temp = mod_add(temp, mult, &b_base[i]);
                }
                let r_mul = mod_mul(r_, prod_d_to_b[i + (block_y * r_prime)], &b_base[i]);
                let result = mod_sub(temp, r_mul, &b_base[i]);
                output[location_out + (i << n_power)] = result;
            }
        }
    }
}

// ===========================================================================
// 19. base_conversion_DtoQtilde_relin_leveled_kernel — CPU reference
// ===========================================================================

/// Leveled base conversion D to Q-tilde with level-aware modulus selection.
pub fn base_conversion_dtoqtilde_relin_leveled_cpu(
    ciphertext: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    base_change_matrix_d_to_qtilda: &[u64],
    mi_inv_d_to_qtilda: &[u64],
    prod_d_to_qtilda: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    d: usize,
    current_qtilda_size: usize,
    current_q_size: usize,
    level: usize,
    ring_size: usize,
) {
    for block_y in 0..d {
        let ij = i_j[block_y] as usize;
        let iloc = i_location[block_y] as usize;
        let matrix_index = iloc * current_qtilda_size;

        for idx in 0..ring_size {
            let location = idx + (iloc << n_power);
            let location_out = idx + ((block_y * current_qtilda_size) << n_power);

            let mut partial = [0u64; 20];
            let mut r: f64 = 0.0;

            for i in 0..ij {
                let temp = ciphertext[location + (i << n_power)];
                partial[i] =
                    mod_mul(temp, mi_inv_d_to_qtilda[iloc + i], &modulus[iloc + i]);
                let div = partial[i] as f64;
                let modv = modulus[iloc + i].value as f64;
                r += div / modv;
            }

            r = r.round();
            let r_ = r as u64;

            for i in 0..current_qtilda_size {
                let mod_location = if i < current_q_size { i } else { i + level };
                let mut temp = 0u64;
                for j in 0..ij {
                    let mut mult =
                        mod_reduce_forced(partial[j], &modulus[mod_location]);
                    mult = mod_mul(
                        mult,
                        base_change_matrix_d_to_qtilda[j + (i * ij) + matrix_index],
                        &modulus[mod_location],
                    );
                    temp = mod_add(temp, mult, &modulus[mod_location]);
                }
                let r_mul = mod_mul(
                    r_,
                    prod_d_to_qtilda[i + (block_y * current_qtilda_size)],
                    &modulus[mod_location],
                );
                let result = mod_sub(temp, r_mul, &modulus[mod_location]);
                output[location_out + (i << n_power)] = result;
            }
        }
    }
}

// ===========================================================================
// 20. multiply_accumulate_extended_kernel — CPU reference (2 outputs, loop)
// ===========================================================================

/// Extended multiply-accumulate over auxiliary B basis for method-I relinearization.
pub fn multiply_accumulate_extended_cpu(
    input: &[u64],
    relinkey: &[u64],
    output: &mut [u64],
    b_prime: &[Modulus64],
    n_power: u32,
    d_tilda: usize,
    d: usize,
    r_prime: usize,
    ring_size: usize,
) {
    let key_offset1 = (r_prime * d_tilda) << n_power;
    let key_offset2 = (r_prime * d_tilda) << (n_power + 1);

    for block_z in 0..d_tilda {
        for block_y in 0..r_prime {
            let modulus = &b_prime[block_y];
            for idx in 0..ring_size {
                let offset1 = idx + (block_y << n_power);
                let offset2 = offset1 + ((r_prime << n_power) * block_z);

                let mut ct_0_sum = 0u64;
                let mut ct_1_sum = 0u64;

                for i in 0..d {
                    let in_piece = input[offset1 + ((i * r_prime) << n_power)];
                    let rk0 = relinkey[offset2 + (key_offset2 * i)];
                    let mult0 = mod_mul(in_piece, rk0, modulus);
                    ct_0_sum = mod_add(ct_0_sum, mult0, modulus);

                    let rk1 = relinkey[offset2 + (key_offset2 * i) + key_offset1];
                    let mult1 = mod_mul(in_piece, rk1, modulus);
                    ct_1_sum = mod_add(ct_1_sum, mult1, modulus);
                }

                output[offset2] = ct_0_sum;
                output[offset2 + key_offset1] = ct_1_sum;
            }
        }
    }
}

// ===========================================================================
// 21. base_conversion_BtoD_relin_kernel — CPU reference (f32 rounding)
// ===========================================================================

/// Base conversion from auxiliary B basis back to decomposition D basis.
pub fn base_conversion_btod_relin_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    b_base: &[Modulus64],
    base_change_matrix_b_to_d: &[u64],
    mi_inv_b_to_d: &[u64],
    prod_b_to_d: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    l_tilda: usize,
    d_tilda: usize,
    _d: usize,
    r_prime: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..d_tilda {
            let ij = i_j[block_y] as usize;
            let iloc = i_location[block_y] as usize;
            let matrix_index = iloc * r_prime;

            for idx in 0..ring_size {
                let location_out =
                    idx + (iloc << n_power) + ((l_tilda << n_power) * block_z);
                let location = idx
                    + ((r_prime << n_power) * block_y)
                    + (((d_tilda * r_prime) << n_power) * block_z);

                let mut partial = [0u64; 20];
                let mut r: f64 = 0.0;

                for i in 0..r_prime {
                    let temp = input[location + (i << n_power)];
                    partial[i] = mod_mul(temp, mi_inv_b_to_d[i], &b_base[i]);
                    let div = partial[i] as f64;
                    let modv = b_base[i].value as f64;
                    r += div / modv;
                }

                r = r.round();
                let r_ = r as u64;

                for i in 0..ij {
                    let mut temp = 0u64;
                    for j in 0..r_prime {
                        let partial_ =
                            cpu_mod_reduce(partial[j], &modulus[i + iloc]);
                        let mult = mod_mul(
                            partial_,
                            base_change_matrix_b_to_d[j + (i * r_prime) + matrix_index],
                            &modulus[i + iloc],
                        );
                        temp = mod_add(temp, mult, &modulus[i + iloc]);
                    }

                    let r_mul =
                        mod_mul(r_, prod_b_to_d[i + iloc], &modulus[i + iloc]);
                    temp = mod_sub(temp, r_mul, &modulus[i + iloc]);
                    temp = cpu_mod_reduce(temp, &modulus[i + iloc]);

                    output[location_out + (i << n_power)] = temp;
                }
            }
        }
    }
}

// ===========================================================================
// 22. base_conversion_BtoD_relin_leveled_kernel — CPU reference
// ===========================================================================

/// Leveled base conversion B to D with indirect modulus indexing.
pub fn base_conversion_btod_relin_leveled_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    b_base: &[Modulus64],
    base_change_matrix_b_to_d: &[u64],
    mi_inv_b_to_d: &[u64],
    prod_b_to_d: &[u64],
    i_j: &[i32],
    i_location: &[i32],
    n_power: u32,
    l_tilda: usize,
    d_tilda: usize,
    _d: usize,
    r_prime: usize,
    mod_index: &[usize],
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..d_tilda {
            let ij = i_j[block_y] as usize;
            let iloc = i_location[block_y] as usize;
            let matrix_index = iloc * r_prime;

            for idx in 0..ring_size {
                let location_out =
                    idx + (iloc << n_power) + ((l_tilda << n_power) * block_z);
                let location = idx
                    + ((r_prime << n_power) * block_y)
                    + (((d_tilda * r_prime) << n_power) * block_z);

                let mut partial = [0u64; 20];
                let mut r: f64 = 0.0;

                for i in 0..r_prime {
                    let temp = input[location + (i << n_power)];
                    partial[i] = mod_mul(temp, mi_inv_b_to_d[i], &b_base[i]);
                    let div = partial[i] as f64;
                    let modv = b_base[i].value as f64;
                    r += div / modv;
                }

                r = r.round();
                let r_ = r as u64;

                for i in 0..ij {
                    let mi = mod_index[iloc + i];
                    let mut temp = 0u64;
                    for j in 0..r_prime {
                        let partial_ = mod_reduce_forced(partial[j], &modulus[mi]);
                        let mult = mod_mul(
                            partial_,
                            base_change_matrix_b_to_d[j + (i * r_prime) + matrix_index],
                            &modulus[mi],
                        );
                        temp = mod_add(temp, mult, &modulus[mi]);
                    }

                    let r_mul = mod_mul(r_, prod_b_to_d[i + iloc], &modulus[mi]);
                    temp = mod_sub(temp, r_mul, &modulus[mi]);
                    temp = mod_reduce_forced(temp, &modulus[mi]);

                    output[location_out + (i << n_power)] = temp;
                }
            }
        }
    }
}

// ===========================================================================
// 23. divide_round_lastq_extended_leveled_kernel — CPU reference
// ===========================================================================

/// Extended leveled divide-and-round with first_Q_prime_size/first_Q_size params.
pub fn divide_round_lastq_extended_leveled_cpu(
    input: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    first_q_prime_size: usize,
    first_q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_ct_arr = [0u64; 15];
                for i in 0..p_size {
                    last_ct_arr[i] = input[idx
                        + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z)];
                }

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + ((q_prime_size << n_power) * block_z)];

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let mut last_ct_add_half_ = last_ct_arr[p_size - 1 - i];
                    last_ct_add_half_ = mod_add(
                        last_ct_add_half_,
                        half[i],
                        &modulus[first_q_prime_size - 1 - i],
                    );
                    for j in 0..(p_size - 1 - i) {
                        let mut temp1 = mod_reduce_forced(
                            last_ct_add_half_,
                            &modulus[first_q_size + j],
                        );
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + first_q_size + j],
                            &modulus[first_q_size + j],
                        );
                        temp1 = mod_sub(
                            last_ct_arr[j],
                            temp1,
                            &modulus[first_q_size + j],
                        );
                        last_ct_arr[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + first_q_size + j],
                            &modulus[first_q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_ct_add_half_, &modulus[block_y]);
                    temp1 = mod_sub(
                        temp1,
                        half_mod[location_ + block_y],
                        &modulus[block_y],
                    );
                    temp1 = mod_sub(input_, temp1, &modulus[block_y]);
                    input_ = mod_mul(
                        temp1,
                        last_q_modinv[location_ + block_y],
                        &modulus[block_y],
                    );
                    location_ += first_q_prime_size - 1 - i;
                }

                output[idx
                    + (block_y << n_power)
                    + ((q_size << n_power) * block_z)] = input_;
            }
        }
    }
}

// ===========================================================================
// 24. global_memory_replace_kernel — GPU kernel (simple 1:1 copy)
// ===========================================================================

/// GPU kernel: copies data from input to output using 3D grid indexing.
///
/// Grid: `(ring_size/block_size, q_size, cipher_size)`.
#[gpu::cuda_kernel]
pub fn global_memory_replace_kernel(
    input: &[u64],
    output: &mut [u64],
    n_power: u32,
    grid_dim_y: u32,
) {
    let idx = (gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as usize;
    let block_y = gpu::block_id::<gpu::DimY>() as usize;
    let block_z = gpu::block_id::<gpu::DimZ>() as usize;
    let gdy = grid_dim_y as usize;
    let np = n_power as usize;

    let location = idx + (block_y << np) + ((gdy << np) * block_z);
    let in_reg = input[location];

    let mut out = gpu::chunk_mut(output, gpu::MapLinear::new(1));
    out[0] = in_reg;
}

/// CPU reference for global_memory_replace_kernel.
pub fn global_memory_replace_cpu(
    input: &[u64],
    output: &mut [u64],
    n_power: u32,
    q_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let loc = idx + (block_y << n_power) + ((q_size << n_power) * block_z);
                output[loc] = input[loc];
            }
        }
    }
}

// ===========================================================================
// 25. global_memory_replace_offset_kernel — GPU kernel (1:1 with offset)
// ===========================================================================

/// GPU kernel: copies data with offset between decomposition counts.
///
/// Grid: `(ring_size/block_size, q_size, cipher_size)`.
#[gpu::cuda_kernel]
pub fn global_memory_replace_offset_kernel(
    input: &[u64],
    output: &mut [u64],
    current_decomposition_count: u32,
    n_power: u32,
) {
    let idx = (gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as usize;
    let block_y = gpu::block_id::<gpu::DimY>() as usize;
    let block_z = gpu::block_id::<gpu::DimZ>() as usize;
    let cdc = current_decomposition_count as usize;
    let np = n_power as usize;

    let location_in = idx + (block_y << np) + ((cdc << np) * block_z);
    let in_reg = input[location_in];

    let mut out = gpu::chunk_mut(output, gpu::MapLinear::new(1));
    out[0] = in_reg;
}

/// CPU reference for global_memory_replace_offset_kernel.
pub fn global_memory_replace_offset_cpu(
    input: &[u64],
    output: &mut [u64],
    current_decomposition_count: usize,
    n_power: u32,
    q_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let location_in = idx
                    + (block_y << n_power)
                    + ((current_decomposition_count << n_power) * block_z);
                let location_out = idx
                    + (block_y << n_power)
                    + (((current_decomposition_count - 1) << n_power) * block_z);
                output[location_out] = input[location_in];
            }
        }
    }
}

// ===========================================================================
// 26. cipher_broadcast_switchkey_kernel — CPU reference (branching + loop)
// ===========================================================================

/// Switch-key cipher broadcast: component 0 goes to out0, component 1 is
/// broadcast across RNS limbs into out1.
pub fn cipher_broadcast_switchkey_cpu(
    cipher: &[u64],
    out0: &mut [u64],
    out1: &mut [u64],
    _modulus: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let rns_mod_count = decomp_mod_count + 1;
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let result_value = cipher[idx
                    + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z)];

                if block_z == 0 {
                    out0[idx
                        + (block_y << n_power)
                        + ((decomp_mod_count << n_power) * block_z)] = result_value;
                } else {
                    let location = (rns_mod_count * block_y) << n_power;
                    for i in 0..rns_mod_count {
                        out1[idx + (i << n_power) + location] = result_value;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 27. cipher_broadcast_switchkey_method_II_kernel — GPU kernel (simple 1:1)
// ===========================================================================

/// GPU kernel: method-II switch-key broadcast — simple per-element copy to
/// either out0 or out1 based on block_z.
///
/// Grid: `(ring_size/block_size, decomp_mod_count, 2)`.
/// Note: this kernel writes to two different output buffers depending on block_z.
/// For GPU we port the block_z==0 path (out0). The block_z==1 path is identical
/// in structure but writes to a different buffer.
///
/// CPU reference provided below handles both paths.
pub fn cipher_broadcast_switchkey_method_ii_cpu(
    cipher: &[u64],
    out0: &mut [u64],
    out1: &mut [u64],
    _modulus: &[Modulus64],
    n_power: u32,
    decomp_mod_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                let result_value = cipher[idx
                    + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z)];

                if block_z == 0 {
                    out0[idx + (block_y << n_power)] = result_value;
                } else {
                    out1[idx + (block_y << n_power)] = result_value;
                }
            }
        }
    }
}

// ===========================================================================
// 28. cipher_broadcast_switchkey_leveled_kernel — CPU reference (loop + reduce)
// ===========================================================================

/// Leveled switch-key cipher broadcast with level-aware modulus reduction.
pub fn cipher_broadcast_switchkey_leveled_cpu(
    cipher: &[u64],
    out0: &mut [u64],
    out1: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    first_rns_mod_count: usize,
    current_rns_mod_count: usize,
    current_decomp_mod_count: usize,
    decomp_y: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let level = first_rns_mod_count - current_rns_mod_count;
    for block_z in 0..cipher_size {
        for block_y in 0..decomp_y {
            for idx in 0..ring_size {
                let result_value = cipher[idx
                    + (block_y << n_power)
                    + ((current_decomp_mod_count << n_power) * block_z)];

                if block_z == 0 {
                    out0[idx
                        + (block_y << n_power)
                        + ((current_decomp_mod_count << n_power) * block_z)] =
                        result_value;
                } else {
                    let location = (current_rns_mod_count * block_y) << n_power;
                    for i in 0..current_rns_mod_count {
                        let mod_index = if i < current_decomp_mod_count {
                            i
                        } else {
                            i + level
                        };
                        let reduced_result =
                            mod_reduce_forced(result_value, &modulus[mod_index]);
                        out1[idx + (i << n_power) + location] = reduced_result;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 29. addition_switchkey — GPU kernel (simple 1-output-per-thread)
// ===========================================================================

/// GPU kernel: element-wise modular addition for switch-key.
///
/// For cipher component 0 (block_z == 0): adds in1 and in2.
/// For cipher component 1: copies in1 unchanged.
///
/// Grid: `(ring_size/block_size, rns_count, cipher_count)`.
#[gpu::cuda_kernel]
pub fn sk_addition_switchkey(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    grid_dim_y: u32,
) {
    let idx = (gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as usize;
    let idy = gpu::block_id::<gpu::DimY>() as usize;
    let idz = gpu::block_id::<gpu::DimZ>() as usize;
    let gdy = grid_dim_y as usize;
    let np = n_power as usize;

    let location = idx + (idy << np) + ((gdy * idz) << np);
    let mod_val = mod_values[idy];

    let result = if idz == 0 {
        let sum = in1[location] + in2[location];
        if sum >= mod_val { sum - mod_val } else { sum }
    } else {
        in1[location]
    };

    let mut out = gpu::chunk_mut(output, gpu::MapLinear::new(1));
    out[0] = result;
}

/// CPU reference for addition_switchkey.
pub fn addition_switchkey_cpu(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    ring_size: usize,
    cipher_count: usize,
) {
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let location =
                    idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    output[location] =
                        mod_add(in1[location], in2[location], &modulus[idy]);
                } else {
                    output[location] = in1[location];
                }
            }
        }
    }
}

// ===========================================================================
// 30. negacyclic_shift_poly_coeffmod_kernel — CPU reference (index permutation)
// ===========================================================================

/// Negacyclic shift of polynomial coefficients modulo the coefficient modulus.
///
/// Shifts polynomial by `shift` positions with negacyclic wraparound:
/// coefficients that wrap past the ring boundary are negated.
pub fn negacyclic_shift_poly_coeffmod_cpu(
    cipher_in: &[u64],
    cipher_out: &mut [u64],
    modulus: &[Modulus64],
    shift: i32,
    n_power: u32,
    grid_dim_y: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let coeff_count_minus_one = (1usize << n_power) - 1;

    for block_z in 0..cipher_size {
        for block_y in 0..grid_dim_y {
            for idx in 0..ring_size {
                let index_raw = idx as i32 + shift;
                let index = (index_raw as usize) & coeff_count_minus_one;
                let mut result_value = cipher_in[idx
                    + (block_y << n_power)
                    + ((grid_dim_y << n_power) * block_z)];

                if ((index_raw as usize) >> n_power) & 1 != 0 {
                    result_value = modulus[block_y].value - result_value;
                }

                cipher_out[index
                    + (block_y << n_power)
                    + ((grid_dim_y << n_power) * block_z)] = result_value;
            }
        }
    }
}

// ===========================================================================
// 31. galois_permute_ntt_pql_kernel — CPU reference (bit-reversal permutation)
// ===========================================================================

/// NTT-domain Galois automorphism X -> X^galois_elt via bit-reversed index
/// permutation.
///
/// Per-thread formula:
/// 1. br_j    = bit-reverse of destination index j
/// 2. exp_j   = 2*br_j + 1
/// 3. new_exp = galois_elt * exp_j mod 2N
/// 4. s       = (new_exp - 1) / 2
/// 5. src_idx = bit-reverse of s
pub fn galois_permute_ntt_pql_cpu(
    input: &[u64],
    output: &mut [u64],
    galois_elt: u32,
    n_power: u32,
    pql_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let two_n = 2u64 << n_power;

    for block_z in 0..cipher_size {
        for block_y in 0..pql_count {
            for idx in 0..ring_size {
                let br_j = bit_reverse(idx as u32, n_power) as u64;
                let exp_j = 2 * br_j + 1;
                let new_exp =
                    ((galois_elt as u64).wrapping_mul(exp_j)) % two_n;
                let s = ((new_exp - 1) >> 1) as u32;
                let src_idx = bit_reverse(s, n_power) as usize;

                output[idx + (block_y << n_power) + ((pql_count << n_power) * block_z)] =
                    input[src_idx + (block_y << n_power) + ((pql_count << n_power) * block_z)];
            }
        }
    }
}

/// n_power-bit reversal of a 32-bit value.
fn bit_reverse(val: u32, n_power: u32) -> u32 {
    val.reverse_bits() >> (32 - n_power)
}

// ===========================================================================
// 32. broadcast_scale_P_kernel — CPU reference (branching per limb)
// ===========================================================================

/// Creates P*c in PQ_l NTT domain: Q limbs get `P_mod_q[j] * c[j]`, P limbs
/// get 0.
pub fn broadcast_scale_p_cpu(
    c_ntt: &[u64],
    output: &mut [u64],
    p_mod_q: &[u64],
    pq_modulus: &[Modulus64],
    n_power: u32,
    current_decomp_count: usize,
    pql_count: usize,
    ring_size: usize,
) {
    for block_y in 0..pql_count {
        for idx in 0..ring_size {
            let result = if block_y < current_decomp_count {
                let c_val = c_ntt[idx + (block_y << n_power)];
                mod_mul(c_val, p_mod_q[block_y], &pq_modulus[block_y])
            } else {
                0u64
            };
            output[idx + (block_y << n_power)] = result;
        }
    }
}

// ===========================================================================
// 33. addition_pql_kernel — GPU kernel (simple 1-output-per-thread)
// ===========================================================================

/// GPU kernel: element-wise modular addition for PQ_l polynomials.
///
/// Grid: `(ring_size/block_size, pql_count, 2)`.
#[gpu::cuda_kernel]
pub fn sk_addition_pql_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    pql_count: u32,
) {
    let idx = (gpu::block_id::<gpu::DimX>() * gpu::block_dim::<gpu::DimX>()
        + gpu::thread_id::<gpu::DimX>()) as usize;
    let block_y = gpu::block_id::<gpu::DimY>() as usize;
    let block_z = gpu::block_id::<gpu::DimZ>() as usize;
    let pqlc = pql_count as usize;
    let np = n_power as usize;

    let offset = idx + (block_y << np) + ((pqlc << np) * block_z);
    let mod_val = mod_values[block_y];

    let val1 = in1[offset];
    let val2 = in2[offset];
    let sum = val1 + val2;
    let result = if sum >= mod_val { sum - mod_val } else { sum };

    let mut out = gpu::chunk_mut(output, gpu::MapLinear::new(1));
    out[0] = result;
}

/// CPU reference for addition_pql_kernel.
pub fn addition_pql_cpu(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    pq_modulus: &[Modulus64],
    n_power: u32,
    pql_count: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..pql_count {
            for idx in 0..ring_size {
                let offset =
                    idx + (block_y << n_power) + ((pql_count << n_power) * block_z);
                output[offset] =
                    mod_add(in1[offset], in2[offset], &pq_modulus[block_y]);
            }
        }
    }
}

// ===========================================================================
// 34. ckks_duplicate_kernel — CPU reference (loop with conditional mod index)
// ===========================================================================

/// CKKS duplicate: broadcasts cipher[1] component across RNS limbs with
/// level-aware modulus reduction for hoisting optimizations.
pub fn ckks_duplicate_cpu(
    cipher: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    first_rns_mod_count: usize,
    current_rns_mod_count: usize,
    current_decomp_mod_count: usize,
    decomp_y: usize,
    ring_size: usize,
) {
    let level = first_rns_mod_count - current_rns_mod_count;
    for block_y in 0..decomp_y {
        let location = (current_rns_mod_count * block_y) << n_power;
        for idx in 0..ring_size {
            let result_value = cipher[idx
                + (block_y << n_power)
                + (current_decomp_mod_count << n_power)];

            for i in 0..current_rns_mod_count {
                let mod_index = if i < current_decomp_mod_count {
                    i
                } else {
                    i + level
                };
                let reduced_result =
                    mod_reduce_forced(result_value, &modulus[mod_index]);
                output[idx + (i << n_power) + location] = reduced_result;
            }
        }
    }
}

// ===========================================================================
// 35. bfv_duplicate_kernel — CPU reference (branching + loop)
// ===========================================================================

/// BFV duplicate: component 0 goes to output1, component 1 is broadcast with
/// reduction across RNS limbs into output2.
pub fn bfv_duplicate_cpu(
    cipher: &[u64],
    output1: &mut [u64],
    output2: &mut [u64],
    modulus: &[Modulus64],
    n_power: u32,
    rns_mod_count: usize,
    grid_dim_y: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    for block_z in 0..cipher_size {
        for block_y in 0..grid_dim_y {
            for idx in 0..ring_size {
                let result_value = cipher[idx
                    + (block_y << n_power)
                    + ((grid_dim_y << n_power) * block_z)];

                if block_z == 0 {
                    output1[idx + (block_y << n_power)] = result_value;
                } else {
                    let location = (rns_mod_count * block_y) << n_power;
                    for i in 0..rns_mod_count {
                        let reduced_result =
                            mod_reduce_forced(result_value, &modulus[i]);
                        output2[idx + (i << n_power) + location] = reduced_result;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 36. divide_round_lastq_permute_ckks_kernel — CPU reference
// ===========================================================================

/// CKKS permuted divide-and-round: combines extended divide-round with
/// negacyclic coefficient permutation for Galois automorphism.
pub fn divide_round_lastq_permute_ckks_cpu(
    input: &[u64],
    input2: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    galois_elt: i32,
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    first_q_prime_size: usize,
    first_q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let coeff_count_minus_one = (1usize << n_power) - 1;

    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_ct_arr = [0u64; 15];
                for i in 0..p_size {
                    last_ct_arr[i] = input[idx
                        + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z)];
                }

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + ((q_prime_size << n_power) * block_z)];

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let mut last_ct_add_half_ = last_ct_arr[p_size - 1 - i];
                    last_ct_add_half_ = mod_add(
                        last_ct_add_half_,
                        half[i],
                        &modulus[first_q_prime_size - 1 - i],
                    );
                    for j in 0..(p_size - 1 - i) {
                        let mut temp1 = mod_reduce_forced(
                            last_ct_add_half_,
                            &modulus[first_q_size + j],
                        );
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + first_q_size + j],
                            &modulus[first_q_size + j],
                        );
                        temp1 = mod_sub(
                            last_ct_arr[j],
                            temp1,
                            &modulus[first_q_size + j],
                        );
                        last_ct_arr[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + first_q_size + j],
                            &modulus[first_q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_ct_add_half_, &modulus[block_y]);
                    temp1 = mod_sub(
                        temp1,
                        half_mod[location_ + block_y],
                        &modulus[block_y],
                    );
                    temp1 = mod_sub(input_, temp1, &modulus[block_y]);
                    input_ = mod_mul(
                        temp1,
                        last_q_modinv[location_ + block_y],
                        &modulus[block_y],
                    );
                    location_ += first_q_prime_size - 1 - i;
                }

                if block_z == 0 {
                    let mut ct_in = input2[idx + (block_y << n_power)];
                    ct_in = mod_add(ct_in, input_, &modulus[block_y]);

                    let index_raw = (idx as i32).wrapping_mul(galois_elt);
                    let index = (index_raw as usize) & coeff_count_minus_one;
                    if ((index_raw as usize) >> n_power) & 1 != 0 {
                        ct_in = modulus[block_y].value - ct_in;
                    }

                    output[index
                        + (block_y << n_power)
                        + ((q_size << n_power) * block_z)] = ct_in;
                } else {
                    let index_raw = (idx as i32).wrapping_mul(galois_elt);
                    let index = (index_raw as usize) & coeff_count_minus_one;
                    if ((index_raw as usize) >> n_power) & 1 != 0 {
                        input_ = modulus[block_y].value - input_;
                    }

                    output[index
                        + (block_y << n_power)
                        + ((q_size << n_power) * block_z)] = input_;
                }
            }
        }
    }
}

// ===========================================================================
// 37. divide_round_lastq_permute_bfv_kernel — CPU reference
// ===========================================================================

/// BFV permuted divide-and-round: combines extended divide-round with
/// negacyclic coefficient permutation for Galois automorphism.
pub fn divide_round_lastq_permute_bfv_cpu(
    input: &[u64],
    ct: &[u64],
    output: &mut [u64],
    modulus: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    galois_elt: i32,
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
    ring_size: usize,
    cipher_size: usize,
) {
    let coeff_count_minus_one = (1usize << n_power) - 1;

    for block_z in 0..cipher_size {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_ct_arr = [0u64; 15];
                for i in 0..p_size {
                    last_ct_arr[i] = input[idx
                        + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z)];
                }

                let mut input_ = input[idx
                    + (block_y << n_power)
                    + ((q_prime_size << n_power) * block_z)];

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let mut last_ct_add_half_ = last_ct_arr[p_size - 1 - i];
                    last_ct_add_half_ = mod_add(
                        last_ct_add_half_,
                        half[i],
                        &modulus[q_prime_size - 1 - i],
                    );
                    for j in 0..(p_size - 1 - i) {
                        let mut temp1 = mod_reduce_forced(
                            last_ct_add_half_,
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                        temp1 = mod_sub(
                            last_ct_arr[j],
                            temp1,
                            &modulus[q_size + j],
                        );
                        last_ct_arr[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + q_size + j],
                            &modulus[q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_ct_add_half_, &modulus[block_y]);
                    temp1 = mod_sub(
                        temp1,
                        half_mod[location_ + block_y],
                        &modulus[block_y],
                    );
                    temp1 = mod_sub(input_, temp1, &modulus[block_y]);
                    input_ = mod_mul(
                        temp1,
                        last_q_modinv[location_ + block_y],
                        &modulus[block_y],
                    );
                    location_ += q_prime_size - 1 - i;
                }

                if block_z == 0 {
                    let mut ct_in = ct[idx + (block_y << n_power)];
                    ct_in = mod_add(ct_in, input_, &modulus[block_y]);

                    let index_raw = (idx as i32).wrapping_mul(galois_elt);
                    let index = (index_raw as usize) & coeff_count_minus_one;
                    if ((index_raw as usize) >> n_power) & 1 != 0 {
                        ct_in = modulus[block_y].value - ct_in;
                    }

                    output[index
                        + (block_y << n_power)
                        + ((q_size << n_power) * block_z)] = ct_in;
                } else {
                    let index_raw = (idx as i32).wrapping_mul(galois_elt);
                    let index = (index_raw as usize) & coeff_count_minus_one;
                    if ((index_raw as usize) >> n_power) & 1 != 0 {
                        input_ = modulus[block_y].value - input_;
                    }

                    output[index
                        + (block_y << n_power)
                        + ((q_size << n_power) * block_z)] = input_;
                }
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mod(value: u64) -> Modulus64 {
        Modulus64::new(value)
    }

    #[test]
    fn test_global_memory_replace_cpu() {
        let n_power = 2u32;
        let ring_size = 1 << n_power;
        let q_size = 2;
        let cipher_size = 1;
        let total = ring_size * q_size * cipher_size;

        let input: Vec<u64> = (0..total as u64).collect();
        let mut output = vec![0u64; total];

        global_memory_replace_cpu(&input, &mut output, n_power, q_size, ring_size, cipher_size);
        assert_eq!(input, output);
    }

    #[test]
    fn test_addition_switchkey_cpu() {
        let n_power = 2u32;
        let ring_size = 1 << n_power;
        let rns_count = 2;
        let cipher_count = 2;
        let total = ring_size * rns_count * cipher_count;

        let m = make_mod(17);
        let moduli = vec![m; rns_count];

        let in1: Vec<u64> = (0..total).map(|i| (i as u64) % 17).collect();
        let in2: Vec<u64> = (0..total).map(|i| ((i + 3) as u64) % 17).collect();
        let mut output = vec![0u64; total];

        addition_switchkey_cpu(
            &in1, &in2, &mut output, &moduli, n_power, rns_count, ring_size, cipher_count,
        );

        // For idz == 0, output = (in1 + in2) mod 17
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power);
                assert_eq!(output[loc], mod_add(in1[loc], in2[loc], &m));
            }
        }
        // For idz == 1, output = in1
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + (rns_count << n_power);
                assert_eq!(output[loc], in1[loc]);
            }
        }
    }

    #[test]
    fn test_addition_pql_cpu() {
        let n_power = 2u32;
        let ring_size = 1 << n_power;
        let pql_count = 3;
        let cipher_size = 2;
        let total = ring_size * pql_count * cipher_size;

        let moduli: Vec<Modulus64> = vec![make_mod(17), make_mod(19), make_mod(23)];

        let in1: Vec<u64> = (0..total).map(|i| (i as u64) % 13).collect();
        let in2: Vec<u64> = (0..total).map(|i| ((i + 5) as u64) % 13).collect();
        let mut output = vec![0u64; total];

        addition_pql_cpu(
            &in1, &in2, &mut output, &moduli, n_power, pql_count, ring_size, cipher_size,
        );

        for block_z in 0..cipher_size {
            for block_y in 0..pql_count {
                for idx in 0..ring_size {
                    let offset =
                        idx + (block_y << n_power) + ((pql_count << n_power) * block_z);
                    assert_eq!(
                        output[offset],
                        mod_add(in1[offset], in2[offset], &moduli[block_y])
                    );
                }
            }
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 4), 0);
        assert_eq!(bit_reverse(1, 4), 8);
        assert_eq!(bit_reverse(8, 4), 1);
        assert_eq!(bit_reverse(0b1010, 4), 0b0101);
    }

    #[test]
    fn test_negacyclic_shift_basic() {
        let n_power = 2u32;
        let ring_size = 1 << n_power; // 4
        let grid_dim_y = 1;
        let cipher_size = 1;
        let m = make_mod(17);
        let moduli = vec![m];

        // Input: [1, 2, 3, 4]
        let input = vec![1u64, 2, 3, 4];
        let mut output = vec![0u64; ring_size];

        // Shift by 0 should be identity
        negacyclic_shift_poly_coeffmod_cpu(
            &input, &mut output, &moduli, 0, n_power, grid_dim_y, ring_size, cipher_size,
        );
        assert_eq!(output, vec![1, 2, 3, 4]);

        // Shift by 1: element at position i goes to position (i+1) mod 4
        // with negation if wrapping
        let mut output2 = vec![0u64; ring_size];
        negacyclic_shift_poly_coeffmod_cpu(
            &input, &mut output2, &moduli, 1, n_power, grid_dim_y, ring_size, cipher_size,
        );
        // idx=0 -> index_raw=1, index=1, no wrap -> output[1]=1
        // idx=1 -> index_raw=2, index=2, no wrap -> output[2]=2
        // idx=2 -> index_raw=3, index=3, no wrap -> output[3]=3
        // idx=3 -> index_raw=4, index=0, wrap (4>>2)&1=1 -> output[0]=17-4=13
        assert_eq!(output2, vec![13, 1, 2, 3]);
    }
}
