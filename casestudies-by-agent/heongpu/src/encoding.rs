/// BFV encoding/decoding for homomorphic encryption.
///
/// Encodes plaintext message vectors into RNS polynomial representation
/// and decodes them back.

use crate::modular::Modulus64;

/// Encode plaintext messages into BFV polynomial (RNS representation).
///
/// - `messages`: plaintext values in `[0, t)`, length `N = 1 << n_power`
/// - `output`: flat RNS array of length `N * moduli.len()`
/// - `moduli`: RNS moduli (length `rns_count`)
/// - `plain_modulus`: plaintext modulus `t`
/// - `n_power`: `log2(ring_size)`
pub fn encode_bfv_cpu(
    messages: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    plain_modulus: u64,
    n_power: u32,
) {
    let ring_size = 1usize << n_power;
    let rns_count = moduli.len();
    for idy in 0..rns_count {
        let q = moduli[idy].value;
        let delta = q / plain_modulus; // floor(q / t)
        for idx in 0..ring_size {
            let loc = idx + (idy << n_power);
            let m = messages[idx];
            // Use u128 to avoid overflow: m * delta can exceed u64
            output[loc] = ((m as u128 * delta as u128) % q as u128) as u64;
        }
    }
}

/// Decode BFV polynomial back to plaintext messages.
///
/// Decodes from the first RNS level (`idy=0`).
/// - `encoded`: flat RNS array
/// - `messages`: output plaintext values, length `N = 1 << n_power`
/// - `modulus`: first RNS modulus (`q_0`)
/// - `plain_modulus`: plaintext modulus `t`
/// - `n_power`: `log2(ring_size)`
pub fn decode_bfv_cpu(
    encoded: &[u64],
    messages: &mut [u64],
    modulus: &Modulus64,
    plain_modulus: u64,
    n_power: u32,
) {
    let ring_size = 1usize << n_power;
    let q = modulus.value;
    let t = plain_modulus;
    for idx in 0..ring_size {
        let val = encoded[idx]; // first RNS level (offset 0)
        // round(val * t / q) = (val * t + q/2) / q
        let numerator = val as u128 * t as u128 + (q as u128 / 2);
        let decoded = (numerator / q as u128) as u64;
        messages[idx] = decoded % t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let plain_modulus = 65537u64;

        let moduli = vec![
            Modulus64::new(1152921504606846883),
            Modulus64::new(1152921504606846819),
        ];
        let rns_count = moduli.len();

        let messages: Vec<u64> = (0..ring_size as u64).collect();

        let mut encoded = vec![0u64; ring_size * rns_count];
        encode_bfv_cpu(&messages, &mut encoded, &moduli, plain_modulus, n_power);

        let mut decoded = vec![0u64; ring_size];
        decode_bfv_cpu(&encoded, &mut decoded, &moduli[0], plain_modulus, n_power);

        assert_eq!(messages, decoded);
    }

    #[test]
    fn test_encode_decode_random_messages() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let plain_modulus = 65537u64;
        let moduli = vec![Modulus64::new(1152921504606846883)];

        let mut rng_state: u64 = 0xcafebabe;
        let messages: Vec<u64> = (0..ring_size)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                rng_state % plain_modulus
            })
            .collect();

        let mut encoded = vec![0u64; ring_size * moduli.len()];
        encode_bfv_cpu(&messages, &mut encoded, &moduli, plain_modulus, n_power);

        let mut decoded = vec![0u64; ring_size];
        decode_bfv_cpu(&encoded, &mut decoded, &moduli[0], plain_modulus, n_power);

        assert_eq!(messages, decoded);
    }

    #[test]
    fn test_encode_decode_edge_values() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let plain_modulus = 65537u64;
        let moduli = vec![Modulus64::new(1152921504606846883)];

        let mut messages = vec![0u64; ring_size];
        messages[0] = 0;
        messages[1] = plain_modulus - 1;
        messages[2] = plain_modulus / 2;

        let mut encoded = vec![0u64; ring_size * moduli.len()];
        encode_bfv_cpu(&messages, &mut encoded, &moduli, plain_modulus, n_power);

        let mut decoded = vec![0u64; ring_size];
        decode_bfv_cpu(&encoded, &mut decoded, &moduli[0], plain_modulus, n_power);

        assert_eq!(messages, decoded);
    }
}

// ---------------------------------------------------------------------------
// GPU kernels and CPU references ported from HEonGPU encoding.cu
// ---------------------------------------------------------------------------

use crate::modular::{mod_mul, mod_reduce_forced, mod_sub};

/// Reduce a 128-bit value represented as two u64 limbs `[lo, hi]`
/// (i.e. `hi * 2^64 + lo`) modulo `modulus`.
/// Mirrors `OPERATOR_GPU_64::reduce(Data64 coeff[2], Modulus64 mod)`.
#[inline]
fn reduce_u128_limbs(lo: u64, hi: u64, modulus: &Modulus64) -> u64 {
    let val = (hi as u128) << 64 | (lo as u128);
    (val % modulus.value as u128) as u64
}

// ===== 1. encode_kernel_bfv (GPU) =====
// Places messages (or zero) at permuted locations.

#[gpu::cuda_kernel]
pub fn encode_kernel_bfv(
    message_encoded: &mut [u64],
    message: &[u64],
    location_info: &[u32],
    plain_mod_value: u64,
    message_size: u32,
) {
    let bid = gpu::block_id::<gpu::DimX>();
    let tid = gpu::thread_id::<gpu::DimX>();
    let bdim = gpu::block_dim::<gpu::DimX>();
    let idx = (bid * bdim + tid) as usize;

    let location = location_info[idx] as usize;
    let mut out = gpu::chunk_mut(message_encoded, gpu::MapLinear::new(1));

    if (idx as u32) < message_size {
        let raw = message[idx] as i64;
        let val = if raw < 0 {
            (raw + plain_mod_value as i64) as u64
        } else {
            raw as u64
        };
        let _ = location; // location used by CPU ref; chunk_mut handles output mapping
        out[0] = val;
    } else {
        let _ = location;
        out[0] = 0u64;
    }
}

/// CPU reference for `encode_kernel_bfv`.
/// Places messages at permuted `location_info` indices; indices beyond
/// `message_size` are zeroed.
pub fn encode_kernel_bfv_cpu(
    message_encoded: &mut [u64],
    message: &[u64],
    location_info: &[u32],
    plain_mod_value: u64,
    message_size: usize,
) {
    for idx in 0..location_info.len() {
        let location = location_info[idx] as usize;
        if idx < message_size {
            let raw = message[idx] as i64;
            let val = if raw < 0 {
                (raw + plain_mod_value as i64) as u64
            } else {
                raw as u64
            };
            message_encoded[location] = val;
        } else {
            message_encoded[location] = 0u64;
        }
    }
}

// ===== 2. decode_kernel_bfv (GPU) =====
// Reads values from permuted locations.

#[gpu::cuda_kernel]
pub fn decode_kernel_bfv_gpu(
    message: &mut [u64],
    message_encoded: &[u64],
    location_info: &[u32],
) {
    let bid = gpu::block_id::<gpu::DimX>();
    let tid = gpu::thread_id::<gpu::DimX>();
    let bdim = gpu::block_dim::<gpu::DimX>();
    let idx = (bid * bdim + tid) as usize;

    let mut out = gpu::chunk_mut(message, gpu::MapLinear::new(1));
    let location = location_info[idx] as usize;
    out[0] = message_encoded[location];
}

/// CPU reference for `decode_kernel_bfv`.
pub fn decode_kernel_bfv_cpu(
    message: &mut [u64],
    message_encoded: &[u64],
    location_info: &[u32],
) {
    for idx in 0..message.len() {
        let location = location_info[idx] as usize;
        message[idx] = message_encoded[location];
    }
}

// ===== 3. encode_kernel_double_ckks_conversion (CPU-only: requires f64 operations) =====
// Converts a single double value to RNS representation across all moduli.

/// CPU reference for `encode_kernel_double_ckks_conversion`.
/// Encodes a single f64 `message` into RNS plaintext at coefficient index `idx`.
///
/// CPU-only: requires f64 operations (fmod, round).
pub fn encode_kernel_double_ckks_conversion_cpu(
    plaintext: &mut [u64],
    message: f64,
    moduli: &[Modulus64],
    coeff_modulus_count: usize,
    n_power: u32,
) {
    let two_pow_64: f64 = (1u128 << 64) as f64;
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let coeff_double = message.round();
        let is_negative = coeff_double.is_sign_negative();
        let coeff_abs = coeff_double.abs();

        let lo = (coeff_abs % two_pow_64) as u64;
        let hi = (coeff_abs / two_pow_64) as u64;

        for i in 0..coeff_modulus_count {
            let reduced = reduce_u128_limbs(lo, hi, &moduli[i]);
            plaintext[idx + (i << n_power)] = if is_negative {
                mod_sub(moduli[i].value, reduced, &moduli[i])
            } else {
                reduced
            };
        }
    }
}

// ===== 4. encode_kernel_int_ckks_conversion (GPU) =====
// Encodes a single i64 value into RNS plaintext.

#[gpu::cuda_kernel]
pub fn encode_kernel_int_ckks_conversion(
    plaintext: &mut [u64],
    message: i64,
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
) {
    let bid_x = gpu::block_id::<gpu::DimX>();
    let tid = gpu::thread_id::<gpu::DimX>();
    let bdim = gpu::block_dim::<gpu::DimX>();
    let idx = (bid_x * bdim + tid) as usize;
    let block_y = gpu::block_id::<gpu::DimY>() as usize;
    let _location = idx + (block_y << (n_power as usize));

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let message_r = if message < 0 {
        (message + mod_val as i64) as u64
    } else {
        message as u64
    };

    // Barrett forced reduction inline
    let mut a = message_r;
    loop {
        let z = a as u128;
        let w = z >> (bit as u32 - 2);
        let w = (w * (mu as u128)) >> (bit as u32 + 3);
        let w = w * (mod_val as u128);
        let r = (z - w) as u64;
        a = r;
        if a < mod_val {
            break;
        }
    }
    let mut out = gpu::chunk_mut(plaintext, gpu::MapLinear::new(1));
    out[0] = a;
}

/// CPU reference for `encode_kernel_int_ckks_conversion`.
pub fn encode_kernel_int_ckks_conversion_cpu(
    plaintext: &mut [u64],
    message: i64,
    moduli: &[Modulus64],
    n_power: u32,
) {
    let ring_size = 1usize << n_power;
    for block_y in 0..moduli.len() {
        let modulus = &moduli[block_y];
        let message_r = if message < 0 {
            (message + modulus.value as i64) as u64
        } else {
            message as u64
        };
        let val = mod_reduce_forced(message_r, modulus);
        for idx in 0..ring_size {
            plaintext[idx + (block_y << n_power)] = val;
        }
    }
}

// ===== 5. encode_kernel_coeff_ckks_conversion (CPU-only: requires f64 operations) =====
// Converts an array of f64 coefficients (scaled) to RNS representation.

/// CPU reference for `encode_kernel_coeff_ckks_conversion`.
///
/// CPU-only: requires f64 operations (fmod, round).
pub fn encode_kernel_coeff_ckks_conversion_cpu(
    plaintext: &mut [u64],
    message: &[f64],
    moduli: &[Modulus64],
    coeff_modulus_count: usize,
    scale: f64,
    n_power: u32,
) {
    let two_pow_64: f64 = (1u128 << 64) as f64;
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        let coeff_double = (message[idx] * scale).round();
        let is_negative = coeff_double.is_sign_negative();
        let coeff_abs = coeff_double.abs();

        let lo = (coeff_abs % two_pow_64) as u64;
        let hi = (coeff_abs / two_pow_64) as u64;

        for i in 0..coeff_modulus_count {
            let reduced = reduce_u128_limbs(lo, hi, &moduli[i]);
            plaintext[idx + (i << n_power)] = if is_negative {
                mod_sub(moduli[i].value, reduced, &moduli[i])
            } else {
                reduced
            };
        }
    }
}

// ===== 6. double_to_complex_kernel (CPU-only: requires f64 operations) =====

/// CPU reference for `double_to_complex_kernel`.
/// Converts real f64 values to complex (real, imag=0.0) pairs.
///
/// CPU-only: requires f64 operations.
pub fn double_to_complex_kernel_cpu(input: &[f64], output_real: &mut [f64], output_imag: &mut [f64]) {
    for idx in 0..input.len() {
        output_real[idx] = input[idx];
        output_imag[idx] = 0.0;
    }
}

// ===== 7. complex_to_double_kernel (CPU-only: requires f64 operations) =====

/// CPU reference for `complex_to_double_kernel`.
/// Extracts the real part of complex values.
///
/// CPU-only: requires f64 operations.
pub fn complex_to_double_kernel_cpu(input_real: &[f64], _input_imag: &[f64], output: &mut [f64]) {
    for idx in 0..output.len() {
        output[idx] = input_real[idx];
    }
}

// ===== 8. encode_kernel_ckks_conversion / encode_kernel_compose_Nttified_reverse_order =====
// (CPU-only: requires f64 operations)
// Encodes complex messages with bit-reversal permutation into RNS plaintext.

/// CPU reference for `encode_kernel_ckks_conversion` (bit-reversal permuted CKKS encoding).
/// This kernel reads complex messages in reverse order and encodes both real and imaginary
/// parts into the RNS plaintext.
///
/// CPU-only: requires f64 operations.
pub fn encode_kernel_ckks_conversion_cpu(
    plaintext: &mut [u64],
    complex_message_real: &[f64],
    complex_message_imag: &[f64],
    moduli: &[Modulus64],
    coeff_modulus_count: usize,
    reverse_order: &[u32],
    n_power: u32,
) {
    let two_pow_64: f64 = (1u128 << 64) as f64;
    let slot_count = 1usize << (n_power - 1);
    let offset = slot_count; // 1 << (n_power - 1)

    for idx in 0..slot_count {
        let order = reverse_order[idx] as usize;

        // Real part
        {
            let coeff_double = complex_message_real[order].round();
            let is_negative = coeff_double.is_sign_negative();
            let coeff_abs = coeff_double.abs();

            let lo = (coeff_abs % two_pow_64) as u64;
            let hi = (coeff_abs / two_pow_64) as u64;

            for i in 0..coeff_modulus_count {
                let reduced = reduce_u128_limbs(lo, hi, &moduli[i]);
                plaintext[idx + (i << n_power)] = if is_negative {
                    mod_sub(moduli[i].value, reduced, &moduli[i])
                } else {
                    reduced
                };
            }
        }

        // Imaginary part
        {
            let coeff_double = complex_message_imag[order].round();
            let is_negative = coeff_double.is_sign_negative();
            let coeff_abs = coeff_double.abs();

            let lo = (coeff_abs % two_pow_64) as u64;
            let hi = (coeff_abs / two_pow_64) as u64;

            for i in 0..coeff_modulus_count {
                let reduced = reduce_u128_limbs(lo, hi, &moduli[i]);
                plaintext[idx + offset + (i << n_power)] = if is_negative {
                    mod_sub(moduli[i].value, reduced, &moduli[i])
                } else {
                    reduced
                };
            }
        }
    }
}

// ===== Big-integer helpers for CRT compose/decompose =====

/// Multiply a big-integer `a` (of `len` limbs) by a single u64 `scalar`,
/// writing the result into `result` (of `len` limbs, truncated).
fn bigint_multiply(a: &[u64], scalar: u64, result: &mut [u64]) {
    let len = a.len();
    let mut carry: u128 = 0;
    for i in 0..len {
        let prod = (a[i] as u128) * (scalar as u128) + carry;
        result[i] = prod as u64;
        carry = prod >> 64;
    }
}

/// In-place add: `a += b`. Returns carry (0 or 1).
fn bigint_add_inplace(a: &mut [u64], b: &[u64]) -> u64 {
    let mut carry: u64 = 0;
    for i in 0..a.len() {
        let sum = (a[i] as u128) + (b[i] as u128) + (carry as u128);
        a[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }
    carry
}

/// Returns true if `a >= b` (big-integer comparison, MSB-first).
fn bigint_is_greater_or_equal(a: &[u64], b: &[u64]) -> bool {
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

/// Compute `result = a - b` (big-integer subtraction, assumes a >= b).
fn bigint_sub(a: &[u64], b: &[u64], result: &mut [u64]) {
    let mut borrow: u64 = 0;
    for i in 0..a.len() {
        let (diff, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
    }
}

/// Convert a big-integer to f64, using `inv_scale` and `two_pow_64` per-limb scaling.
/// If `is_negative`, compute `decryption_modulus - compose_result` with sign.
fn bigint_to_f64_scaled(
    compose_result: &[u64],
    decryption_modulus: &[u64],
    is_negative: bool,
    inv_scale: f64,
    two_pow_64: f64,
) -> f64 {
    let len = compose_result.len();
    let mut result = 0.0f64;
    let mut scaled = inv_scale;

    if is_negative {
        for j in 0..len {
            if compose_result[j] > decryption_modulus[j] {
                let diff = compose_result[j] - decryption_modulus[j];
                if diff != 0 {
                    result += (diff as f64) * scaled;
                }
            } else {
                let diff = decryption_modulus[j] - compose_result[j];
                if diff != 0 {
                    result -= (diff as f64) * scaled;
                }
            }
            scaled *= two_pow_64;
        }
    } else {
        for j in 0..len {
            let c = compose_result[j];
            if c != 0 {
                result += (c as f64) * scaled;
            }
            scaled *= two_pow_64;
        }
    }
    result
}

/// CRT compose helper: given RNS residues for one coefficient, reconstruct via CRT
/// and convert to f64.
fn crt_compose_to_f64(
    plaintext: &[u64],
    idx: usize,
    moduli: &[Modulus64],
    mi_inv: &[u64],
    mi: &[u64],
    upper_half_threshold: &[u64],
    decryption_modulus: &[u64],
    coeff_modulus_count: usize,
    inv_scale: f64,
    two_pow_64: f64,
    n_power: u32,
) -> f64 {
    let mut compose_result = vec![0u64; coeff_modulus_count];
    let mut big_integer_result = vec![0u64; coeff_modulus_count];

    for i in 0..coeff_modulus_count {
        let base = plaintext[idx + (i << n_power)];
        let temp = mod_mul(base, mi_inv[i], &moduli[i]);

        let mi_row = &mi[i * coeff_modulus_count..(i + 1) * coeff_modulus_count];
        bigint_multiply(mi_row, temp, &mut big_integer_result);
        bigint_add_inplace(&mut compose_result, &big_integer_result);

        if bigint_is_greater_or_equal(&compose_result, decryption_modulus) {
            let mut tmp = vec![0u64; coeff_modulus_count];
            bigint_sub(&compose_result, decryption_modulus, &mut tmp);
            compose_result.copy_from_slice(&tmp);
        }
    }

    let is_negative = bigint_is_greater_or_equal(&compose_result, upper_half_threshold);
    bigint_to_f64_scaled(&compose_result, decryption_modulus, is_negative, inv_scale, two_pow_64)
}

// ===== 9. encode_kernel_compose (CPU-only: requires f64 operations) =====
// CRT-composes RNS plaintext back to complex values with bit-reversal.

/// CPU reference for `encode_kernel_compose`.
/// Reconstructs complex message from RNS plaintext via CRT composition.
///
/// - `mi_inv`: array of `M_i^{-1} mod q_i` values (length `coeff_modulus_count`)
/// - `mi`: flattened big-integer `M_i` values (`coeff_modulus_count × coeff_modulus_count` limbs)
/// - `upper_half_threshold`, `decryption_modulus`: big-integers of `coeff_modulus_count` limbs
///
/// CPU-only: requires f64 operations.
pub fn encode_kernel_compose_cpu(
    complex_message_real: &mut [f64],
    complex_message_imag: &mut [f64],
    plaintext: &[u64],
    moduli: &[Modulus64],
    mi_inv: &[u64],
    mi: &[u64],
    upper_half_threshold: &[u64],
    decryption_modulus: &[u64],
    coeff_modulus_count: usize,
    scale: f64,
    reverse_order: &[u32],
    n_power: u32,
) {
    let two_pow_64: f64 = (1u128 << 64) as f64;
    let inv_scale = 1.0 / scale;
    let slot_count = 1usize << (n_power - 1);
    let offset = slot_count;

    for idx in 0..slot_count {
        // Real part from first half of coefficients
        let result_real = crt_compose_to_f64(
            plaintext,
            idx,
            moduli,
            mi_inv,
            mi,
            upper_half_threshold,
            decryption_modulus,
            coeff_modulus_count,
            inv_scale,
            two_pow_64,
            n_power,
        );

        // Imaginary part from second half (offset by slot_count)
        let result_imag = crt_compose_to_f64(
            plaintext,
            idx + offset,
            moduli,
            mi_inv,
            mi,
            upper_half_threshold,
            decryption_modulus,
            coeff_modulus_count,
            inv_scale,
            two_pow_64,
            n_power,
        );

        let order = reverse_order[idx] as usize;
        complex_message_real[order] = result_real;
        complex_message_imag[order] = result_imag;
    }
}

// ===== 10. decode_kernel_coeff_ckks_compose (CPU-only: requires f64 operations) =====
// CRT-decomposes RNS plaintext to f64 coefficient values.

/// CPU reference for `decode_kernel_coeff_ckks_compose`.
/// Reconstructs f64 message coefficients from RNS plaintext via CRT composition.
///
/// CPU-only: requires f64 operations.
pub fn decode_kernel_coeff_ckks_compose_cpu(
    message: &mut [f64],
    plaintext: &[u64],
    moduli: &[Modulus64],
    mi_inv: &[u64],
    mi: &[u64],
    upper_half_threshold: &[u64],
    decryption_modulus: &[u64],
    coeff_modulus_count: usize,
    scale: f64,
    n_power: u32,
) {
    let two_pow_64: f64 = (1u128 << 64) as f64;
    let inv_scale = 1.0 / scale;
    let ring_size = 1usize << n_power;

    for idx in 0..ring_size {
        message[idx] = crt_compose_to_f64(
            plaintext,
            idx,
            moduli,
            mi_inv,
            mi,
            upper_half_threshold,
            decryption_modulus,
            coeff_modulus_count,
            inv_scale,
            two_pow_64,
            n_power,
        );
    }
}
