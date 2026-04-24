/// Element-wise modular addition, subtraction, and negation on RNS polynomials.
///
/// Data layout: flat arrays of length `ring_size * rns_count * cipher_count`.
/// For element at `(idx, idy, idz)`:
///   `location = idx + (idy << n_power) + ((rns_count * idz) << n_power)`
/// where `n_power = log2(ring_size)`.

use crate::modular::{mod_add, mod_mul, mod_sub, Modulus64};
use gpu::prelude::*;

// ---------------------------------------------------------------------------
// CPU reference functions
// ---------------------------------------------------------------------------

/// Element-wise modular addition on flat RNS polynomial arrays.
pub fn addition_cpu(
    in1: &[u64],
    in2: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                out[loc] = mod_add(in1[loc], in2[loc], &moduli[idy]);
            }
        }
    }
}

/// Element-wise modular subtraction on flat RNS polynomial arrays.
pub fn subtraction_cpu(
    in1: &[u64],
    in2: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                out[loc] = mod_sub(in1[loc], in2[loc], &moduli[idy]);
            }
        }
    }
}

/// Element-wise modular negation on flat RNS polynomial arrays.
pub fn negation_cpu(
    in1: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                out[loc] = mod_sub(0, in1[loc], &moduli[idy]);
            }
        }
    }
}

/// Element-wise Barrett modular multiplication on flat RNS polynomial arrays.
pub fn multiply_elementwise_cpu(
    in1: &[u64],
    in2: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                out[loc] = mod_mul(in1[loc], in2[loc], &moduli[idy]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernels
//
// NOTE: We use `block_dim * block_id + thread_id` instead of `global_id`
// because `global_id` has an MLIR lowering bug (i64 -> index conversion).
// ---------------------------------------------------------------------------

/// GPU kernel: element-wise modular addition.
///
/// Launch with 1D grid: `ceil(total_elements / block_size)` blocks of `block_size` threads.
/// `mod_values` contains one modulus value per RNS level.
#[gpu::cuda_kernel]
pub fn addition_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let mod_val = mod_values[idy];

    let a = in1[gid as usize];
    let b = in2[gid as usize];
    let sum = a + b;
    let result = if sum >= mod_val { sum - mod_val } else { sum };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

/// GPU kernel: element-wise modular subtraction.
#[gpu::cuda_kernel]
pub fn subtraction_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let mod_val = mod_values[idy];

    let a = in1[gid as usize];
    let b = in2[gid as usize];
    let dif = a + mod_val - b;
    let result = if dif >= mod_val { dif - mod_val } else { dif };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

/// GPU kernel: element-wise modular negation (0 - x mod p).
#[gpu::cuda_kernel]
pub fn negation_kernel(
    in1: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let mod_val = mod_values[idy];

    let a = in1[gid as usize];
    let dif = mod_val - a;
    let result = if dif >= mod_val { dif - mod_val } else { dif };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

/// GPU kernel: element-wise Barrett modular multiplication.
///
/// Uses u128 intermediates for the Barrett reduction. Each thread computes
/// one element: `output[gid] = (in1[gid] * in2[gid]) mod modulus`.
#[gpu::cuda_kernel]
pub fn multiply_elementwise_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;

    let a = in1[gid as usize];
    let b = in2[gid as usize];

    // Inline Barrett multiplication
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
// CPU reference functions: BFV plaintext addition/subtraction
// ---------------------------------------------------------------------------

/// BFV plaintext addition to ciphertext (out-of-place with cipher_count dimension).
///
/// For `idz == 0`: encodes the plaintext coefficient with the delta scaling
/// `coeffdiv_plain` and correction factor, then adds to ciphertext.
/// For `idz != 0`: copies ciphertext unchanged.
pub fn addition_plain_bfv_poly_cpu(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    let message = plain[idx];
                    let ciphertext = cipher[loc];
                    let fix = message * q_mod_t;
                    let fix = fix + upper_threshold;
                    let fix = fix / plain_mod.value;
                    let result = mod_mul(message, coeffdiv_plain[idy], &moduli[idy]);
                    let result = mod_add(result, fix, &moduli[idy]);
                    let result = mod_add(result, ciphertext, &moduli[idy]);
                    output[loc] = result;
                } else {
                    output[loc] = cipher[loc];
                }
            }
        }
    }
}

/// BFV plaintext addition to ciphertext (inplace, no cipher_count dimension).
pub fn addition_plain_bfv_poly_inplace_cpu(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idy in 0..rns_count {
        for idx in 0..ring_size {
            let loc = idx + (idy << n_power);
            let message = plain[idx];
            let ciphertext = cipher[loc];
            let fix = message * q_mod_t;
            let fix = fix + upper_threshold;
            let fix = fix / plain_mod.value;
            let result = mod_mul(message, coeffdiv_plain[idy], &moduli[idy]);
            let result = mod_add(result, fix, &moduli[idy]);
            let result = mod_add(result, ciphertext, &moduli[idy]);
            output[loc] = result;
        }
    }
}

/// BFV plaintext subtraction from ciphertext (out-of-place with cipher_count dimension).
pub fn subtraction_plain_bfv_poly_cpu(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    let message = plain[idx];
                    let ciphertext = cipher[loc];
                    let fix = message * q_mod_t;
                    let fix = fix + upper_threshold;
                    let fix = fix / plain_mod.value;
                    let result = mod_mul(message, coeffdiv_plain[idy], &moduli[idy]);
                    let result = mod_add(result, fix, &moduli[idy]);
                    let result = mod_sub(ciphertext, result, &moduli[idy]);
                    output[loc] = result;
                } else {
                    output[loc] = cipher[loc];
                }
            }
        }
    }
}

/// BFV plaintext subtraction from ciphertext (inplace, no cipher_count dimension).
pub fn subtraction_plain_bfv_poly_inplace_cpu(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idy in 0..rns_count {
        for idx in 0..ring_size {
            let loc = idx + (idy << n_power);
            let message = plain[idx];
            let ciphertext = cipher[loc];
            let fix = message * q_mod_t;
            let fix = fix + upper_threshold;
            let fix = fix / plain_mod.value;
            let result = mod_mul(message, coeffdiv_plain[idy], &moduli[idy]);
            let result = mod_add(result, fix, &moduli[idy]);
            let result = mod_sub(ciphertext, result, &moduli[idy]);
            output[loc] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference functions: CKKS plaintext addition/subtraction
// ---------------------------------------------------------------------------

/// CKKS plaintext addition to ciphertext.
///
/// For `idz == 0`: modular add of ciphertext + plaintext.
/// For `idz != 0`: copies ciphertext unchanged.
pub fn addition_plain_ckks_poly_cpu(
    in1: &[u64],
    in2: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    out[loc] = mod_add(in1[loc], in2[loc], &moduli[idy]);
                } else {
                    out[loc] = in1[loc];
                }
            }
        }
    }
}

/// CKKS plaintext subtraction from ciphertext.
pub fn subtraction_plain_ckks_poly_cpu(
    in1: &[u64],
    in2: &[u64],
    out: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    out[loc] = mod_sub(in1[loc], in2[loc], &moduli[idy]);
                } else {
                    out[loc] = in1[loc];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-only: CKKS constant plaintext addition/subtraction
// (requires f64 → u64 conversion not available on GPU)
// ---------------------------------------------------------------------------

/// Reduce a 128-bit value (`low + high * 2^64`) modulo `modulus`.
fn reduce_u128_mod(low: u64, high: u64, modulus: &Modulus64) -> u64 {
    let val = (low as u128) | ((high as u128) << 64);
    (val % (modulus.value as u128)) as u64
}

// CPU-only: requires f64 → u64 conversion not available on GPU
/// CKKS constant plaintext addition: adds a rounded double `in2` to ciphertext.
///
/// Converts the double to a 128-bit integer, reduces mod each RNS prime,
/// then performs modular addition on the `idz == 0` component.
pub fn addition_constant_plain_ckks_poly_cpu(
    in1: &[u64],
    in2: f64,
    out: &mut [u64],
    moduli: &[Modulus64],
    two_pow_64: f64,
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    let coeff_double = in2.round();
                    let is_negative = coeff_double.is_sign_negative();
                    let coeff_double = coeff_double.abs();

                    let coeff_low = (coeff_double % two_pow_64) as u64;
                    let coeff_high = (coeff_double / two_pow_64) as u64;

                    let mut pt = reduce_u128_mod(coeff_low, coeff_high, &moduli[idy]);
                    if is_negative {
                        pt = mod_sub(moduli[idy].value, pt, &moduli[idy]);
                    }

                    out[loc] = mod_add(in1[loc], pt, &moduli[idy]);
                } else {
                    out[loc] = in1[loc];
                }
            }
        }
    }
}

// CPU-only: requires f64 → u64 conversion not available on GPU
/// CKKS constant plaintext subtraction: subtracts a rounded double `in2` from ciphertext.
pub fn subtraction_constant_plain_ckks_poly_cpu(
    in1: &[u64],
    in2: f64,
    out: &mut [u64],
    moduli: &[Modulus64],
    two_pow_64: f64,
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                if idz == 0 {
                    let coeff_double = in2.round();
                    let is_negative = coeff_double.is_sign_negative();
                    let coeff_double = coeff_double.abs();

                    let coeff_low = (coeff_double % two_pow_64) as u64;
                    let coeff_high = (coeff_double / two_pow_64) as u64;

                    let mut pt = reduce_u128_mod(coeff_low, coeff_high, &moduli[idy]);
                    if is_negative {
                        pt = mod_sub(moduli[idy].value, pt, &moduli[idy]);
                    }

                    out[loc] = mod_sub(in1[loc], pt, &moduli[idy]);
                } else {
                    out[loc] = in1[loc];
                }
            }
        }
    }
}

/// Zero all elements of a ciphertext (used to clear higher cipher components in CKKS).
///
/// In the CUDA version this zeros all elements unconditionally. The CPU
/// reference mirrors this: every location is set to 0.
pub fn set_zero_cipher_ckks_poly_cpu(
    data: &mut [u64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (idy << n_power) + ((rns_count * idz) << n_power);
                data[loc] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernels: BFV plaintext addition/subtraction
// ---------------------------------------------------------------------------

/// GPU kernel: BFV plaintext addition to ciphertext.
///
/// Flattened 1D launch over `ring_size * rns_count * cipher_count` elements.
/// Thread computes `idz` from the global index to decide add-path vs copy-path.
#[gpu::cuda_kernel]
pub fn addition_plain_bfv_poly_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    plain_mod_value: u64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = (gid & ((1u32 << n_power) - 1)) as usize;
    let idy = ((gid >> n_power) % rns_count) as usize;
    let idz = (gid >> n_power) / rns_count;

    let mut out = chunk_mut(output, MapLinear::new(1));

    if idz == 0 {
        let message = plain[idx];
        let ciphertext = cipher[gid as usize];

        let fix = message * q_mod_t;
        let fix = fix + upper_threshold;
        let fix = fix / plain_mod_value;

        // Barrett multiply: message * coeffdiv_plain[idy] mod modulus
        let mod_val = mod_values[idy];
        let bit = mod_bits[idy];
        let mu = mod_mus[idy];

        let z = (message as u128) * (coeffdiv_plain[idy] as u128);
        let w = z >> (bit as u32 - 2);
        let w = (w * (mu as u128)) >> (bit as u32 + 3);
        let w = w * (mod_val as u128);
        let mut result = (z - w) as u64;
        if result >= mod_val {
            result -= mod_val;
        }

        // result + fix mod modulus
        let sum = result + fix;
        result = if sum >= mod_val { sum - mod_val } else { sum };

        // result + ciphertext mod modulus
        let sum = result + ciphertext;
        result = if sum >= mod_val { sum - mod_val } else { sum };

        out[0] = result;
    } else {
        out[0] = cipher[gid as usize];
    }
}

/// GPU kernel: BFV plaintext addition to ciphertext (inplace, no cipher_count).
#[gpu::cuda_kernel]
pub fn addition_plain_bfv_poly_inplace_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    plain_mod_value: u64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = (gid & ((1u32 << n_power) - 1)) as usize;
    let idy = ((gid >> n_power) % rns_count) as usize;

    let message = plain[idx];
    let ciphertext = cipher[gid as usize];

    let fix = message * q_mod_t;
    let fix = fix + upper_threshold;
    let fix = fix / plain_mod_value;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let z = (message as u128) * (coeffdiv_plain[idy] as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut result = (z - w) as u64;
    if result >= mod_val {
        result -= mod_val;
    }

    let sum = result + fix;
    result = if sum >= mod_val { sum - mod_val } else { sum };

    let sum = result + ciphertext;
    result = if sum >= mod_val { sum - mod_val } else { sum };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

/// GPU kernel: BFV plaintext subtraction from ciphertext.
#[gpu::cuda_kernel]
pub fn subtraction_plain_bfv_poly_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    plain_mod_value: u64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = (gid & ((1u32 << n_power) - 1)) as usize;
    let idy = ((gid >> n_power) % rns_count) as usize;
    let idz = (gid >> n_power) / rns_count;

    let mut out = chunk_mut(output, MapLinear::new(1));

    if idz == 0 {
        let message = plain[idx];
        let ciphertext = cipher[gid as usize];

        let fix = message * q_mod_t;
        let fix = fix + upper_threshold;
        let fix = fix / plain_mod_value;

        let mod_val = mod_values[idy];
        let bit = mod_bits[idy];
        let mu = mod_mus[idy];

        let z = (message as u128) * (coeffdiv_plain[idy] as u128);
        let w = z >> (bit as u32 - 2);
        let w = (w * (mu as u128)) >> (bit as u32 + 3);
        let w = w * (mod_val as u128);
        let mut result = (z - w) as u64;
        if result >= mod_val {
            result -= mod_val;
        }

        // result + fix mod modulus
        let sum = result + fix;
        result = if sum >= mod_val { sum - mod_val } else { sum };

        // ciphertext - result mod modulus
        let dif = ciphertext + mod_val - result;
        result = if dif >= mod_val { dif - mod_val } else { dif };

        out[0] = result;
    } else {
        out[0] = cipher[gid as usize];
    }
}

/// GPU kernel: BFV plaintext subtraction from ciphertext (inplace, no cipher_count).
#[gpu::cuda_kernel]
pub fn subtraction_plain_bfv_poly_inplace_kernel(
    cipher: &[u64],
    plain: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    plain_mod_value: u64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idx = (gid & ((1u32 << n_power) - 1)) as usize;
    let idy = ((gid >> n_power) % rns_count) as usize;

    let message = plain[idx];
    let ciphertext = cipher[gid as usize];

    let fix = message * q_mod_t;
    let fix = fix + upper_threshold;
    let fix = fix / plain_mod_value;

    let mod_val = mod_values[idy];
    let bit = mod_bits[idy];
    let mu = mod_mus[idy];

    let z = (message as u128) * (coeffdiv_plain[idy] as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut result = (z - w) as u64;
    if result >= mod_val {
        result -= mod_val;
    }

    let sum = result + fix;
    result = if sum >= mod_val { sum - mod_val } else { sum };

    let dif = ciphertext + mod_val - result;
    result = if dif >= mod_val { dif - mod_val } else { dif };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

// ---------------------------------------------------------------------------
// GPU kernels: CKKS plaintext addition/subtraction
// ---------------------------------------------------------------------------

/// GPU kernel: CKKS plaintext addition to ciphertext.
///
/// For `idz == 0`: modular add of ciphertext + plaintext.
/// For `idz != 0`: copies ciphertext unchanged.
#[gpu::cuda_kernel]
pub fn addition_plain_ckks_poly_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let idz = (gid >> n_power) / rns_count;

    let mut out = chunk_mut(output, MapLinear::new(1));

    if idz == 0 {
        let mod_val = mod_values[idy];
        let a = in1[gid as usize];
        let b = in2[gid as usize];
        let sum = a + b;
        out[0] = if sum >= mod_val { sum - mod_val } else { sum };
    } else {
        out[0] = in1[gid as usize];
    }
}

/// GPU kernel: CKKS plaintext subtraction from ciphertext.
#[gpu::cuda_kernel]
pub fn subtraction_plain_ckks_poly_kernel(
    in1: &[u64],
    in2: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();
    let idy = ((gid >> n_power) % rns_count) as usize;
    let idz = (gid >> n_power) / rns_count;

    let mut out = chunk_mut(output, MapLinear::new(1));

    if idz == 0 {
        let mod_val = mod_values[idy];
        let a = in1[gid as usize];
        let b = in2[gid as usize];
        let dif = a + mod_val - b;
        out[0] = if dif >= mod_val { dif - mod_val } else { dif };
    } else {
        out[0] = in1[gid as usize];
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: set_zero_cipher_ckks_poly
// ---------------------------------------------------------------------------

/// GPU kernel: zeros all elements of a ciphertext polynomial.
///
/// The CUDA kernel writes 0 to every location unconditionally.
#[gpu::cuda_kernel]
pub fn set_zero_cipher_ckks_poly_kernel(
    data: &mut [u64],
) {
    let _gid = block_dim::<DimX>() * block_id::<DimX>() + thread_id::<DimX>();

    let mut out = chunk_mut(data, MapLinear::new(1));
    out[0] = 0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::Modulus64;
    use gpu_host::cuda_ctx;
    use rand::Rng;

    // Two 60-bit primes for testing
    const P0: u64 = 1152921504606846883;
    const P1: u64 = 1152921504606830593;

    fn make_moduli() -> Vec<Modulus64> {
        vec![Modulus64::new(P0), Modulus64::new(P1)]
    }

    fn mod_values(moduli: &[Modulus64]) -> Vec<u64> {
        moduli.iter().map(|m| m.value).collect()
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
    // CPU reference tests (small parameters)
    // -----------------------------------------------------------------------

    #[test]
    fn test_addition_cpu_basic() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let len = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(len, &moduli, n_power);
        let b = random_poly(len, &moduli, n_power);
        let mut out = vec![0u64; len];
        addition_cpu(&a, &b, &mut out, &moduli, n_power, rns_count, cipher_count);

        for i in 0..len {
            let idy = (i >> n_power) % rns_count;
            let expected = {
                let s = a[i] + b[i];
                if s >= moduli[idy].value {
                    s - moduli[idy].value
                } else {
                    s
                }
            };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_subtraction_cpu_basic() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let len = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(len, &moduli, n_power);
        let b = random_poly(len, &moduli, n_power);
        let mut out = vec![0u64; len];
        subtraction_cpu(&a, &b, &mut out, &moduli, n_power, rns_count, cipher_count);

        for i in 0..len {
            let idy = (i >> n_power) % rns_count;
            let expected = mod_sub(a[i], b[i], &moduli[idy]);
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_negation_cpu_double_negate() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let len = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(len, &moduli, n_power);
        let mut neg = vec![0u64; len];
        let mut neg_neg = vec![0u64; len];
        negation_cpu(&a, &mut neg, &moduli, n_power, rns_count, cipher_count);
        negation_cpu(&neg, &mut neg_neg, &moduli, n_power, rns_count, cipher_count);

        assert_eq!(a, neg_neg, "double negation should be identity");
    }

    // -----------------------------------------------------------------------
    // GPU vs CPU comparison tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addition_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let total = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(total, &moduli, n_power);
        let b = random_poly(total, &moduli, n_power);

        // CPU reference
        let mut cpu_out = vec![0u64; total];
        addition_cpu(&a, &b, &mut cpu_out, &moduli, n_power, rns_count, cipher_count);

        // GPU
        let mv = mod_values(&moduli);
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(a.as_slice()).expect("alloc");
            let d_b = ctx.new_tensor_view(b.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            addition_kernel::launch(
                config,
                ctx,
                m,
                &d_a,
                &d_b,
                &mut d_out,
                &d_mv,
                n_power,
                rns_count as u32,
            )
            .expect("kernel launch");
            d_out.copy_to_host(&mut gpu_out).expect("copy");
        });

        assert_eq!(gpu_out, cpu_out, "GPU addition mismatch");
    }

    #[test]
    fn test_subtraction_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let total = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(total, &moduli, n_power);
        let b = random_poly(total, &moduli, n_power);

        let mut cpu_out = vec![0u64; total];
        subtraction_cpu(&a, &b, &mut cpu_out, &moduli, n_power, rns_count, cipher_count);

        let mv = mod_values(&moduli);
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(a.as_slice()).expect("alloc");
            let d_b = ctx.new_tensor_view(b.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            subtraction_kernel::launch(
                config,
                ctx,
                m,
                &d_a,
                &d_b,
                &mut d_out,
                &d_mv,
                n_power,
                rns_count as u32,
            )
            .expect("kernel launch");
            d_out.copy_to_host(&mut gpu_out).expect("copy");
        });

        assert_eq!(gpu_out, cpu_out, "GPU subtraction mismatch");
    }

    #[test]
    fn test_negation_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let total = ring_size * rns_count * cipher_count;
        let moduli = make_moduli();

        let a = random_poly(total, &moduli, n_power);

        let mut cpu_out = vec![0u64; total];
        negation_cpu(&a, &mut cpu_out, &moduli, n_power, rns_count, cipher_count);

        let mv = mod_values(&moduli);
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(a.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            negation_kernel::launch(
                config,
                ctx,
                m,
                &d_a,
                &mut d_out,
                &d_mv,
                n_power,
                rns_count as u32,
            )
            .expect("kernel launch");
            d_out.copy_to_host(&mut gpu_out).expect("copy");
        });

        assert_eq!(gpu_out, cpu_out, "GPU negation mismatch");
    }

    #[test]
    fn test_gpu_multiply() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let rns_count = 2usize;
        let cipher_count = 1usize;
        let total = ring_size * rns_count * cipher_count;

        // Use primes from the task spec
        let moduli = vec![
            Modulus64::new(1152921504606846883),
            Modulus64::new(1152921504606846819),
        ];

        // Deterministic LCG for reproducibility
        let mut rng_state: u64 = 0xdeadbeef;
        let lcg_next = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        };
        let a: Vec<u64> = (0..total)
            .map(|i| {
                let idy = (i >> n_power) % rns_count;
                lcg_next(&mut rng_state) % moduli[idy].value
            })
            .collect();
        let b: Vec<u64> = (0..total)
            .map(|i| {
                let idy = (i >> n_power) % rns_count;
                lcg_next(&mut rng_state) % moduli[idy].value
            })
            .collect();

        // CPU reference
        let mut cpu_out = vec![0u64; total];
        multiply_elementwise_cpu(
            &a, &b, &mut cpu_out, &moduli, n_power, rns_count, cipher_count,
        );

        // GPU
        let mv: Vec<u64> = moduli.iter().map(|m| m.value).collect();
        let mb: Vec<u64> = moduli.iter().map(|m| m.bit).collect();
        let mm: Vec<u64> = moduli.iter().map(|m| m.mu).collect();
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_a = ctx.new_tensor_view(a.as_slice()).expect("alloc");
            let d_b = ctx.new_tensor_view(b.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let d_mb = ctx.new_tensor_view(mb.as_slice()).expect("alloc");
            let d_mm = ctx.new_tensor_view(mm.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            multiply_elementwise_kernel::launch(
                config,
                ctx,
                m,
                &d_a,
                &d_b,
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

        assert_eq!(gpu_out, cpu_out, "GPU multiply mismatch");
    }
}
