/// TFHE bootstrapping and CKKS diagonal-matrix kernels ported from HEonGPU.
///
/// Source: `HEonGPU/src/lib/kernel/bootstrapping.cu`
///
/// This module contains:
/// - CKKS E-diagonal matrix generation and multiplication (complex f64 math)
/// - Complex vector scaling and rotation helpers
/// - BFV/CKKS mod-raise for RNS basis extension
/// - TFHE gate pre-computation kernels (NAND, AND, OR, NOR, XOR, XNOR, NOT)
/// - TFHE bootstrapping (blind rotation + NTT-domain accumulation)
/// - TFHE sample extraction and key switching
///
/// Most kernels are CPU references because they use shared memory, NTT,
/// complex f64 arithmetic, or stride-based loops.  `mod_raise_kernel` is
/// ported as a GPU kernel (single u64 output per thread, pure arithmetic).

use crate::modular::{mod_add, mod_mul, mod_reduce_forced, mod_sub, Modulus64};
use gpu::prelude::*;

// =====================================================================
// Helper functions (device helpers in CUDA)
// =====================================================================

/// Compute 5^index mod (4n - 1) via binary exponentiation.
///
/// Mirrors `exponent_calculation` __device__ function.
#[inline]
fn exponent_calculation(index: i32, n: i32) -> i32 {
    let modulus = ((n as u64) << 2) - 1;
    let mut result: u64 = 1;
    let five: u64 = 5;
    let bits = 32 - (index as u32).leading_zeros() as i32;
    for i in (0..bits).rev() {
        result = (result * result) & modulus;
        if ((index >> i) & 1) != 0 {
            result = (result * five) & modulus;
        }
    }
    result as i32
}

/// Map block_y (diagonal matrix index) to output row offset (forward).
#[inline]
fn matrix_location(index: i32) -> i32 {
    if index == 0 {
        0
    } else {
        3 * index - 1
    }
}

/// Map block_y to output row offset (inverse / reversed).
#[inline]
fn matrix_reverse_location(index: i32, grid_dim_y: i32) -> i32 {
    let total = (grid_dim_y - 1) * 3;
    if index == 0 {
        total
    } else {
        total - 3 * index
    }
}

/// Torus modulus switching: convert a 32-bit torus element to a Z_{2N} index.
///
/// Mirrors `torus_modulus_switch_log` __device__ function.
#[inline]
fn torus_modulus_switch_log(input: i32, modulus_log: i32) -> i32 {
    let range_log = 63 - modulus_log;
    let half_range: u64 = 1u64 << (range_log - 1);
    let result64 = ((input as u32 as u64) << 32) + half_range;
    (result64 >> range_log as u32) as i32
}

/// Rotated access into a complex vector: `vec[(idx - rotate) mod n]` with
/// negation when the index wraps around.
///
/// Returns (real, imag) of the rotated element.
#[inline]
fn rotated_access(
    real: &[f64],
    imag: &[f64],
    base: usize,
    rotate_index: i32,
    idx: i32,
    n_power: i32,
) -> (f64, f64) {
    let n = 1i32 << n_power;
    let rot = ((idx - rotate_index) % n + n) % n;
    let loc = base + rot as usize;
    if (idx - rotate_index) < 0 && (((idx - rotate_index) % n) + n) % n != 0 {
        // negation on wrap-around
        (-real[loc], -imag[loc])
    } else {
        (real[loc], imag[loc])
    }
}

/// Complex multiplication: (a_re + i*a_im) * (b_re + i*b_im).
#[inline]
fn complex_mul(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
}

/// Complex addition.
#[inline]
fn complex_add(a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> (f64, f64) {
    (a_re + b_re, a_im + b_im)
}

/// Complex exponentiation by integer: w^exp via repeated squaring.
#[inline]
fn complex_exp(re: f64, im: f64, exp: i32) -> (f64, f64) {
    if exp == 0 {
        return (1.0, 0.0);
    }
    let mut result_re = 1.0;
    let mut result_im = 0.0;
    let mut base_re = re;
    let mut base_im = im;
    let mut e = exp as u32;
    while e > 0 {
        if e & 1 != 0 {
            let (rr, ri) = complex_mul(result_re, result_im, base_re, base_im);
            result_re = rr;
            result_im = ri;
        }
        let (br, bi) = complex_mul(base_re, base_im, base_re, base_im);
        base_re = br;
        base_im = bi;
        e >>= 1;
    }
    (result_re, result_im)
}

/// Complex inverse: 1 / (re + i*im).
#[inline]
fn complex_inv(re: f64, im: f64) -> (f64, f64) {
    let denom = re * re + im * im;
    (re / denom, -im / denom)
}

// =====================================================================
// CPU reference: E_diagonal_generate_kernel
// =====================================================================

/// Generate CKKS E-diagonal matrices for bootstrapping (forward direction).
///
/// `output_real`/`output_imag`: flat arrays of complex values, row-major.
/// Grid: (n / blockDim, n_power) — block_y iterates over log-levels.
pub fn e_diagonal_generate_cpu(
    output_real: &mut [f64],
    output_imag: &mut [f64],
    n_power: i32,
    grid_dim_y: i32,
) {
    let n = 1i32 << n_power;
    for block_y in 0..grid_dim_y {
        let logk = block_y + 1;
        let output_location = matrix_location(block_y);
        let v_size = 1i32 << (n_power - logk);

        for idx in 0..n {
            let index1 = idx & ((v_size << 1) - 1);
            let index2 = index1 >> (n_power - logk);

            let mut w1_re = 1.0f64;
            let mut w1_im = 0.0f64;
            let mut w2_re = 0.0f64;
            let mut w2_im = 0.0f64;
            let mut w3_re = 0.0f64;
            let mut w3_im = 0.0f64;

            let angle = std::f64::consts::PI / ((v_size as f64) * 4.0);
            let omega_re = angle.cos();
            let omega_im = angle.sin();
            let expo = exponent_calculation(index1, n);
            let (w_re, w_im) = complex_exp(omega_re, omega_im, expo);

            if block_y == 0 {
                if index2 == 1 {
                    w1_re = w_re;
                    w1_im = w_im;
                    w2_re = 1.0;
                    w2_im = 0.0;
                } else {
                    w2_re = w_re;
                    w2_im = w_im;
                }
                let loc0 = ((output_location as usize) << n_power as u32) + idx as usize;
                let loc1 = (((output_location + 1) as usize) << n_power as u32) + idx as usize;
                output_real[loc0] = w1_re;
                output_imag[loc0] = w1_im;
                output_real[loc1] = w2_re;
                output_imag[loc1] = w2_im;
            } else {
                if index2 == 1 {
                    w1_re = w_re;
                    w1_im = w_im;
                    w3_re = 1.0;
                    w3_im = 0.0;
                } else {
                    w2_re = w_re;
                    w2_im = w_im;
                }
                let loc0 = ((output_location as usize) << n_power as u32) + idx as usize;
                let loc1 = (((output_location + 1) as usize) << n_power as u32) + idx as usize;
                let loc2 = (((output_location + 2) as usize) << n_power as u32) + idx as usize;
                output_real[loc0] = w1_re;
                output_imag[loc0] = w1_im;
                output_real[loc1] = w2_re;
                output_imag[loc1] = w2_im;
                output_real[loc2] = w3_re;
                output_imag[loc2] = w3_im;
            }
        }
    }
}

// =====================================================================
// CPU reference: E_diagonal_inverse_generate_kernel
// =====================================================================

/// Generate CKKS E-diagonal matrices for bootstrapping (inverse direction).
pub fn e_diagonal_inverse_generate_cpu(
    output_real: &mut [f64],
    output_imag: &mut [f64],
    n_power: i32,
    grid_dim_y: i32,
) {
    let n = 1i32 << n_power;
    for block_y in 0..grid_dim_y {
        let logk = block_y + 1;
        let output_location = matrix_reverse_location(block_y, grid_dim_y);
        let v_size = 1i32 << (n_power - logk);

        for idx in 0..n {
            let index1 = idx & ((v_size << 1) - 1);
            let index2 = index1 >> (n_power - logk);

            let mut w1_re = 0.5f64;
            let mut w1_im = 0.0f64;
            let mut w2_re = 0.5f64;
            let mut w2_im = 0.0f64;
            let mut w3_re = 0.0f64;
            let mut w3_im = 0.0f64;

            if block_y == 0 {
                if index2 == 1 {
                    let angle = std::f64::consts::PI / ((v_size as f64) * 4.0);
                    let omega_re = angle.cos();
                    let omega_im = angle.sin();
                    let (inv_re, inv_im) = complex_inv(omega_re, omega_im);
                    let expo = exponent_calculation(index1, n);
                    let (exp_re, exp_im) = complex_exp(inv_re, inv_im, expo);
                    // divide by 2
                    w1_re = exp_re / 2.0;
                    w1_im = exp_im / 2.0;
                    // negate
                    w2_re = -w1_re;
                    w2_im = -w1_im;
                }
                let loc0 = ((output_location as usize) << n_power as u32) + idx as usize;
                let loc1 = (((output_location + 1) as usize) << n_power as u32) + idx as usize;
                output_real[loc0] = w1_re;
                output_imag[loc0] = w1_im;
                output_real[loc1] = w2_re;
                output_imag[loc1] = w2_im;
            } else {
                if index2 == 1 {
                    let angle = std::f64::consts::PI / ((v_size as f64) * 4.0);
                    let omega_re = angle.cos();
                    let omega_im = angle.sin();
                    let (inv_re, inv_im) = complex_inv(omega_re, omega_im);
                    let expo = exponent_calculation(index1, n);
                    let (exp_re, exp_im) = complex_exp(inv_re, inv_im, expo);
                    w1_re = exp_re / 2.0;
                    w1_im = exp_im / 2.0;
                    w2_re = 0.0;
                    w2_im = 0.0;
                    w3_re = -w1_re;
                    w3_im = -w1_im;
                }
                let loc0 = ((output_location as usize) << n_power as u32) + idx as usize;
                let loc1 = (((output_location + 1) as usize) << n_power as u32) + idx as usize;
                let loc2 = (((output_location + 2) as usize) << n_power as u32) + idx as usize;
                output_real[loc0] = w1_re;
                output_imag[loc0] = w1_im;
                output_real[loc1] = w2_re;
                output_imag[loc1] = w2_im;
                output_real[loc2] = w3_re;
                output_imag[loc2] = w3_im;
            }
        }
    }
}

// =====================================================================
// CPU reference: E_diagonal_inverse_matrix_mult_single_kernel
// =====================================================================

/// Copy 2 or 3 rows of a complex matrix (depending on `last`).
///
/// Used to initialize the output buffer for inverse diagonal multiplication.
pub fn e_diagonal_inverse_matrix_mult_single_cpu(
    input_real: &[f64],
    input_imag: &[f64],
    output_real: &mut [f64],
    output_imag: &mut [f64],
    last: bool,
    n_power: i32,
) {
    let n = 1usize << n_power as u32;
    let rows = if last { 2 } else { 3 };
    for i in 0..rows {
        for idx in 0..n {
            let loc = idx + (i << n_power as u32);
            output_real[loc] = input_real[loc];
            output_imag[loc] = input_imag[loc];
        }
    }
}

// =====================================================================
// CPU reference: E_diagonal_matrix_mult_kernel
// =====================================================================

/// Sparse diagonal-matrix multiply for CKKS E-matrix decomposition (forward).
///
/// Performs: `output[out_row] += L_m[diag_rotated] * R_m[in_row]`
/// over multiple iterations, accumulating into `output`.
#[allow(clippy::too_many_arguments)]
pub fn e_diagonal_matrix_mult_cpu(
    input_real: &[f64],
    input_imag: &[f64],
    output_real: &mut [f64],
    output_imag: &mut [f64],
    temp_real: &[f64],
    temp_imag: &[f64],
    diag_index: &[i32],
    input_index: &[i32],
    output_index: &[i32],
    iteration_count: i32,
    r_matrix_counter: i32,
    output_index_counter: i32,
    mul_index: i32,
    first1: bool,
    first2: bool,
    n_power: i32,
) {
    let n = 1usize << n_power as u32;
    let offset = if first1 { 2 } else { 3 };
    let l_matrix_loc = offset + 3 * mul_index;
    let l_matrix_size = 3;

    let mut r_ctr = r_matrix_counter;
    let mut o_ctr = output_index_counter;

    for i in 0..iteration_count {
        let diag_idx = diag_index[r_ctr as usize];
        for idx in 0..n {
            let (r_re, r_im) = if first2 {
                let loc = idx + ((i as usize) << n_power as u32);
                (input_real[loc], input_imag[loc])
            } else {
                let in_idx = input_index[(r_ctr - offset) as usize];
                let loc = idx + ((in_idx as usize) << n_power as u32);
                (temp_real[loc], temp_imag[loc])
            };

            for j in 0..l_matrix_size {
                let l_base = ((l_matrix_loc + j) as usize) << n_power as u32;
                let (l_re, l_im) =
                    rotated_access(input_real, input_imag, l_base, diag_idx, idx as i32, n_power);

                let out_loc_row = output_index[o_ctr as usize];
                let out_loc = ((out_loc_row as usize) << n_power as u32) + idx;

                let (prod_re, prod_im) = complex_mul(l_re, l_im, r_re, r_im);
                let (sum_re, sum_im) =
                    complex_add(output_real[out_loc], output_imag[out_loc], prod_re, prod_im);
                output_real[out_loc] = sum_re;
                output_imag[out_loc] = sum_im;

                o_ctr += 1;
            }
        }
        // reset counters per iteration — CUDA kernel runs per-thread
        o_ctr = output_index_counter + (i + 1) * l_matrix_size;
        r_ctr = r_matrix_counter + i + 1;
    }
}

// =====================================================================
// CPU reference: E_diagonal_inverse_matrix_mult_kernel
// =====================================================================

/// Sparse diagonal-matrix multiply for CKKS E-matrix decomposition (inverse).
#[allow(clippy::too_many_arguments)]
pub fn e_diagonal_inverse_matrix_mult_cpu(
    input_real: &[f64],
    input_imag: &[f64],
    output_real: &mut [f64],
    output_imag: &mut [f64],
    temp_real: &[f64],
    temp_imag: &[f64],
    diag_index: &[i32],
    input_index: &[i32],
    output_index: &[i32],
    iteration_count: i32,
    r_matrix_counter: i32,
    output_index_counter: i32,
    mul_index: i32,
    first: bool,
    last: bool,
    n_power: i32,
) {
    let n = 1usize << n_power as u32;
    let l_matrix_loc = 3 + 3 * mul_index;
    let l_matrix_size = if last { 2 } else { 3 };

    for i in 0..iteration_count {
        let r_ctr = r_matrix_counter + i;
        let diag_idx = diag_index[r_ctr as usize];

        for idx in 0..n {
            let (r_re, r_im) = if first {
                let loc = idx + ((i as usize) << n_power as u32);
                (input_real[loc], input_imag[loc])
            } else {
                let in_idx = input_index[(r_ctr - 3) as usize];
                let loc = idx + ((in_idx as usize) << n_power as u32);
                (temp_real[loc], temp_imag[loc])
            };

            for j in 0..l_matrix_size {
                let o_ctr = output_index_counter + i * l_matrix_size + j;
                let l_base = ((l_matrix_loc + j) as usize) << n_power as u32;
                let (l_re, l_im) =
                    rotated_access(input_real, input_imag, l_base, diag_idx, idx as i32, n_power);

                let out_loc_row = output_index[o_ctr as usize];
                let out_loc = ((out_loc_row as usize) << n_power as u32) + idx;

                let (prod_re, prod_im) = complex_mul(l_re, l_im, r_re, r_im);
                let (sum_re, sum_im) =
                    complex_add(output_real[out_loc], output_imag[out_loc], prod_re, prod_im);
                output_real[out_loc] = sum_re;
                output_imag[out_loc] = sum_im;
            }
        }
    }
}

// =====================================================================
// CPU reference: complex_vector_scale_kernel
// =====================================================================

/// Scale every element of a complex matrix by a constant complex factor.
///
/// `data_real`/`data_imag`: flat complex arrays with rows of length `1 << n_power`.
/// `num_rows`: number of rows (blockIdx.y dimension).
pub fn complex_vector_scale_cpu(
    data_real: &mut [f64],
    data_imag: &mut [f64],
    scaling_re: f64,
    scaling_im: f64,
    n_power: i32,
    num_rows: usize,
) {
    let n = 1usize << n_power as u32;
    for idy in 0..num_rows {
        for idx in 0..n {
            let loc = idx + (idy << n_power as u32);
            let (re, im) = complex_mul(data_real[loc], data_imag[loc], scaling_re, scaling_im);
            data_real[loc] = re;
            data_imag[loc] = im;
        }
    }
}

// =====================================================================
// CPU reference: vector_rotate_kernel
// =====================================================================

/// Rotate a complex vector by `rotate_index` positions with negacyclic wrap.
pub fn vector_rotate_cpu(
    input_real: &[f64],
    input_imag: &[f64],
    output_real: &mut [f64],
    output_imag: &mut [f64],
    rotate_index: i32,
    n_power: i32,
) {
    let n = 1usize << n_power as u32;
    for idx in 0..n {
        let (re, im) = rotated_access(input_real, input_imag, 0, rotate_index, idx as i32, n_power);
        output_real[idx] = re;
        output_imag[idx] = im;
    }
}

// =====================================================================
// CPU reference + GPU kernel: mod_raise_kernel
// =====================================================================

/// Mod-raise: extend an RNS coefficient from base q0 to all qi primes.
///
/// Uses centered reduction: if coeff >= q0/2, negate before reducing mod qi.
pub fn mod_raise_cpu(
    input: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
    cipher_count: usize,
) {
    let ring_size = 1usize << n_power;
    let q0 = moduli[0].value;
    for idz in 0..cipher_count {
        for idy in 0..rns_count {
            for idx in 0..ring_size {
                let loc_in = idx + (idz << n_power);
                let loc_out = idx + (idy << n_power) + (rns_count * idz) * ring_size;
                let coeff = input[loc_in];
                if idy == 0 {
                    output[loc_out] = coeff;
                } else {
                    let qi = moduli[idy].value;
                    if coeff >= (q0 >> 1) {
                        let neg_coeff = q0 - coeff;
                        let tmp = mod_reduce_forced(neg_coeff, &moduli[idy]);
                        output[loc_out] = qi - tmp;
                    } else {
                        output[loc_out] = mod_reduce_forced(coeff, &moduli[idy]);
                    }
                }
            }
        }
    }
}

/// GPU kernel: mod-raise for RNS basis extension.
///
/// Launch grid: (ring_size, rns_count, cipher_count).
/// Each thread produces one output element.
#[gpu::cuda_kernel]
pub fn mod_raise_gpu_kernel(
    input: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    mod_bits: &[u64],
    mod_mus: &[u64],
    n_power: u32,
    _rns_count: u32,
) {
    let bid_x = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let idx = (bid_x * bdim + tid) as usize;

    let bid_y = block_id::<DimY>() as usize; // rns index
    let bid_z = block_id::<DimZ>() as usize; // cipher index

    let loc_in = idx + (bid_z << n_power);

    let q0 = mod_values[0];
    let coeff = input[loc_in];

    let result = if bid_y == 0 {
        coeff
    } else {
        let qi = mod_values[bid_y];
        let bit = mod_bits[bid_y];
        let mu = mod_mus[bid_y];
        let is_negative = coeff >= (q0 >> 1);
        let abs_coeff = if is_negative { q0 - coeff } else { coeff };

        // Barrett reduction of abs_coeff mod qi
        let z = abs_coeff as u128;
        let w = z >> (bit as u32 - 2);
        let w = (w * (mu as u128)) >> (bit as u32 + 3);
        let w = w * (qi as u128);
        let mut r = (z - w) as u64;
        if r >= qi {
            r -= qi;
        }

        if is_negative { qi - r } else { r }
    };

    let mut out = chunk_mut(output, MapLinear::new(1));
    out[0] = result;
}

// =====================================================================
// CPU reference: TFHE gate pre-computation kernels
// =====================================================================

/// TFHE NAND gate pre-computation.
///
/// `output_a[i] = encoded - input1_a[i] - input2_a[i]` (wrapping u32 arithmetic).
/// `output_b[block] = encoded - input1_b[block] - input2_b[block]`.
pub fn tfhe_nand_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32.wrapping_sub(input1_a[offset + i] as u32).wrapping_sub(input2_a[offset + i] as u32);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_sub(input1_b[block_x] as u32)
            .wrapping_sub(input2_b[block_x] as u32);
        output_b[block_x] = b as i32;
    }
}

/// TFHE AND gate pre-computation.
///
/// `output_a[i] = input1_a[i] + input2_a[i]` (wrapping).
/// `output_b[block] = encoded + input1_b + input2_b`.
pub fn tfhe_and_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32.wrapping_add(input1_a[offset + i] as u32).wrapping_add(input2_a[offset + i] as u32);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_add(input1_b[block_x] as u32)
            .wrapping_add(input2_b[block_x] as u32);
        output_b[block_x] = b as i32;
    }
}

/// TFHE AND-with-first-NOT gate pre-computation.
///
/// `output_a[i] = -input1_a[i] + input2_a[i]`.
/// `output_b[block] = encoded - input1_b + input2_b`.
pub fn tfhe_and_first_not_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32
                .wrapping_sub(input1_a[offset + i] as u32)
                .wrapping_add(input2_a[offset + i] as u32);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_sub(input1_b[block_x] as u32)
            .wrapping_add(input2_b[block_x] as u32);
        output_b[block_x] = b as i32;
    }
}

/// TFHE NOR gate pre-computation.
///
/// `output_a[i] = -input1_a[i] - input2_a[i]`.
/// `output_b[block] = encoded - input1_b - input2_b`.
pub fn tfhe_nor_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32.wrapping_sub(input1_a[offset + i] as u32).wrapping_sub(input2_a[offset + i] as u32);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_sub(input1_b[block_x] as u32)
            .wrapping_sub(input2_b[block_x] as u32);
        output_b[block_x] = b as i32;
    }
}

/// TFHE OR gate pre-computation.
///
/// `output_a[i] = input1_a[i] + input2_a[i]`.
/// `output_b[block] = encoded + input1_b + input2_b`.
pub fn tfhe_or_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32.wrapping_add(input1_a[offset + i] as u32).wrapping_add(input2_a[offset + i] as u32);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_add(input1_b[block_x] as u32)
            .wrapping_add(input2_b[block_x] as u32);
        output_b[block_x] = b as i32;
    }
}

/// TFHE XNOR gate pre-computation.
///
/// `output_a[i] = 2 * (-input1_a[i] - input2_a[i])`.
/// `output_b[block] = encoded - 2*input1_b - 2*input2_b`.
pub fn tfhe_xnor_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32
                .wrapping_sub(input1_a[offset + i] as u32)
                .wrapping_sub(input2_a[offset + i] as u32);
            let a = a.wrapping_mul(2);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_sub((input1_b[block_x] as u32).wrapping_mul(2))
            .wrapping_sub((input2_b[block_x] as u32).wrapping_mul(2));
        output_b[block_x] = b as i32;
    }
}

/// TFHE XOR gate pre-computation.
///
/// `output_a[i] = 2 * (input1_a[i] + input2_a[i])`.
/// `output_b[block] = encoded + 2*input1_b + 2*input2_b`.
pub fn tfhe_xor_pre_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    input2_a: &[i32],
    input2_b: &[i32],
    encoded: i32,
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            let a = 0u32
                .wrapping_add(input1_a[offset + i] as u32)
                .wrapping_add(input2_a[offset + i] as u32);
            let a = a.wrapping_mul(2);
            output_a[offset + i] = a as i32;
        }
        let b = (encoded as u32)
            .wrapping_add((input1_b[block_x] as u32).wrapping_mul(2))
            .wrapping_add((input2_b[block_x] as u32).wrapping_mul(2));
        output_b[block_x] = b as i32;
    }
}

/// TFHE NOT gate computation.
///
/// `output_a[i] = -input1_a[i]`, `output_b[block] = -input1_b[block]`.
pub fn tfhe_not_comp_cpu(
    output_a: &mut [i32],
    output_b: &mut [i32],
    input1_a: &[i32],
    input1_b: &[i32],
    n: usize,
    num_blocks: usize,
) {
    for block_x in 0..num_blocks {
        let offset = block_x * n;
        for i in 0..n {
            output_a[offset + i] = (input1_a[offset + i] as u32).wrapping_neg() as i32;
        }
        output_b[block_x] = (input1_b[block_x] as u32).wrapping_neg() as i32;
    }
}

// =====================================================================
// CPU reference: tfhe_bootstrapping_kernel (fused single-step)
// =====================================================================

/// Small forward NTT (in-place, Cooley-Tukey, power-of-2).
///
/// `data`: length-N array of u64 in `[0, modulus.value)`.
/// `roots`: forward root-of-unity table.
fn small_forward_ntt(data: &mut [u64], roots: &[u64], modulus: &Modulus64) {
    let n = data.len();
    let mut len = n >> 1;
    let mut root_idx = 1;
    while len >= 1 {
        let mut start = 0;
        while start < n {
            let w = roots[root_idx];
            root_idx += 1;
            for j in start..start + len {
                let u = data[j];
                let v = mod_mul(data[j + len], w, modulus);
                data[j] = mod_add(u, v, modulus);
                data[j + len] = mod_sub(u, v, modulus);
            }
            start += len << 1;
        }
        len >>= 1;
    }
}

/// Small inverse NTT (in-place, Gentleman-Sande).
///
/// `n_inv`: multiplicative inverse of N modulo modulus.value.
fn small_inverse_ntt(data: &mut [u64], roots: &[u64], modulus: &Modulus64, n_inv: u64) {
    let n = data.len();
    let mut len = 1;
    let mut root_idx = 1;
    while len < n {
        let mut start = 0;
        while start < n {
            let w = roots[root_idx];
            root_idx += 1;
            for j in start..start + len {
                let u = data[j];
                let v = data[j + len];
                data[j] = mod_add(u, v, modulus);
                data[j + len] = mod_mul(mod_sub(u, v, modulus), w, modulus);
            }
            start += len << 1;
        }
        len <<= 1;
    }
    for x in data.iter_mut() {
        *x = mod_mul(*x, n_inv, modulus);
    }
}

/// TFHE bootstrapping — fused blind-rotation + NTT accumulation (single-step).
///
/// This is the monolithic version where all `n` rotations are done in one pass.
/// Each "block" processes one ciphertext sample.
#[allow(clippy::too_many_arguments)]
pub fn tfhe_bootstrapping_cpu(
    input_a: &[i32],
    input_b: &[i32],
    output: &mut [i32],
    boot_key: &[u64],
    forward_roots: &[u64],
    inverse_roots: &[u64],
    n_inverse: u64,
    modulus: &Modulus64,
    encoded: i32,
    bk_offset: i32,
    bk_mask: i32,
    bk_half: i32,
    n: usize,
    big_n: usize,
    n_power: i32,
    k: usize,
    bk_bit: i32,
    bk_length: usize,
    num_blocks: usize,
) {
    let threshold = modulus.value >> 1;

    for block_x in 0..num_blocks {
        let offset_lwe = block_x * n;

        let input_b_i = input_b[block_x];
        let mut input_b_i_n = torus_modulus_switch_log(input_b_i, n_power);
        input_b_i_n = (big_n as i32 * 2) - input_b_i_n;

        // accum: (k+1) vectors of length N
        let kp1 = k + 1;
        let mut accum = vec![vec![0i32; big_n]; kp1];

        // Initialize accum[k] with test vector
        for idx in 0..big_n {
            accum[k][idx] = if input_b_i_n < big_n as i32 {
                if (idx as i32) < input_b_i_n {
                    -encoded
                } else {
                    encoded
                }
            } else {
                let minus = input_b_i_n - big_n as i32;
                if (idx as i32) < minus {
                    encoded
                } else {
                    -encoded
                }
            };
        }

        // Blind rotation: iterate over n LWE coefficients
        for i in 0..n {
            let input_a_i = input_a[offset_lwe + i];
            let input_a_i_n = torus_modulus_switch_log(input_a_i, n_power);

            let offset_i = i * kp1 * bk_length * kp1 * big_n;

            let mut accum3 = vec![vec![0u64; big_n]; kp1];

            for i2 in 0..kp1 {
                // Negacyclic rotation of accum[i2] by input_a_i_n
                let mut accum2 = vec![0i32; big_n];
                for idx in 0..big_n {
                    let rotated = if input_a_i_n < big_n as i32 {
                        if (idx as i32) < input_a_i_n {
                            -(accum[i2][(big_n as i32 - input_a_i_n + idx as i32) as usize])
                        } else {
                            accum[i2][(idx as i32 - input_a_i_n) as usize]
                        }
                    } else {
                        let minus = input_a_i_n - big_n as i32;
                        if (idx as i32) < minus {
                            accum[i2][(big_n as i32 - minus + idx as i32) as usize]
                        } else {
                            -(accum[i2][(idx as i32 - minus) as usize])
                        }
                    };
                    accum2[idx] = rotated - accum[i2][idx];
                }

                let offset_i2 = i2 * bk_length * kp1 * big_n;

                for i3 in 0..bk_length {
                    let offset_i3 = offset_i + offset_i2 + i3 * kp1 * big_n;

                    // Decompose and convert to NTT domain
                    let mut ntt_data = vec![0u64; big_n];
                    let shift = 32 - (bk_bit * (i3 as i32 + 1));
                    for idx in 0..big_n {
                        let temp_val = (((accum2[idx].wrapping_add(bk_offset)) >> shift) & bk_mask)
                            - bk_half;
                        ntt_data[idx] = if temp_val <= 0 {
                            (modulus.value as i64 + temp_val as i64) as u64
                        } else {
                            temp_val as u64
                        };
                    }

                    small_forward_ntt(&mut ntt_data, forward_roots, modulus);

                    for i4 in 0..kp1 {
                        for idx in 0..big_n {
                            let bk_val = boot_key[offset_i3 + i4 * big_n + idx];
                            let mul = mod_mul(ntt_data[idx], bk_val, modulus);
                            accum3[i4][idx] = mod_add(accum3[i4][idx], mul, modulus);
                        }
                    }
                }
            }

            // INTT and post-process
            for i4 in 0..kp1 {
                small_inverse_ntt(&mut accum3[i4], inverse_roots, modulus, n_inverse);
                for idx in 0..big_n {
                    let post = if accum3[i4][idx] >= threshold {
                        (accum3[i4][idx] as i64 - modulus.value as i64) as i32
                    } else {
                        accum3[i4][idx] as i32
                    };
                    accum[i4][idx] = accum[i4][idx].wrapping_add(post);
                }
            }
        }

        // Write output
        let global_loc = block_x * kp1 * big_n;
        for i4 in 0..kp1 {
            for idx in 0..big_n {
                output[global_loc + i4 * big_n + idx] = accum[i4][idx];
            }
        }
    }
}

// =====================================================================
// CPU reference: tfhe_bootstrapping_kernel_unique_step1
// =====================================================================

/// TFHE bootstrapping step 1 (unique / first iteration).
///
/// Computes the initial blind-rotation for the first LWE coefficient,
/// decomposes, forward-NTTs, and multiplies by bootstrapping key in NTT domain.
#[allow(clippy::too_many_arguments)]
pub fn tfhe_bootstrapping_unique_step1_cpu(
    input_a: &[i32],
    input_b: &[i32],
    output: &mut [u64],
    boot_key: &[u64],
    forward_roots: &[u64],
    modulus: &Modulus64,
    encoded: i32,
    bk_offset: i32,
    bk_mask: i32,
    bk_half: i32,
    n: usize,
    big_n: usize,
    n_power: i32,
    k: usize,
    bk_bit: i32,
    bk_length: usize,
    num_blocks: usize,
) {
    let kp1 = k + 1;

    for block_x in 0..num_blocks {
        let offset_lwe = block_x * n;

        let input_b_reg = input_b[block_x];
        let mut input_b_reg_n = torus_modulus_switch_log(input_b_reg, n_power);
        input_b_reg_n = (big_n as i32 * 2) - input_b_reg_n;

        let input_a_reg = input_a[offset_lwe]; // first coefficient
        let input_a_reg_n = torus_modulus_switch_log(input_a_reg, n_power);

        for block_y in 0..kp1 {
            // Initialize test vector for accum[block_y]
            let mut temp = vec![0i32; big_n];
            if block_y == k {
                for idx in 0..big_n {
                    temp[idx] = if input_b_reg_n < big_n as i32 {
                        if (idx as i32) < input_b_reg_n {
                            -encoded
                        } else {
                            encoded
                        }
                    } else {
                        let minus = input_b_reg_n - big_n as i32;
                        if (idx as i32) < minus {
                            encoded
                        } else {
                            -encoded
                        }
                    };
                }
            }

            // Negacyclic rotation by input_a_reg_n, subtract original
            let mut temp2 = vec![0i32; big_n];
            for idx in 0..big_n {
                let rotated = if input_a_reg_n < big_n as i32 {
                    if (idx as i32) < input_a_reg_n {
                        -(temp[(big_n as i32 - input_a_reg_n + idx as i32) as usize])
                    } else {
                        temp[(idx as i32 - input_a_reg_n) as usize]
                    }
                } else {
                    let minus = input_a_reg_n - big_n as i32;
                    if (idx as i32) < minus {
                        temp[(big_n as i32 - minus + idx as i32) as usize]
                    } else {
                        -(temp[(idx as i32 - minus) as usize])
                    }
                };
                temp2[idx] = rotated - temp[idx];
            }

            for block_z in 0..bk_length {
                let offset_i2 = block_y * bk_length * kp1 * big_n;
                let offset_i3 = offset_i2 + block_z * kp1 * big_n;

                let offset_o = block_x * kp1 * bk_length * kp1 * big_n;
                let offset_o2 = block_y * bk_length * kp1 * big_n;
                let offset_o3 = offset_o + offset_o2 + block_z * kp1 * big_n;

                // Decompose
                let shift = 32 - (bk_bit * (block_z as i32 + 1));
                let mut ntt_data = vec![0u64; big_n];
                for idx in 0..big_n {
                    let val =
                        (((temp2[idx].wrapping_add(bk_offset)) >> shift) & bk_mask) - bk_half;
                    ntt_data[idx] = if val < 0 {
                        (modulus.value as i64 + val as i64) as u64
                    } else {
                        val as u64
                    };
                }

                small_forward_ntt(&mut ntt_data, forward_roots, modulus);

                for i in 0..kp1 {
                    for idx in 0..big_n {
                        let bk_val = boot_key[offset_i3 + i * big_n + idx];
                        let mul = mod_mul(ntt_data[idx], bk_val, modulus);
                        output[offset_o3 + i * big_n + idx] = mul;
                    }
                }
            }
        }
    }
}

// =====================================================================
// CPU reference: tfhe_bootstrapping_kernel_regular_step1
// =====================================================================

/// TFHE bootstrapping step 1 (regular / subsequent iterations).
///
/// Same as unique_step1, but reads the accumulator from `input_c` (previous
/// step's output) and uses `boot_index`-th LWE coefficient.
#[allow(clippy::too_many_arguments)]
pub fn tfhe_bootstrapping_regular_step1_cpu(
    input_a: &[i32],
    _input_b: &[i32],
    input_c: &[i32],
    output: &mut [u64],
    boot_key: &[u64],
    boot_index: usize,
    forward_roots: &[u64],
    modulus: &Modulus64,
    bk_offset: i32,
    bk_mask: i32,
    bk_half: i32,
    n: usize,
    big_n: usize,
    n_power: i32,
    k: usize,
    bk_bit: i32,
    bk_length: usize,
    num_blocks: usize,
) {
    let kp1 = k + 1;

    for block_x in 0..num_blocks {
        let offset_lwe = block_x * n;
        let input_a_reg = input_a[offset_lwe + boot_index];
        let input_a_reg_n = torus_modulus_switch_log(input_a_reg, n_power);

        for block_y in 0..kp1 {
            // Read accumulator from input_c
            let offset_acc = block_x * kp1 * big_n + block_y * big_n;
            let temp: Vec<i32> = (0..big_n).map(|idx| input_c[offset_acc + idx]).collect();

            // Negacyclic rotation by input_a_reg_n, subtract original
            let mut temp2 = vec![0i32; big_n];
            for idx in 0..big_n {
                let rotated = if input_a_reg_n < big_n as i32 {
                    if (idx as i32) < input_a_reg_n {
                        -(temp[(big_n as i32 - input_a_reg_n + idx as i32) as usize])
                    } else {
                        temp[(idx as i32 - input_a_reg_n) as usize]
                    }
                } else {
                    let minus = input_a_reg_n - big_n as i32;
                    if (idx as i32) < minus {
                        temp[(big_n as i32 - minus + idx as i32) as usize]
                    } else {
                        -(temp[(idx as i32 - minus) as usize])
                    }
                };
                temp2[idx] = rotated - temp[idx];
            }

            for block_z in 0..bk_length {
                let offset_i = boot_index * kp1 * bk_length * kp1 * big_n;
                let offset_i2 = block_y * bk_length * kp1 * big_n;
                let offset_i3 = offset_i + offset_i2 + block_z * kp1 * big_n;

                let offset_o = block_x * kp1 * bk_length * kp1 * big_n;
                let offset_o2 = block_y * bk_length * kp1 * big_n;
                let offset_o3 = offset_o + offset_o2 + block_z * kp1 * big_n;

                let shift = 32 - (bk_bit * (block_z as i32 + 1));
                let mut ntt_data = vec![0u64; big_n];
                for idx in 0..big_n {
                    let val =
                        (((temp2[idx].wrapping_add(bk_offset)) >> shift) & bk_mask) - bk_half;
                    ntt_data[idx] = if val < 0 {
                        (modulus.value as i64 + val as i64) as u64
                    } else {
                        val as u64
                    };
                }

                small_forward_ntt(&mut ntt_data, forward_roots, modulus);

                for i in 0..kp1 {
                    for idx in 0..big_n {
                        let bk_val = boot_key[offset_i3 + i * big_n + idx];
                        let mul = mod_mul(ntt_data[idx], bk_val, modulus);
                        output[offset_o3 + i * big_n + idx] = mul;
                    }
                }
            }
        }
    }
}

// =====================================================================
// CPU reference: tfhe_bootstrapping_kernel_unique_step2
// =====================================================================

/// TFHE bootstrapping step 2 (unique / first iteration).
///
/// Accumulates NTT-domain products from step 1, performs inverse NTT,
/// converts back to torus representation, and adds the test-vector init.
#[allow(clippy::too_many_arguments)]
pub fn tfhe_bootstrapping_unique_step2_cpu(
    input: &[u64],
    input_b: &[i32],
    output: &mut [i32],
    inverse_roots: &[u64],
    n_inverse: u64,
    modulus: &Modulus64,
    encoded: i32,
    _n: usize,
    big_n: usize,
    n_power: i32,
    k: usize,
    bk_length: usize,
    num_blocks: usize,
) {
    let kp1 = k + 1;
    let threshold = modulus.value >> 1;

    for block_x in 0..num_blocks {
        let offset_i_base = block_x * kp1 * bk_length * kp1 * big_n;

        let input_b_reg = input_b[block_x];
        let mut input_b_reg_n = torus_modulus_switch_log(input_b_reg, n_power);
        input_b_reg_n = (big_n as i32 * 2) - input_b_reg_n;

        for block_y in 0..kp1 {
            // Accumulate across all (k+1) and bk_length dimensions
            let mut accum = vec![0u64; big_n];
            for i in 0..kp1 {
                let offset_i2 = i * bk_length * kp1 * big_n;
                for j in 0..bk_length {
                    let offset_i3 = offset_i_base + offset_i2 + j * kp1 * big_n + block_y * big_n;
                    for idx in 0..big_n {
                        accum[idx] = mod_add(accum[idx], input[offset_i3 + idx], modulus);
                    }
                }
            }

            small_inverse_ntt(&mut accum, inverse_roots, modulus, n_inverse);

            // Post-process: convert back to torus
            let offset_o = block_x * kp1 * big_n + block_y * big_n;
            for idx in 0..big_n {
                let mut post = if accum[idx] >= threshold {
                    (accum[idx] as i64 - modulus.value as i64) as i32
                } else {
                    accum[idx] as i32
                };

                // Add test vector init for block_y == k
                if block_y == k {
                    let tv = if input_b_reg_n < big_n as i32 {
                        if (idx as i32) < input_b_reg_n {
                            -encoded
                        } else {
                            encoded
                        }
                    } else {
                        let minus = input_b_reg_n - big_n as i32;
                        if (idx as i32) < minus {
                            encoded
                        } else {
                            -encoded
                        }
                    };
                    post = post.wrapping_add(tv);
                }

                output[offset_o + idx] = post;
            }
        }
    }
}

// =====================================================================
// CPU reference: tfhe_bootstrapping_kernel_regular_step2
// =====================================================================

/// TFHE bootstrapping step 2 (regular / subsequent iterations).
///
/// Same accumulation and INTT as unique_step2, but adds result to existing
/// accumulator in `output` (no test-vector initialization).
#[allow(clippy::too_many_arguments)]
pub fn tfhe_bootstrapping_regular_step2_cpu(
    input: &[u64],
    output: &mut [i32],
    inverse_roots: &[u64],
    n_inverse: u64,
    modulus: &Modulus64,
    _n: usize,
    big_n: usize,
    k: usize,
    bk_length: usize,
    num_blocks: usize,
) {
    let kp1 = k + 1;
    let threshold = modulus.value >> 1;

    for block_x in 0..num_blocks {
        let offset_i_base = block_x * kp1 * bk_length * kp1 * big_n;

        for block_y in 0..kp1 {
            let mut accum = vec![0u64; big_n];
            for i in 0..kp1 {
                let offset_i2 = i * bk_length * kp1 * big_n;
                for j in 0..bk_length {
                    let offset_i3 = offset_i_base + offset_i2 + j * kp1 * big_n + block_y * big_n;
                    for idx in 0..big_n {
                        accum[idx] = mod_add(accum[idx], input[offset_i3 + idx], modulus);
                    }
                }
            }

            small_inverse_ntt(&mut accum, inverse_roots, modulus, n_inverse);

            let offset_o = block_x * kp1 * big_n + block_y * big_n;
            for idx in 0..big_n {
                let post = if accum[idx] >= threshold {
                    (accum[idx] as i64 - modulus.value as i64) as i32
                } else {
                    accum[idx] as i32
                };
                output[offset_o + idx] = output[offset_o + idx].wrapping_add(post);
            }
        }
    }
}

// =====================================================================
// CPU reference: tfhe_sample_extraction_kernel
// =====================================================================

/// TFHE sample extraction: extract an LWE sample from a RLWE ciphertext.
///
/// For each RLWE block, extracts coefficients into LWE format:
/// - `output_a[i] = input[i]` for `i <= index`, negated and reversed for `i > index`
/// - `output_b[block] = input[k*N]` (constant term of last polynomial)
pub fn tfhe_sample_extraction_cpu(
    input: &[i32],
    output_a: &mut [i32],
    output_b: &mut [i32],
    big_n: usize,
    k: usize,
    index: usize,
    num_blocks: usize,
) {
    let kp1 = k + 1;
    let inner_index = index + 1;

    for block_x in 0..num_blocks {
        let offset_i = block_x * kp1 * big_n;

        for block_y in 0..k {
            let offset_i2 = offset_i + block_y * big_n;
            let offset_o = block_x * k * big_n;
            let offset_o2 = offset_o + block_y * big_n;

            for idx in 0..big_n {
                let value = if idx < inner_index {
                    input[offset_i2 + idx]
                } else {
                    -(input[offset_i2 + big_n - idx])
                };
                output_a[offset_o2 + idx] = value;
            }
        }

        // Extract b
        let offset_i3 = offset_i + k * big_n;
        output_b[block_x] = input[offset_i3];
    }
}

// =====================================================================
// CPU reference: tfhe_key_switching_kernel
// =====================================================================

/// TFHE key switching: switch from dimension N*k to dimension n.
///
/// Decomposes each input coefficient, looks up key-switching key entries,
/// and accumulates the result.
#[allow(clippy::too_many_arguments)]
pub fn tfhe_key_switching_cpu(
    input_a: &[i32],
    input_b: &[i32],
    output_a: &mut [i32],
    output_b: &mut [i32],
    ks_key_a: &[i32],
    ks_key_b: &[i32],
    ks_base_bit: i32,
    ks_length: i32,
    n: usize,
    big_n: usize,
    k: usize,
    num_blocks: usize,
) {
    let base = 1i32 << ks_base_bit;
    let precision_offset = 1i32 << (32 - (1 + ks_base_bit * ks_length));
    let mask = base - 1;
    let nk = big_n * k;

    for block_x in 0..num_blocks {
        let offset_i = block_x * nk;
        let mut accum_a = vec![0i32; n];
        let mut accum_b: i32 = input_b[block_x];

        for i in 0..nk {
            let input_a_reg = input_a[offset_i + i];
            let offset_key_b_i = i * ks_length as usize * mask as usize;
            let offset_key_a_i = offset_key_b_i * n;

            for i2 in 0..ks_length as usize {
                let input_a_decomp = ((((input_a_reg.wrapping_add(precision_offset))
                    >> (32 - ((i2 as i32 + 1) * ks_base_bit)))
                    & mask)
                    + 1)
                    - 1;

                let offset_key_b_i2 = offset_key_b_i + i2 * mask as usize;
                let offset_key_a_i3 = i2 * mask as usize * n;

                if input_a_decomp != 0 {
                    let offset_key_a_i2 =
                        (input_a_decomp as usize - 1) * n + offset_key_a_i + offset_key_a_i3;

                    for i3 in 0..n {
                        accum_a[i3] = (accum_a[i3] as u32)
                            .wrapping_sub(ks_key_a[offset_key_a_i2 + i3] as u32)
                            as i32;
                    }

                    let ks_key_b_val = ks_key_b[offset_key_b_i2 + input_a_decomp as usize - 1];
                    accum_b = (accum_b as u32).wrapping_sub(ks_key_b_val as u32) as i32;
                }
            }
        }

        let offset_o = block_x * n;
        for i in 0..n {
            output_a[offset_o + i] = accum_a[i];
        }
        output_b[block_x] = accum_b;
    }
}
