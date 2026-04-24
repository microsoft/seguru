/// BFV encryption kernels: public-key multiply and cipher-message add.
///
/// Data layout follows HEonGPU conventions:
/// - Public key `pk`: flat array of length `ring_size * rns_count * 2` (pk0 then pk1)
/// - Random polynomial `u`: flat array of length `ring_size * rns_count`
/// - Ciphertext: flat array of length `ring_size * rns_count * 2` (c0 then c1)

use crate::modular::{mod_add, mod_mul, mod_reduce_forced, mod_sub, Modulus64};
use gpu::prelude::*;

// ---------------------------------------------------------------------------
// CPU reference functions
// ---------------------------------------------------------------------------

/// Element-wise modular multiply: pk[i] * u[i] mod q, for each RNS level.
///
/// pk layout: flat array of length ring_size * rns_count * 2 (pk0 then pk1).
/// u layout: flat array of length ring_size * rns_count.
/// output: same layout as pk.
pub fn pk_u_mul_cpu(
    pk: &[u64],
    u: &[u64],
    pk_u: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_z in 0..2usize {
        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let pk_loc = idx + (block_y << n_power) + ((rns_count << n_power) * block_z);
                let u_loc = idx + (block_y << n_power);
                pk_u[pk_loc] = mod_mul(pk[pk_loc], u[u_loc], &moduli[block_y]);
            }
        }
    }
}

/// Add encoded message to the first cipher component (c0).
///
/// cipher layout: ring_size * rns_count * 2 (c0 then c1).
/// message layout: ring_size * rns_count (just one polynomial).
/// Only adds to block_z=0 (c0), copies block_z=1 (c1) unchanged.
pub fn cipher_message_add_cpu(
    cipher: &[u64],
    message: &[u64],
    output: &mut [u64],
    moduli: &[Modulus64],
    n_power: u32,
    rns_count: usize,
) {
    let ring_size = 1usize << n_power;
    for block_z in 0..2usize {
        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (block_y << n_power) + ((rns_count << n_power) * block_z);
                if block_z == 0 {
                    let msg_loc = idx + (block_y << n_power);
                    output[loc] = mod_add(cipher[loc], message[msg_loc], &moduli[block_y]);
                } else {
                    output[loc] = cipher[loc];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

/// GPU kernel: cipher_message_add — add encoded message to c0, copy c1.
///
/// Launch with 1D grid over `ring_size * rns_count * 2` elements.
#[gpu::cuda_kernel]
pub fn cipher_message_add_kernel(
    cipher: &[u64],
    message: &[u64],
    output: &mut [u64],
    mod_values: &[u64],
    n_power: u32,
    rns_count: u32,
) {
    let bid = block_id::<DimX>();
    let tid = thread_id::<DimX>();
    let bdim = block_dim::<DimX>();
    let gid = (bid * bdim + tid) as usize;

    let rns_chunk = (rns_count as usize) << (n_power as usize);
    let block_z = gid / rns_chunk; // 0 or 1
    let remainder = gid % rns_chunk;
    let block_y = remainder >> (n_power as usize);

    let mut out = chunk_mut(output, MapLinear::new(1));

    if block_z == 0 {
        let mod_val = mod_values[block_y];
        let sum = cipher[gid] + message[gid];
        out[0] = if sum >= mod_val { sum - mod_val } else { sum };
    } else {
        out[0] = cipher[gid];
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: pk_u_kernel
// ---------------------------------------------------------------------------

/// GPU kernel: element-wise Barrett multiply pk[i] * u[i] mod q.
///
/// Launch with grid (ring_size, rns_count, 2), block (256, 1, 1).
/// Flattened 1D launch: grid over `ring_size * rns_count * 2` elements.
#[gpu::cuda_kernel]
pub fn pk_u_kernel(
    pk: &[u64],
    u: &[u64],
    pk_u: &mut [u64],
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

    let rns_chunk = (rns_count as usize) << (n_power as usize);
    let _block_z = gid / rns_chunk; // 0 or 1 (for documentation)
    let remainder = gid % rns_chunk;
    let block_y = remainder >> (n_power as usize);

    // u index strips away the block_z component
    let u_loc = remainder;

    let mod_val = mod_values[block_y];
    let bit = mod_bits[block_y];
    let mu = mod_mus[block_y];

    let a = pk[gid];
    let b = u[u_loc];

    // Barrett multiplication
    let z = (a as u128) * (b as u128);
    let w = z >> (bit as u32 - 2);
    let w = (w * (mu as u128)) >> (bit as u32 + 3);
    let w = w * (mod_val as u128);
    let mut r = (z - w) as u64;
    if r >= mod_val {
        r -= mod_val;
    }

    let mut out = chunk_mut(pk_u, MapLinear::new(1));
    out[0] = r;
}

// ---------------------------------------------------------------------------
// CPU: enc_div_lastq — divide by last q modulus with rounding
// ---------------------------------------------------------------------------

/// CPU reference: enc_div_lastq — divides by the last RNS modulus with rounding
/// and adds the encoded message for the first cipher component (block_z=0).
///
/// This implements the base-conversion step in BFV encryption that removes the
/// auxiliary modulus q_last by:
///   1. Computing last_pk = pk[last_level] + e[last_level] + half (mod q_last)
///   2. Reducing last_pk into each target modulus q_j
///   3. Subtracting half_mod[j] to center the rounding
///   4. Computing (input_j - last_pk_j) * last_q_modinv[j] (mod q_j)
///   5. For c0 (block_z=0): adding the scaled plaintext message
///
/// # Parameters
/// - `pk`: public-key polynomial after NTT multiply with u, length = ring_size * (decomp_mod_count+1) * 2
/// - `e`: error polynomial, same layout as pk
/// - `plain`: plaintext polynomial, length = ring_size
/// - `ct`: output ciphertext, length = ring_size * decomp_mod_count * 2
/// - `moduli`: RNS moduli array, length = decomp_mod_count + 1
/// - `half`: floor(q_last / 2), for rounding
/// - `half_mod`: half reduced mod each q_j, length = decomp_mod_count
/// - `last_q_modinv`: inverse of q_last mod each q_j, length = decomp_mod_count
/// - `plain_mod`: plaintext modulus
/// - `q_mod_t`: Q mod t (product of all q_i mod plaintext modulus)
/// - `upper_threshold`: rounding threshold = (t - (1 + Q mod t)) / 2
/// - `coeffdiv_plain`: floor(Q/t) mod q_j for each j, length = decomp_mod_count
pub fn enc_div_lastq_cpu(
    pk: &[u64],
    e: &[u64],
    plain: &[u64],
    ct: &mut [u64],
    moduli: &[Modulus64],
    half: u64,
    half_mod: &[u64],
    last_q_modinv: &[u64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    decomp_mod_count: usize,
) {
    let ring_size = 1usize << n_power;
    let total_rns = decomp_mod_count + 1;

    for block_z in 0..2usize {
        for block_y in 0..decomp_mod_count {
            for idx in 0..ring_size {
                // Load last-level pk and e
                let last_pk_loc =
                    idx + (decomp_mod_count << n_power) + ((total_rns << n_power) * block_z);
                let mut last_pk = pk[last_pk_loc];
                let last_e = e[last_pk_loc];

                last_pk = mod_add(last_pk, last_e, &moduli[decomp_mod_count]);
                last_pk = mod_add(last_pk, half, &moduli[decomp_mod_count]);
                last_pk = mod_reduce_forced(last_pk, &moduli[block_y]);
                last_pk = mod_sub(last_pk, half_mod[block_y], &moduli[block_y]);

                // Load current-level pk and e
                let input_loc = idx + (block_y << n_power) + ((total_rns << n_power) * block_z);
                let mut input_ = pk[input_loc];
                let e_ = e[input_loc];
                input_ = mod_add(input_, e_, &moduli[block_y]);

                input_ = mod_sub(input_, last_pk, &moduli[block_y]);
                input_ = mod_mul(input_, last_q_modinv[block_y], &moduli[block_y]);

                if block_z == 0 {
                    let message = plain[idx];
                    let fix = message * q_mod_t;
                    let fix = fix + upper_threshold;
                    let fix = fix / plain_mod.value;

                    let mut ct_0 =
                        mod_mul(message, coeffdiv_plain[block_y], &moduli[block_y]);
                    ct_0 = mod_add(ct_0, fix, &moduli[block_y]);
                    input_ = mod_add(input_, ct_0, &moduli[block_y]);
                }

                let ct_loc = idx + (block_y << n_power)
                    + ((decomp_mod_count << n_power) * block_z);
                ct[ct_loc] = input_;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: enc_div_lastq_bfv — BFV variant with Q', Q, P bases
// ---------------------------------------------------------------------------

/// CPU reference: enc_div_lastq_bfv — BFV variant that iteratively removes
/// P auxiliary moduli using a triangular loop structure.
///
/// This is the multi-prime variant where the key-switching base has P_size
/// auxiliary primes (P basis). The algorithm iteratively peels off one P-prime
/// at a time from the highest index down:
///   1. For each P-prime (from highest to lowest):
///      a. Add half[i] to center the value
///      b. Reduce into remaining P-primes and subtract, then divide by the removed prime
///      c. Similarly reduce into the target Q-prime and divide
///   2. For c0 (block_z=0): add the scaled plaintext message
///
/// # Parameters
/// - `pk`: public-key after multiply, length = ring_size * Q_prime_size * 2
/// - `e`: error, same layout
/// - `plain`: plaintext, length = ring_size
/// - `ct`: output ciphertext, length = ring_size * Q_size * 2
/// - `moduli`: all Q'=Q∪P moduli, length = Q_prime_size
/// - `half`: rounding constants for each P-prime, length = P_size
/// - `half_mod`: half values reduced mod all primes, flattened triangular layout
/// - `last_q_modinv`: inverse of removed prime in each target, triangular layout
/// - `plain_mod`, `q_mod_t`, `upper_threshold`, `coeffdiv_plain`: same as enc_div_lastq
/// - `Q_prime_size`: total number of primes in Q∪P
/// - `Q_size`: number of Q primes
/// - `P_size`: number of P auxiliary primes (max 15 in CUDA)
pub fn enc_div_lastq_bfv_cpu(
    pk: &[u64],
    e: &[u64],
    plain: &[u64],
    ct: &mut [u64],
    moduli: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    plain_mod: &Modulus64,
    q_mod_t: u64,
    upper_threshold: u64,
    coeffdiv_plain: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
) {
    let ring_size = 1usize << n_power;

    for block_z in 0..2usize {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                // Load P-level pk+e values
                let mut last_pk = vec![0u64; p_size];
                for i in 0..p_size {
                    let loc = idx + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z);
                    let last_pk_ = pk[loc];
                    let last_e_ = e[loc];
                    last_pk[i] = mod_add(last_pk_, last_e_, &moduli[q_size + i]);
                }

                // Load Q-level pk+e
                let input_loc =
                    idx + (block_y << n_power) + ((q_prime_size << n_power) * block_z);
                let mut input_ = pk[input_loc];
                let e_ = e[input_loc];
                input_ = mod_add(input_, e_, &moduli[block_y]);

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let pi = p_size - 1 - i;
                    let mut last_pk_add_half_ = last_pk[pi];
                    last_pk_add_half_ = mod_add(
                        last_pk_add_half_,
                        half[i],
                        &moduli[q_prime_size - 1 - i],
                    );

                    // Update remaining P-primes
                    for j in 0..pi {
                        let mut temp1 =
                            mod_reduce_forced(last_pk_add_half_, &moduli[q_size + j]);
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + q_size + j],
                            &moduli[q_size + j],
                        );
                        temp1 = mod_sub(last_pk[j], temp1, &moduli[q_size + j]);
                        last_pk[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + q_size + j],
                            &moduli[q_size + j],
                        );
                    }

                    // Update the Q-target
                    let mut temp1 =
                        mod_reduce_forced(last_pk_add_half_, &moduli[block_y]);
                    temp1 = mod_sub(temp1, half_mod[location_ + block_y], &moduli[block_y]);
                    temp1 = mod_sub(input_, temp1, &moduli[block_y]);
                    input_ = mod_mul(temp1, last_q_modinv[location_ + block_y], &moduli[block_y]);

                    location_ += q_prime_size - 1 - i;
                }

                if block_z == 0 {
                    let message = plain[idx];
                    let fix = message * q_mod_t;
                    let fix = fix + upper_threshold;
                    let fix = fix / plain_mod.value;

                    let mut ct_0 =
                        mod_mul(message, coeffdiv_plain[block_y], &moduli[block_y]);
                    ct_0 = mod_add(ct_0, fix, &moduli[block_y]);
                    input_ = mod_add(input_, ct_0, &moduli[block_y]);
                }

                let ct_loc =
                    idx + (block_y << n_power) + ((q_size << n_power) * block_z);
                ct[ct_loc] = input_;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU: enc_div_lastq_ckks — CKKS variant (no message add)
// ---------------------------------------------------------------------------

/// CPU reference: enc_div_lastq_ckks — same iterative P-prime removal as
/// enc_div_lastq_bfv but without the plaintext message addition step.
///
/// CKKS does not add a scaled message during encryption; the message is
/// already encoded into the ciphertext via a different mechanism.
/// Otherwise the triangular loop structure is identical to the BFV variant.
///
/// # Parameters
/// Same as `enc_div_lastq_bfv_cpu` but without `plain`, `plain_mod`,
/// `q_mod_t`, `upper_threshold`, `coeffdiv_plain`.
pub fn enc_div_lastq_ckks_cpu(
    pk: &[u64],
    e: &[u64],
    ct: &mut [u64],
    moduli: &[Modulus64],
    half: &[u64],
    half_mod: &[u64],
    last_q_modinv: &[u64],
    n_power: u32,
    q_prime_size: usize,
    q_size: usize,
    p_size: usize,
) {
    let ring_size = 1usize << n_power;

    for block_z in 0..2usize {
        for block_y in 0..q_size {
            for idx in 0..ring_size {
                let mut last_pk = vec![0u64; p_size];
                for i in 0..p_size {
                    let loc = idx + ((q_size + i) << n_power)
                        + ((q_prime_size << n_power) * block_z);
                    let last_pk_ = pk[loc];
                    let last_e_ = e[loc];
                    last_pk[i] = mod_add(last_pk_, last_e_, &moduli[q_size + i]);
                }

                let input_loc =
                    idx + (block_y << n_power) + ((q_prime_size << n_power) * block_z);
                let mut input_ = pk[input_loc];
                let e_ = e[input_loc];
                input_ = mod_add(input_, e_, &moduli[block_y]);

                let mut location_ = 0usize;
                for i in 0..p_size {
                    let pi = p_size - 1 - i;
                    let mut last_pk_add_half_ = last_pk[pi];
                    last_pk_add_half_ = mod_add(
                        last_pk_add_half_,
                        half[i],
                        &moduli[q_prime_size - 1 - i],
                    );

                    for j in 0..pi {
                        let mut temp1 =
                            mod_reduce_forced(last_pk_add_half_, &moduli[q_size + j]);
                        temp1 = mod_sub(
                            temp1,
                            half_mod[location_ + q_size + j],
                            &moduli[q_size + j],
                        );
                        temp1 = mod_sub(last_pk[j], temp1, &moduli[q_size + j]);
                        last_pk[j] = mod_mul(
                            temp1,
                            last_q_modinv[location_ + q_size + j],
                            &moduli[q_size + j],
                        );
                    }

                    let mut temp1 =
                        mod_reduce_forced(last_pk_add_half_, &moduli[block_y]);
                    temp1 = mod_sub(temp1, half_mod[location_ + block_y], &moduli[block_y]);
                    temp1 = mod_sub(input_, temp1, &moduli[block_y]);
                    input_ = mod_mul(temp1, last_q_modinv[location_ + block_y], &moduli[block_y]);

                    location_ += q_prime_size - 1 - i;
                }

                let ct_loc =
                    idx + (block_y << n_power) + ((q_size << n_power) * block_z);
                ct[ct_loc] = input_;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU stubs: curand-dependent kernels
// ---------------------------------------------------------------------------

/// CPU stub for `initialize_random_states_kernel`.
///
/// The CUDA kernel initializes `curandState` objects for each thread using
/// `curand_init(seed, thread_id, 0, &states[thread_id])`. This sets up
/// per-thread PRNG states for generating random polynomials on the GPU.
///
/// This function is a no-op stub because curand state initialization has
/// no meaningful CPU equivalent — use the host-side RNG instead.
pub fn initialize_random_states_cpu(_seed: u64, _total_threads: usize) {
    // No-op: curand state initialization is GPU-specific.
    // On CPU, use rand::thread_rng() or similar.
}

/// CPU stub for `encrypt_lwe_kernel`.
///
/// The CUDA kernel performs LWE encryption:
///   For each of k segments:
///     1. For each coefficient i in [0, n):
///        - Generate random a[seg*n + i] via curand
///        - Accumulate local_sum += a[i] * sk[i]  (wrapping u32 arithmetic)
///     2. Warp-reduce and block-reduce local_sum via shared memory
///     3. output_b[seg] = input_b[seg] - block_sum  (wrapping i32 arithmetic)
///
/// This stub documents the algorithm but does not generate random values.
/// A real CPU implementation would use a host-side PRNG to fill output_a,
/// compute the inner products, and update output_b.
pub fn encrypt_lwe_cpu(
    _sk: &[i32],
    _output_a: &mut [i32],
    _output_b: &mut [i32],
    _n: usize,
    _k: usize,
) {
    // No-op stub: requires random number generation.
    // A CPU implementation would:
    // for seg in 0..k {
    //     let mut sum: u32 = 0;
    //     for i in 0..n {
    //         let r = rng.gen::<u32>() as i32;
    //         output_a[seg * n + i] = r;
    //         sum = sum.wrapping_add((r as u32).wrapping_mul(sk[i] as u32));
    //     }
    //     output_b[seg] = (output_b[seg] as u32).wrapping_sub(sum) as i32;
    // }
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

    const P0: u64 = 1152921504606846883;
    const P1: u64 = 1152921504606830593;

    fn make_moduli() -> Vec<Modulus64> {
        vec![Modulus64::new(P0), Modulus64::new(P1)]
    }

    fn mod_values(moduli: &[Modulus64]) -> Vec<u64> {
        moduli.iter().map(|m| m.value).collect()
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

    fn random_cipher_poly(ring_size: usize, moduli: &[Modulus64], n_power: u32) -> Vec<u64> {
        let rns_count = moduli.len();
        let mut rng = rand::rng();
        (0..ring_size * rns_count * 2)
            .map(|i| {
                let idy = (i >> n_power) % rns_count;
                rng.random::<u64>() % moduli[idy].value
            })
            .collect()
    }

    #[test]
    fn test_pk_u_mul_cpu_basic() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();

        let pk = random_cipher_poly(ring_size, &moduli, n_power);
        let u = random_rns_poly(ring_size, &moduli, n_power);
        let mut pk_u = vec![0u64; ring_size * rns_count * 2];

        pk_u_mul_cpu(&pk, &u, &mut pk_u, &moduli, n_power, rns_count);

        for block_z in 0..2usize {
            for block_y in 0..rns_count {
                for idx in 0..ring_size {
                    let pk_loc = idx + (block_y << n_power) + ((rns_count << n_power) * block_z);
                    let u_loc = idx + (block_y << n_power);
                    let expected = mod_mul(pk[pk_loc], u[u_loc], &moduli[block_y]);
                    assert_eq!(pk_u[pk_loc], expected, "mismatch at z={block_z} y={block_y} x={idx}");
                }
            }
        }
    }

    #[test]
    fn test_cipher_message_add_cpu_basic() {
        let n_power = 4u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();

        let cipher = random_cipher_poly(ring_size, &moduli, n_power);
        let message = random_rns_poly(ring_size, &moduli, n_power);
        let mut output = vec![0u64; ring_size * rns_count * 2];

        cipher_message_add_cpu(&cipher, &message, &mut output, &moduli, n_power, rns_count);

        // block_z=0: c0 + message
        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (block_y << n_power);
                let expected = mod_add(cipher[loc], message[loc], &moduli[block_y]);
                assert_eq!(output[loc], expected, "c0 mismatch at y={block_y} x={idx}");
            }
        }
        // block_z=1: cipher c1 unchanged
        let c1_offset = rns_count << n_power;
        for block_y in 0..rns_count {
            for idx in 0..ring_size {
                let loc = idx + (block_y << n_power) + c1_offset;
                assert_eq!(output[loc], cipher[loc], "c1 mismatch at y={block_y} x={idx}");
            }
        }
    }

    #[test]
    fn test_cipher_message_add_gpu_vs_cpu() {
        let n_power = 12u32;
        let ring_size = 1usize << n_power;
        let moduli = make_moduli();
        let rns_count = moduli.len();
        let total = ring_size * rns_count * 2;

        let cipher = random_cipher_poly(ring_size, &moduli, n_power);
        let message = random_rns_poly(ring_size, &moduli, n_power);

        // CPU reference
        let mut cpu_out = vec![0u64; total];
        cipher_message_add_cpu(&cipher, &message, &mut cpu_out, &moduli, n_power, rns_count);

        // GPU
        let mv = mod_values(&moduli);
        let mut gpu_out = vec![0u64; total];
        let block_size = 256u32;
        let grid_size = (total as u32 + block_size - 1) / block_size;

        cuda_ctx(0, |ctx, m| {
            let d_cipher = ctx.new_tensor_view(cipher.as_slice()).expect("alloc");
            let d_message = ctx.new_tensor_view(message.as_slice()).expect("alloc");
            let mut d_out = ctx
                .new_tensor_view(gpu_out.as_mut_slice())
                .expect("alloc");
            let d_mv = ctx.new_tensor_view(mv.as_slice()).expect("alloc");
            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, 0);
            cipher_message_add_kernel::launch(
                config,
                ctx,
                m,
                &d_cipher,
                &d_message,
                &mut d_out,
                &d_mv,
                n_power,
                rns_count as u32,
            )
            .expect("kernel launch");
            d_out.copy_to_host(&mut gpu_out).expect("copy");
        });

        assert_eq!(gpu_out, cpu_out, "GPU cipher_message_add mismatch");
    }
}
