/// Homomorphic multiplication kernels for BFV ciphertexts.
///
/// Cross-multiplication takes two ciphertexts ct1=(c0,c1) and ct2=(d0,d1) and
/// produces three components: (c0*d0, c0*d1+c1*d0, c1*d1).
///
/// Cipher-plain multiplication element-wise multiplies each ciphertext
/// component by a plaintext polynomial.

use crate::modular::{mod_add, mod_mul, Modulus64};
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
