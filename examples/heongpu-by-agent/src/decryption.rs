/// BFV decryption kernels: secret-key multiplication and full decrypt.
///
/// Data layout follows HEonGPU conventions:
/// - Ciphertext component `c0`, `c1`: flat arrays of length `ring_size * rns_count`
/// - Secret key `sk`: flat array of length `ring_size * rns_count`

use crate::modular::{mod_add, mod_mul, Modulus64};
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
