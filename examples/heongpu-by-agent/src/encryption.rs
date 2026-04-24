/// BFV encryption kernels: public-key multiply and cipher-message add.
///
/// Data layout follows HEonGPU conventions:
/// - Public key `pk`: flat array of length `ring_size * rns_count * 2` (pk0 then pk1)
/// - Random polynomial `u`: flat array of length `ring_size * rns_count`
/// - Ciphertext: flat array of length `ring_size * rns_count * 2` (c0 then c1)

use crate::modular::{mod_add, mod_mul, Modulus64};
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
