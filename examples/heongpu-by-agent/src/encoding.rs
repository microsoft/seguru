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
