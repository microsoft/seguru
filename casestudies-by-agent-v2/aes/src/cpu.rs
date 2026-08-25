//! Scalar CPU AES-128-ECB used as a correctness oracle and benchmark baseline.

use crate::tables::{INV_SBOX, SBOX, inv_round_keys, key_expansion};

fn xtime(x: u8) -> u8 {
    (x << 1) ^ (((x >> 7) & 1) * 0x1b)
}

fn mul(a: u8, b: u8) -> u8 {
    let mut p = 0u8;
    let mut a = a;
    let mut b = b;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    p
}

fn add_round_key(state: &mut [u8; 16], rk: &[u32; 44], round: usize) {
    for c in 0..4 {
        let w = rk[round * 4 + c].to_be_bytes();
        for r in 0..4 {
            state[4 * c + r] ^= w[r];
        }
    }
}

fn shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    for r in 1..4 {
        for c in 0..4 {
            state[4 * c + r] = s[4 * ((c + r) % 4) + r];
        }
    }
}

fn inv_shift_rows(state: &mut [u8; 16]) {
    let s = *state;
    for r in 1..4 {
        for c in 0..4 {
            state[4 * ((c + r) % 4) + r] = s[4 * c + r];
        }
    }
}

fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let a: [u8; 4] = [state[4 * c], state[4 * c + 1], state[4 * c + 2], state[4 * c + 3]];
        state[4 * c] = mul(a[0], 2) ^ mul(a[1], 3) ^ a[2] ^ a[3];
        state[4 * c + 1] = a[0] ^ mul(a[1], 2) ^ mul(a[2], 3) ^ a[3];
        state[4 * c + 2] = a[0] ^ a[1] ^ mul(a[2], 2) ^ mul(a[3], 3);
        state[4 * c + 3] = mul(a[0], 3) ^ a[1] ^ a[2] ^ mul(a[3], 2);
    }
}

fn inv_mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let a: [u8; 4] = [state[4 * c], state[4 * c + 1], state[4 * c + 2], state[4 * c + 3]];
        state[4 * c] = mul(a[0], 14) ^ mul(a[1], 11) ^ mul(a[2], 13) ^ mul(a[3], 9);
        state[4 * c + 1] = mul(a[0], 9) ^ mul(a[1], 14) ^ mul(a[2], 11) ^ mul(a[3], 13);
        state[4 * c + 2] = mul(a[0], 13) ^ mul(a[1], 9) ^ mul(a[2], 14) ^ mul(a[3], 11);
        state[4 * c + 3] = mul(a[0], 11) ^ mul(a[1], 13) ^ mul(a[2], 9) ^ mul(a[3], 14);
    }
}

/// Encrypt a single 16-byte block with the expanded key `rk`.
pub fn encrypt_block(rk: &[u32; 44], block: &[u8; 16]) -> [u8; 16] {
    let mut state = *block;
    add_round_key(&mut state, rk, 0);
    for round in 1..10 {
        for b in state.iter_mut() {
            *b = SBOX[*b as usize];
        }
        shift_rows(&mut state);
        mix_columns(&mut state);
        add_round_key(&mut state, rk, round);
    }
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
    shift_rows(&mut state);
    add_round_key(&mut state, rk, 10);
    state
}

/// Decrypt a single 16-byte block with the expanded key `rk`.
pub fn decrypt_block(rk: &[u32; 44], block: &[u8; 16]) -> [u8; 16] {
    let mut state = *block;
    add_round_key(&mut state, rk, 10);
    for round in (1..10).rev() {
        inv_shift_rows(&mut state);
        for b in state.iter_mut() {
            *b = INV_SBOX[*b as usize];
        }
        add_round_key(&mut state, rk, round);
        inv_mix_columns(&mut state);
    }
    inv_shift_rows(&mut state);
    for b in state.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
    add_round_key(&mut state, rk, 0);
    state
}

/// Encrypt a whole buffer in ECB mode. `data.len()` must be a multiple of 16.
pub fn encrypt_ecb(key: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let rk = key_expansion(key);
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks_exact(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        out.extend_from_slice(&encrypt_block(&rk, &block));
    }
    out
}

/// Decrypt a whole buffer in ECB mode. `data.len()` must be a multiple of 16.
pub fn decrypt_ecb(key: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let rk = key_expansion(key);
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks_exact(16) {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        out.extend_from_slice(&decrypt_block(&rk, &block));
    }
    out
}

/// Convenience: the decryption round keys used by the GPU T-table kernel.
pub fn gpu_decrypt_round_keys(key: &[u8; 16]) -> [u32; 44] {
    inv_round_keys(&key_expansion(key))
}
