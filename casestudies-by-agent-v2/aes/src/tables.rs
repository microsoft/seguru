//! AES round constants and lookup tables.
//!
//! Everything here is derived at compile time from the GF(2^8) field
//! definition, so the crate carries no transcribed table literals.

/// Multiplication in GF(2^8) modulo the AES polynomial x^8 + x^4 + x^3 + x + 1.
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p: u8 = 0;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let high = a & 0x80;
        a <<= 1;
        if high != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
        i += 1;
    }
    p
}

/// Multiplicative inverse in GF(2^8), with 0 mapped to 0.
const fn ginv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let mut i: u16 = 1;
    while i < 256 {
        if gmul(a, i as u8) == 1 {
            return i as u8;
        }
        i += 1;
    }
    0
}

const fn sbox_entry(x: u8) -> u8 {
    let b = ginv(x);
    b ^ b.rotate_left(1) ^ b.rotate_left(2) ^ b.rotate_left(3) ^ b.rotate_left(4) ^ 0x63
}

/// AES S-box.
pub const SBOX: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = sbox_entry(i as u8);
        i += 1;
    }
    t
};

/// AES inverse S-box.
pub const INV_SBOX: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// Encryption T-table `TE0[x] = [2*S(x), S(x), S(x), 3*S(x)]` as a big-endian word.
///
/// `TE1..TE3` are byte rotations of `TE0`, so the kernel only needs this one
/// table in shared memory and recovers the others with a single shift/or pair.
pub const TE0: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let s = SBOX[i];
        t[i] = u32::from_be_bytes([gmul(s, 2), s, s, gmul(s, 3)]);
        i += 1;
    }
    t
};

/// Decryption T-table `TD0[x] = [14*IS(x), 9*IS(x), 13*IS(x), 11*IS(x)]`.
pub const TD0: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let s = INV_SBOX[i];
        t[i] = u32::from_be_bytes([gmul(s, 14), gmul(s, 9), gmul(s, 13), gmul(s, 11)]);
        i += 1;
    }
    t
};

/// Round constants for the AES-128 key schedule.
pub const RCON: [u32; 10] =
    [0x0100_0000, 0x0200_0000, 0x0400_0000, 0x0800_0000, 0x1000_0000, 0x2000_0000, 0x4000_0000,
     0x8000_0000, 0x1b00_0000, 0x3600_0000];

/// `SubWord` applied to a big-endian word.
const fn sub_word(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        SBOX[b[0] as usize],
        SBOX[b[1] as usize],
        SBOX[b[2] as usize],
        SBOX[b[3] as usize],
    ])
}

/// AES-128 key expansion producing the 44 big-endian round-key words.
pub const fn key_expansion(key: &[u8; 16]) -> [u32; 44] {
    let mut rk = [0u32; 44];
    let mut i = 0;
    while i < 4 {
        rk[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        i += 1;
    }
    while i < 44 {
        let mut temp = rk[i - 1];
        if i % 4 == 0 {
            temp = sub_word(temp.rotate_left(8)) ^ RCON[i / 4 - 1];
        }
        rk[i] = rk[i - 4] ^ temp;
        i += 1;
    }
    rk
}

/// `InvMixColumns` applied to one big-endian column word.
const fn inv_mix_column(w: u32) -> u32 {
    let b = w.to_be_bytes();
    u32::from_be_bytes([
        gmul(b[0], 14) ^ gmul(b[1], 11) ^ gmul(b[2], 13) ^ gmul(b[3], 9),
        gmul(b[0], 9) ^ gmul(b[1], 14) ^ gmul(b[2], 11) ^ gmul(b[3], 13),
        gmul(b[0], 13) ^ gmul(b[1], 9) ^ gmul(b[2], 14) ^ gmul(b[3], 11),
        gmul(b[0], 11) ^ gmul(b[1], 13) ^ gmul(b[2], 9) ^ gmul(b[3], 14),
    ])
}

/// Round keys for the equivalent inverse cipher: the T-table decryption path
/// needs `InvMixColumns` folded into rounds 1..=9.
pub const fn inv_round_keys(enc: &[u32; 44]) -> [u32; 44] {
    let mut dk = *enc;
    let mut i = 4;
    while i < 40 {
        dk[i] = inv_mix_column(dk[i]);
        i += 1;
    }
    dk
}
