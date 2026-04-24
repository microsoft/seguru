/// AES-128 constants, S-boxes, T-tables, and key expansion.
/// All values from FIPS 197 (AES standard).

/// Forward S-box (SubBytes)
pub const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
    0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
    0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
    0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
    0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
    0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
    0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
    0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
    0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
    0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
    0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
    0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
    0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
    0x16,
];

/// Inverse S-box (InvSubBytes)
pub const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7,
    0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde,
    0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42,
    0xfa, 0xc3, 0x4e, 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49,
    0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c,
    0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15,
    0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7,
    0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc,
    0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad,
    0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d,
    0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b,
    0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, 0x1f, 0xdd, 0xa8,
    0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, 0x60, 0x51,
    0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0,
    0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c,
    0x7d,
];

/// Round constants for key expansion
pub const RCON: [u32; 10] = [
    0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000,
    0x80000000, 0x1b000000, 0x36000000,
];

/// Multiply by 2 in GF(2^8)
const fn xtime(x: u8) -> u8 {
    if x & 0x80 != 0 {
        (x << 1) ^ 0x1b
    } else {
        x << 1
    }
}

/// Multiply two bytes in GF(2^8)
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            result ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
        i += 1;
    }
    result
}

/// Generate forward T-table entry: Te[i] for S-box value s = SBOX[i]
/// Te0[i] = [2*s, s, s, 3*s] as big-endian u32
/// Te1, Te2, Te3 are rotations of Te0
const fn make_te0_entry(i: usize) -> u32 {
    let s = SBOX[i];
    let s2 = xtime(s);
    let s3 = s2 ^ s;
    ((s2 as u32) << 24) | ((s as u32) << 16) | ((s as u32) << 8) | (s3 as u32)
}

const fn make_te1_entry(i: usize) -> u32 {
    let v = make_te0_entry(i);
    (v >> 8) | (v << 24)
}

const fn make_te2_entry(i: usize) -> u32 {
    let v = make_te0_entry(i);
    (v << 16) | (v >> 16)
}

const fn make_te3_entry(i: usize) -> u32 {
    let v = make_te0_entry(i);
    (v >> 24) | (v << 8)
}

/// Generate inverse T-table entry
/// Td0[i] = [0xe*s, 0x9*s, 0xd*s, 0xb*s] for s = INV_SBOX[i]
const fn make_td0_entry(i: usize) -> u32 {
    let s = INV_SBOX[i];
    let se = gmul(s, 0x0e);
    let s9 = gmul(s, 0x09);
    let sd = gmul(s, 0x0d);
    let sb = gmul(s, 0x0b);
    ((se as u32) << 24) | ((s9 as u32) << 16) | ((sd as u32) << 8) | (sb as u32)
}

const fn make_td1_entry(i: usize) -> u32 {
    let v = make_td0_entry(i);
    (v >> 8) | (v << 24)
}

const fn make_td2_entry(i: usize) -> u32 {
    let v = make_td0_entry(i);
    (v << 16) | (v >> 16)
}

const fn make_td3_entry(i: usize) -> u32 {
    let v = make_td0_entry(i);
    (v >> 24) | (v << 8)
}

const fn build_te0() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_te0_entry(i);
        i += 1;
    }
    t
}
const fn build_te1() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_te1_entry(i);
        i += 1;
    }
    t
}
const fn build_te2() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_te2_entry(i);
        i += 1;
    }
    t
}
const fn build_te3() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_te3_entry(i);
        i += 1;
    }
    t
}
const fn build_td0() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_td0_entry(i);
        i += 1;
    }
    t
}
const fn build_td1() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_td1_entry(i);
        i += 1;
    }
    t
}
const fn build_td2() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_td2_entry(i);
        i += 1;
    }
    t
}
const fn build_td3() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = make_td3_entry(i);
        i += 1;
    }
    t
}

pub const TE0: [u32; 256] = build_te0();
pub const TE1: [u32; 256] = build_te1();
pub const TE2: [u32; 256] = build_te2();
pub const TE3: [u32; 256] = build_te3();

pub const TD0: [u32; 256] = build_td0();
pub const TD1: [u32; 256] = build_td1();
pub const TD2: [u32; 256] = build_td2();
pub const TD3: [u32; 256] = build_td3();

/// Apply S-box to each byte of a u32 word
fn sub_word(w: u32) -> u32 {
    let b0 = SBOX[((w >> 24) & 0xff) as usize] as u32;
    let b1 = SBOX[((w >> 16) & 0xff) as usize] as u32;
    let b2 = SBOX[((w >> 8) & 0xff) as usize] as u32;
    let b3 = SBOX[(w & 0xff) as usize] as u32;
    (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
}

/// Rotate word left by 8 bits
fn rot_word(w: u32) -> u32 {
    (w << 8) | (w >> 24)
}

/// AES-128 key expansion: 16-byte key → 44 u32 round keys (11 rounds × 4 words)
pub fn key_expansion(key: &[u8; 16]) -> [u32; 44] {
    let mut w = [0u32; 44];

    // First 4 words are the key itself (big-endian)
    for i in 0..4 {
        w[i] = ((key[4 * i] as u32) << 24)
            | ((key[4 * i + 1] as u32) << 16)
            | ((key[4 * i + 2] as u32) << 8)
            | (key[4 * i + 3] as u32);
    }

    for i in 4..44 {
        let mut temp = w[i - 1];
        if i % 4 == 0 {
            temp = sub_word(rot_word(temp)) ^ RCON[i / 4 - 1];
        }
        w[i] = w[i - 4] ^ temp;
    }

    w
}

/// Generate inverse round keys for decryption.
/// Round 0 and 10 keys are unchanged; rounds 1-9 get InvMixColumns applied.
pub fn inv_round_keys(enc_keys: &[u32; 44]) -> [u32; 44] {
    let mut dk = *enc_keys;

    // Apply InvMixColumns to round keys 1..9
    for round in 1..10 {
        for j in 0..4 {
            let w = dk[round * 4 + j];
            let b0 = ((w >> 24) & 0xff) as usize;
            let b1 = ((w >> 16) & 0xff) as usize;
            let b2 = ((w >> 8) & 0xff) as usize;
            let b3 = (w & 0xff) as usize;
            // Use T-tables to compute InvMixColumns on S-box output
            // But we need InvMixColumns on arbitrary bytes, so use direct GF multiply
            let r0 = (w >> 24) as u8;
            let r1 = (w >> 16) as u8;
            let r2 = (w >> 8) as u8;
            let r3 = w as u8;
            let _ = (b0, b1, b2, b3); // suppress unused

            let o0 = gmul(r0, 0x0e) ^ gmul(r1, 0x0b) ^ gmul(r2, 0x0d) ^ gmul(r3, 0x09);
            let o1 = gmul(r0, 0x09) ^ gmul(r1, 0x0e) ^ gmul(r2, 0x0b) ^ gmul(r3, 0x0d);
            let o2 = gmul(r0, 0x0d) ^ gmul(r1, 0x09) ^ gmul(r2, 0x0e) ^ gmul(r3, 0x0b);
            let o3 = gmul(r0, 0x0b) ^ gmul(r1, 0x0d) ^ gmul(r2, 0x09) ^ gmul(r3, 0x0e);

            dk[round * 4 + j] =
                ((o0 as u32) << 24) | ((o1 as u32) << 16) | ((o2 as u32) << 8) | (o3 as u32);
        }
    }

    dk
}

/// CPU reference encryption for testing (textbook AES-128 ECB, single block)
pub fn aes128_encrypt_block(plaintext: &[u8; 16], round_keys: &[u32; 44]) -> [u8; 16] {
    let mut s0 = u32::from_be_bytes([plaintext[0], plaintext[1], plaintext[2], plaintext[3]]);
    let mut s1 = u32::from_be_bytes([plaintext[4], plaintext[5], plaintext[6], plaintext[7]]);
    let mut s2 = u32::from_be_bytes([plaintext[8], plaintext[9], plaintext[10], plaintext[11]]);
    let mut s3 =
        u32::from_be_bytes([plaintext[12], plaintext[13], plaintext[14], plaintext[15]]);

    // Round 0: AddRoundKey
    s0 ^= round_keys[0];
    s1 ^= round_keys[1];
    s2 ^= round_keys[2];
    s3 ^= round_keys[3];

    // Rounds 1-9
    for r in 1..10 {
        let t0 = TE0[((s0 >> 24) & 0xff) as usize]
            ^ TE1[((s1 >> 16) & 0xff) as usize]
            ^ TE2[((s2 >> 8) & 0xff) as usize]
            ^ TE3[(s3 & 0xff) as usize]
            ^ round_keys[4 * r];
        let t1 = TE0[((s1 >> 24) & 0xff) as usize]
            ^ TE1[((s2 >> 16) & 0xff) as usize]
            ^ TE2[((s3 >> 8) & 0xff) as usize]
            ^ TE3[(s0 & 0xff) as usize]
            ^ round_keys[4 * r + 1];
        let t2 = TE0[((s2 >> 24) & 0xff) as usize]
            ^ TE1[((s3 >> 16) & 0xff) as usize]
            ^ TE2[((s0 >> 8) & 0xff) as usize]
            ^ TE3[(s1 & 0xff) as usize]
            ^ round_keys[4 * r + 2];
        let t3 = TE0[((s3 >> 24) & 0xff) as usize]
            ^ TE1[((s0 >> 16) & 0xff) as usize]
            ^ TE2[((s1 >> 8) & 0xff) as usize]
            ^ TE3[(s2 & 0xff) as usize]
            ^ round_keys[4 * r + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }

    // Round 10: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    let t0 = ((SBOX[((s0 >> 24) & 0xff) as usize] as u32) << 24)
        | ((SBOX[((s1 >> 16) & 0xff) as usize] as u32) << 16)
        | ((SBOX[((s2 >> 8) & 0xff) as usize] as u32) << 8)
        | (SBOX[(s3 & 0xff) as usize] as u32);
    let t1 = ((SBOX[((s1 >> 24) & 0xff) as usize] as u32) << 24)
        | ((SBOX[((s2 >> 16) & 0xff) as usize] as u32) << 16)
        | ((SBOX[((s3 >> 8) & 0xff) as usize] as u32) << 8)
        | (SBOX[(s0 & 0xff) as usize] as u32);
    let t2 = ((SBOX[((s2 >> 24) & 0xff) as usize] as u32) << 24)
        | ((SBOX[((s3 >> 16) & 0xff) as usize] as u32) << 16)
        | ((SBOX[((s0 >> 8) & 0xff) as usize] as u32) << 8)
        | (SBOX[(s1 & 0xff) as usize] as u32);
    let t3 = ((SBOX[((s3 >> 24) & 0xff) as usize] as u32) << 24)
        | ((SBOX[((s0 >> 16) & 0xff) as usize] as u32) << 16)
        | ((SBOX[((s1 >> 8) & 0xff) as usize] as u32) << 8)
        | (SBOX[(s2 & 0xff) as usize] as u32);

    let o0 = (t0 ^ round_keys[40]).to_be_bytes();
    let o1 = (t1 ^ round_keys[41]).to_be_bytes();
    let o2 = (t2 ^ round_keys[42]).to_be_bytes();
    let o3 = (t3 ^ round_keys[43]).to_be_bytes();

    [
        o0[0], o0[1], o0[2], o0[3], o1[0], o1[1], o1[2], o1[3], o2[0], o2[1], o2[2], o2[3],
        o3[0], o3[1], o3[2], o3[3],
    ]
}

/// CPU reference decryption for testing
pub fn aes128_decrypt_block(ciphertext: &[u8; 16], inv_keys: &[u32; 44]) -> [u8; 16] {
    let mut s0 =
        u32::from_be_bytes([ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]]);
    let mut s1 =
        u32::from_be_bytes([ciphertext[4], ciphertext[5], ciphertext[6], ciphertext[7]]);
    let mut s2 =
        u32::from_be_bytes([ciphertext[8], ciphertext[9], ciphertext[10], ciphertext[11]]);
    let mut s3 =
        u32::from_be_bytes([ciphertext[12], ciphertext[13], ciphertext[14], ciphertext[15]]);

    // AddRoundKey with last round key (round 10)
    s0 ^= inv_keys[40];
    s1 ^= inv_keys[41];
    s2 ^= inv_keys[42];
    s3 ^= inv_keys[43];

    // Rounds 9..1: InvShiftRows + InvSubBytes + AddRoundKey + InvMixColumns
    for r in (1..10).rev() {
        let t0 = TD0[((s0 >> 24) & 0xff) as usize]
            ^ TD1[((s3 >> 16) & 0xff) as usize]
            ^ TD2[((s2 >> 8) & 0xff) as usize]
            ^ TD3[(s1 & 0xff) as usize]
            ^ inv_keys[4 * r];
        let t1 = TD0[((s1 >> 24) & 0xff) as usize]
            ^ TD1[((s0 >> 16) & 0xff) as usize]
            ^ TD2[((s3 >> 8) & 0xff) as usize]
            ^ TD3[(s2 & 0xff) as usize]
            ^ inv_keys[4 * r + 1];
        let t2 = TD0[((s2 >> 24) & 0xff) as usize]
            ^ TD1[((s1 >> 16) & 0xff) as usize]
            ^ TD2[((s0 >> 8) & 0xff) as usize]
            ^ TD3[(s3 & 0xff) as usize]
            ^ inv_keys[4 * r + 2];
        let t3 = TD0[((s3 >> 24) & 0xff) as usize]
            ^ TD1[((s2 >> 16) & 0xff) as usize]
            ^ TD2[((s1 >> 8) & 0xff) as usize]
            ^ TD3[(s0 & 0xff) as usize]
            ^ inv_keys[4 * r + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }

    // Round 0: InvShiftRows + InvSubBytes + AddRoundKey (no InvMixColumns)
    let t0 = ((INV_SBOX[((s0 >> 24) & 0xff) as usize] as u32) << 24)
        | ((INV_SBOX[((s3 >> 16) & 0xff) as usize] as u32) << 16)
        | ((INV_SBOX[((s2 >> 8) & 0xff) as usize] as u32) << 8)
        | (INV_SBOX[(s1 & 0xff) as usize] as u32);
    let t1 = ((INV_SBOX[((s1 >> 24) & 0xff) as usize] as u32) << 24)
        | ((INV_SBOX[((s0 >> 16) & 0xff) as usize] as u32) << 16)
        | ((INV_SBOX[((s3 >> 8) & 0xff) as usize] as u32) << 8)
        | (INV_SBOX[(s2 & 0xff) as usize] as u32);
    let t2 = ((INV_SBOX[((s2 >> 24) & 0xff) as usize] as u32) << 24)
        | ((INV_SBOX[((s1 >> 16) & 0xff) as usize] as u32) << 16)
        | ((INV_SBOX[((s0 >> 8) & 0xff) as usize] as u32) << 8)
        | (INV_SBOX[(s3 & 0xff) as usize] as u32);
    let t3 = ((INV_SBOX[((s3 >> 24) & 0xff) as usize] as u32) << 24)
        | ((INV_SBOX[((s2 >> 16) & 0xff) as usize] as u32) << 16)
        | ((INV_SBOX[((s1 >> 8) & 0xff) as usize] as u32) << 8)
        | (INV_SBOX[(s0 & 0xff) as usize] as u32);

    let o0 = (t0 ^ inv_keys[0]).to_be_bytes();
    let o1 = (t1 ^ inv_keys[1]).to_be_bytes();
    let o2 = (t2 ^ inv_keys[2]).to_be_bytes();
    let o3 = (t3 ^ inv_keys[3]).to_be_bytes();

    [
        o0[0], o0[1], o0[2], o0[3], o1[0], o1[1], o1[2], o1[3], o2[0], o2[1], o2[2], o2[3],
        o3[0], o3[1], o3[2], o3[3],
    ]
}

/// Parse hex string to bytes
pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_expansion() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let w = key_expansion(&key);

        // First round key = the key itself
        assert_eq!(w[0], 0x2b7e1516);
        assert_eq!(w[1], 0x28aed2a6);
        assert_eq!(w[2], 0xabf71588);
        assert_eq!(w[3], 0x09cf4f3c);

        // Last round key (round 10)
        assert_eq!(w[40], 0xd014f9a8);
        assert_eq!(w[41], 0xc9ee2589);
        assert_eq!(w[42], 0xe13f0cc8);
        assert_eq!(w[43], 0xb6630ca6);
    }

    #[test]
    fn test_nist_encrypt() {
        // FIPS 197 Appendix B test vector
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];
        let expected: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a,
            0x0b, 0x32,
        ];

        let rk = key_expansion(&key);
        let ct = aes128_encrypt_block(&plaintext, &rk);
        assert_eq!(ct, expected);
    }

    #[test]
    fn test_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];

        let enc_keys = key_expansion(&key);
        let dec_keys = inv_round_keys(&enc_keys);

        let ct = aes128_encrypt_block(&plaintext, &enc_keys);
        let pt = aes128_decrypt_block(&ct, &dec_keys);
        assert_eq!(pt, plaintext);
    }
}
