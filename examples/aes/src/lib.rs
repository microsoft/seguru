pub mod aes_common;

use gpu::*;

// ============================================================
// T-table AES-128 ECB encrypt kernel
// ============================================================
// te_tables: concatenated [TE0(256) | TE1(256) | TE2(256) | TE3(256)] = 1024 u32
// S-box extracted from TE0: S[i] = (TE0[i] >> 16) & 0xff
#[gpu::cuda_kernel(dynamic_shared)]
pub fn aes128_encrypt_ttable_kernel(
    input: &[u32],
    output: &mut [u32],
    round_keys: &[u32],
    te_tables: &[u32],
    num_blocks: u32,
) {
    let bdim = block_dim::<DimX>();
    let tid = bdim * block_id::<DimX>() + thread_id::<DimX>();
    let ltid = thread_id::<DimX>();

    // Shared memory allocation BEFORE any divergent code
    let smem = smem_alloc.alloc::<u32>(1024);

    // Cooperative load: all threads load 4 entries each (block_size must be 256)
    let mut sc = smem.chunk_mut(MapLinear::new(4));
    let base = (ltid * 4) as usize;
    sc[0] = te_tables[base];
    sc[1] = te_tables[base + 1];
    sc[2] = te_tables[base + 2];
    sc[3] = te_tables[base + 3];
    sync_threads();

    // Guard: only threads with valid AES blocks proceed to compute
    if tid < num_blocks {
        let off = (tid * 4) as usize;
        let in_slice = &input[off..off + 4];
        let mut s0 = in_slice[0] ^ round_keys[0];
        let mut s1 = in_slice[1] ^ round_keys[1];
        let mut s2 = in_slice[2] ^ round_keys[2];
        let mut s3 = in_slice[3] ^ round_keys[3];

        // Rounds 1-9: T-table lookups
        let mut r: u32 = 1;
        while r < 10 {
            let rk_off = (4 * r) as usize;
            let t0 = *smem[((s0 >> 24) & 0xff) as usize]
                ^ *smem[256 + ((s1 >> 16) & 0xff) as usize]
                ^ *smem[512 + ((s2 >> 8) & 0xff) as usize]
                ^ *smem[768 + (s3 & 0xff) as usize]
                ^ round_keys[rk_off];
            let t1 = *smem[((s1 >> 24) & 0xff) as usize]
                ^ *smem[256 + ((s2 >> 16) & 0xff) as usize]
                ^ *smem[512 + ((s3 >> 8) & 0xff) as usize]
                ^ *smem[768 + (s0 & 0xff) as usize]
                ^ round_keys[rk_off + 1];
            let t2 = *smem[((s2 >> 24) & 0xff) as usize]
                ^ *smem[256 + ((s3 >> 16) & 0xff) as usize]
                ^ *smem[512 + ((s0 >> 8) & 0xff) as usize]
                ^ *smem[768 + (s1 & 0xff) as usize]
                ^ round_keys[rk_off + 2];
            let t3 = *smem[((s3 >> 24) & 0xff) as usize]
                ^ *smem[256 + ((s0 >> 16) & 0xff) as usize]
                ^ *smem[512 + ((s1 >> 8) & 0xff) as usize]
                ^ *smem[768 + (s2 & 0xff) as usize]
                ^ round_keys[rk_off + 3];
            s0 = t0;
            s1 = t1;
            s2 = t2;
            s3 = t3;
            r += 1;
        }

        // Round 10: S-box only (no MixColumns)
        // Extract S-box from TE0: S[i] = (TE0[i] >> 16) & 0xff
        let t0 = (((*smem[((s0 >> 24) & 0xff) as usize] >> 16) & 0xff) << 24)
            | (((*smem[((s1 >> 16) & 0xff) as usize] >> 16) & 0xff) << 16)
            | (((*smem[((s2 >> 8) & 0xff) as usize] >> 16) & 0xff) << 8)
            | ((*smem[(s3 & 0xff) as usize] >> 16) & 0xff);
        let t1 = (((*smem[((s1 >> 24) & 0xff) as usize] >> 16) & 0xff) << 24)
            | (((*smem[((s2 >> 16) & 0xff) as usize] >> 16) & 0xff) << 16)
            | (((*smem[((s3 >> 8) & 0xff) as usize] >> 16) & 0xff) << 8)
            | ((*smem[(s0 & 0xff) as usize] >> 16) & 0xff);
        let t2 = (((*smem[((s2 >> 24) & 0xff) as usize] >> 16) & 0xff) << 24)
            | (((*smem[((s3 >> 16) & 0xff) as usize] >> 16) & 0xff) << 16)
            | (((*smem[((s0 >> 8) & 0xff) as usize] >> 16) & 0xff) << 8)
            | ((*smem[(s1 & 0xff) as usize] >> 16) & 0xff);
        let t3 = (((*smem[((s3 >> 24) & 0xff) as usize] >> 16) & 0xff) << 24)
            | (((*smem[((s0 >> 16) & 0xff) as usize] >> 16) & 0xff) << 16)
            | (((*smem[((s1 >> 8) & 0xff) as usize] >> 16) & 0xff) << 8)
            | ((*smem[(s2 & 0xff) as usize] >> 16) & 0xff);

        // Write output
        let mut c = chunk_mut(
            output,
            reshape_map!([4] | [num_blocks] => layout: [i0, t0]),
        );
        c[0] = t0 ^ round_keys[40];
        c[1] = t1 ^ round_keys[41];
        c[2] = t2 ^ round_keys[42];
        c[3] = t3 ^ round_keys[43];
    }
}

// ============================================================
// T-table AES-128 ECB decrypt kernel (no shared memory, reads T-tables from global memory)
// ============================================================
// td_tables: concatenated [TD0(256) | TD1(256) | TD2(256) | TD3(256)] = 1024 u32
// inv_sbox_packed: 256 inv_sbox bytes packed as 64 u32 (big-endian)
#[gpu::cuda_kernel]
pub fn aes128_decrypt_ttable_kernel(
    input: &[u32],
    output: &mut [u32],
    inv_round_keys: &[u32],
    td_tables: &[u32],
    inv_sbox_packed: &[u32],
    num_blocks: u32,
) {
    let bdim = block_dim::<DimX>();
    let tid = bdim * block_id::<DimX>() + thread_id::<DimX>();
    if tid >= num_blocks {
        return;
    }

    let off = (tid * 4) as usize;
    let in_slice = &input[off..off + 4];
    let mut s0 = in_slice[0] ^ inv_round_keys[40];
    let mut s1 = in_slice[1] ^ inv_round_keys[41];
    let mut s2 = in_slice[2] ^ inv_round_keys[42];
    let mut s3 = in_slice[3] ^ inv_round_keys[43];

    // Rounds 9..1: T-table lookups from global memory
    let mut r: u32 = 9;
    while r >= 1 {
        let rk_off = (4 * r) as usize;
        let t0 = td_tables[((s0 >> 24) & 0xff) as usize]
            ^ td_tables[256 + ((s3 >> 16) & 0xff) as usize]
            ^ td_tables[512 + ((s2 >> 8) & 0xff) as usize]
            ^ td_tables[768 + (s1 & 0xff) as usize]
            ^ inv_round_keys[rk_off];
        let t1 = td_tables[((s1 >> 24) & 0xff) as usize]
            ^ td_tables[256 + ((s0 >> 16) & 0xff) as usize]
            ^ td_tables[512 + ((s3 >> 8) & 0xff) as usize]
            ^ td_tables[768 + (s2 & 0xff) as usize]
            ^ inv_round_keys[rk_off + 1];
        let t2 = td_tables[((s2 >> 24) & 0xff) as usize]
            ^ td_tables[256 + ((s1 >> 16) & 0xff) as usize]
            ^ td_tables[512 + ((s0 >> 8) & 0xff) as usize]
            ^ td_tables[768 + (s3 & 0xff) as usize]
            ^ inv_round_keys[rk_off + 2];
        let t3 = td_tables[((s3 >> 24) & 0xff) as usize]
            ^ td_tables[256 + ((s2 >> 16) & 0xff) as usize]
            ^ td_tables[512 + ((s1 >> 8) & 0xff) as usize]
            ^ td_tables[768 + (s0 & 0xff) as usize]
            ^ inv_round_keys[rk_off + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
        r -= 1;
    }

    // Round 0: InvShiftRows + InvSubBytes + AddRoundKey
    // Read inv_sbox from global memory
    let isb00 = (inv_sbox_packed[((s0 >> 24) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s0 >> 24) & 0xff) & 3))))
        & 0xff;
    let isb01 = (inv_sbox_packed[((s3 >> 16) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s3 >> 16) & 0xff) & 3))))
        & 0xff;
    let isb02 = (inv_sbox_packed[((s2 >> 8) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s2 >> 8) & 0xff) & 3))))
        & 0xff;
    let isb03 =
        (inv_sbox_packed[(s1 & 0xff) as usize >> 2] >> (8 * (3 - ((s1 & 0xff) & 3)))) & 0xff;

    let isb10 = (inv_sbox_packed[((s1 >> 24) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s1 >> 24) & 0xff) & 3))))
        & 0xff;
    let isb11 = (inv_sbox_packed[((s0 >> 16) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s0 >> 16) & 0xff) & 3))))
        & 0xff;
    let isb12 = (inv_sbox_packed[((s3 >> 8) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s3 >> 8) & 0xff) & 3))))
        & 0xff;
    let isb13 =
        (inv_sbox_packed[(s2 & 0xff) as usize >> 2] >> (8 * (3 - ((s2 & 0xff) & 3)))) & 0xff;

    let isb20 = (inv_sbox_packed[((s2 >> 24) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s2 >> 24) & 0xff) & 3))))
        & 0xff;
    let isb21 = (inv_sbox_packed[((s1 >> 16) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s1 >> 16) & 0xff) & 3))))
        & 0xff;
    let isb22 = (inv_sbox_packed[((s0 >> 8) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s0 >> 8) & 0xff) & 3))))
        & 0xff;
    let isb23 =
        (inv_sbox_packed[(s3 & 0xff) as usize >> 2] >> (8 * (3 - ((s3 & 0xff) & 3)))) & 0xff;

    let isb30 = (inv_sbox_packed[((s3 >> 24) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s3 >> 24) & 0xff) & 3))))
        & 0xff;
    let isb31 = (inv_sbox_packed[((s2 >> 16) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s2 >> 16) & 0xff) & 3))))
        & 0xff;
    let isb32 = (inv_sbox_packed[((s1 >> 8) & 0xff) as usize >> 2]
        >> (8 * (3 - (((s1 >> 8) & 0xff) & 3))))
        & 0xff;
    let isb33 =
        (inv_sbox_packed[(s0 & 0xff) as usize >> 2] >> (8 * (3 - ((s0 & 0xff) & 3)))) & 0xff;

    let t0 = (isb00 << 24) | (isb01 << 16) | (isb02 << 8) | isb03;
    let t1 = (isb10 << 24) | (isb11 << 16) | (isb12 << 8) | isb13;
    let t2 = (isb20 << 24) | (isb21 << 16) | (isb22 << 8) | isb23;
    let t3 = (isb30 << 24) | (isb31 << 16) | (isb32 << 8) | isb33;

    let mut c = chunk_mut(
        output,
        reshape_map!([4] | [num_blocks] => layout: [i0, t0]),
    );
    c[0] = t0 ^ inv_round_keys[0];
    c[1] = t1 ^ inv_round_keys[1];
    c[2] = t2 ^ inv_round_keys[2];
    c[3] = t3 ^ inv_round_keys[3];
}

// ============================================================
// Helper: pack bytes to big-endian u32 for GPU input
// ============================================================
pub fn bytes_to_u32_be(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks(4)
        .map(|c| ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) | (c[3] as u32))
        .collect()
}

pub fn u32_be_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_be_bytes()).collect()
}

/// Build concatenated T-tables for encrypt: [TE0 | TE1 | TE2 | TE3]
pub fn build_te_tables() -> Vec<u32> {
    let mut tables = Vec::with_capacity(1024);
    tables.extend_from_slice(&aes_common::TE0);
    tables.extend_from_slice(&aes_common::TE1);
    tables.extend_from_slice(&aes_common::TE2);
    tables.extend_from_slice(&aes_common::TE3);
    tables
}

/// Build concatenated T-tables for decrypt: [TD0 | TD1 | TD2 | TD3]
pub fn build_td_tables() -> Vec<u32> {
    let mut tables = Vec::with_capacity(1024);
    tables.extend_from_slice(&aes_common::TD0);
    tables.extend_from_slice(&aes_common::TD1);
    tables.extend_from_slice(&aes_common::TD2);
    tables.extend_from_slice(&aes_common::TD3);
    tables
}

/// Build concatenated T-tables for decrypt: [TD0 | TD1 | TD2 | TD3 | INV_SBOX_PACKED]
/// inv_sbox is packed as 64 u32 values (256 bytes / 4)
pub fn build_td_tables_with_inv_sbox() -> Vec<u32> {
    let mut tables = Vec::with_capacity(1088);
    tables.extend_from_slice(&aes_common::TD0);
    tables.extend_from_slice(&aes_common::TD1);
    tables.extend_from_slice(&aes_common::TD2);
    tables.extend_from_slice(&aes_common::TD3);
    // Pack inv_sbox as big-endian u32
    for chunk in aes_common::INV_SBOX.chunks(4) {
        let word = ((chunk[0] as u32) << 24)
            | ((chunk[1] as u32) << 16)
            | ((chunk[2] as u32) << 8)
            | (chunk[3] as u32);
        tables.push(word);
    }
    tables
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_aes_encrypt_nist_ttable() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];
        let expected_ct: [u8; 16] = [
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a,
            0x0b, 0x32,
        ];

        let round_keys = aes_common::key_expansion(&key);
        let input_u32 = bytes_to_u32_be(&plaintext);
        let te_tables = build_te_tables();

        gpu_host::cuda_ctx(0, |ctx, m| {
            let block_size: u32 = 256;
            let num_blocks: u32 = 1;
            let grid_size: u32 = 1;
            // Shared memory: 1024 u32 = 4096 bytes
            let shared_bytes: u32 = 1024 * 4;

            let config =
                gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_bytes);

            let d_input = ctx.new_tensor_view(input_u32.as_slice()).unwrap();
            let d_rk = ctx.new_tensor_view(round_keys.as_slice()).unwrap();
            let d_te = ctx.new_tensor_view(te_tables.as_slice()).unwrap();
            let mut d_output = ctx.new_tensor_view(&vec![0u32; 4] as &[u32]).unwrap();

            aes128_encrypt_ttable_kernel::launch(
                config,
                ctx,
                m,
                &d_input,
                &mut d_output,
                &d_rk,
                &d_te,
                num_blocks,
            )
            .unwrap();

            let mut result_u32 = vec![0u32; 4];
            d_output.copy_to_host(&mut result_u32).unwrap();
            let result_bytes = u32_be_to_bytes(&result_u32);
            assert_eq!(
                result_bytes.as_slice(),
                &expected_ct,
                "NIST encrypt mismatch"
            );
        });
    }

    #[test]
    fn test_aes_encrypt_nist_ttable() {
        run_aes_encrypt_nist_ttable();
    }

    fn run_aes_roundtrip_ttable() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        // Test with multiple blocks
        let num_aes_blocks: u32 = 64;
        let plaintext: Vec<u8> = (0..num_aes_blocks * 16).map(|i| (i % 256) as u8).collect();

        let enc_keys = aes_common::key_expansion(&key);
        let dec_keys = aes_common::inv_round_keys(&enc_keys);
        let input_u32 = bytes_to_u32_be(&plaintext);
        let te_tables = build_te_tables();
        let td_tables = build_td_tables();
        let inv_sbox_packed: Vec<u32> = aes_common::INV_SBOX
            .chunks(4)
            .map(|c| {
                ((c[0] as u32) << 24)
                    | ((c[1] as u32) << 16)
                    | ((c[2] as u32) << 8)
                    | (c[3] as u32)
            })
            .collect();

        gpu_host::cuda_ctx(0, |ctx, m| {
            let block_size: u32 = 256;
            let grid_size: u32 = (num_aes_blocks + block_size - 1) / block_size;
            let shared_enc: u32 = 1024 * 4; // encrypt: 1024 u32
            let shared_dec: u32 = 0; // decrypt: no shared memory (global T-tables)
            let n_words = (num_aes_blocks * 4) as usize;

            // Encrypt
            let d_input = ctx.new_tensor_view(input_u32.as_slice()).unwrap();
            let d_enc_rk = ctx.new_tensor_view(enc_keys.as_slice()).unwrap();
            let d_te = ctx.new_tensor_view(te_tables.as_slice()).unwrap();
            let mut d_encrypted = ctx.new_tensor_view(&vec![0u32; n_words] as &[u32]).unwrap();

            let config = gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_enc);
            aes128_encrypt_ttable_kernel::launch(
                config,
                ctx,
                m,
                &d_input,
                &mut d_encrypted,
                &d_enc_rk,
                &d_te,
                num_aes_blocks,
            )
            .unwrap();

            // Decrypt
            let d_dec_rk = ctx.new_tensor_view(dec_keys.as_slice()).unwrap();
            let d_td = ctx.new_tensor_view(td_tables.as_slice()).unwrap();
            let d_inv_sbox = ctx.new_tensor_view(inv_sbox_packed.as_slice()).unwrap();
            let mut d_decrypted = ctx.new_tensor_view(&vec![0u32; n_words] as &[u32]).unwrap();

            let config2 =
                gpu_host::gpu_config!(grid_size, 1, 1, block_size, 1, 1, shared_dec);
            aes128_decrypt_ttable_kernel::launch(
                config2,
                ctx,
                m,
                &d_encrypted,
                &mut d_decrypted,
                &d_dec_rk,
                &d_td,
                &d_inv_sbox,
                num_aes_blocks,
            )
            .unwrap();

            let mut result_u32 = vec![0u32; n_words];
            d_decrypted.copy_to_host(&mut result_u32).unwrap();
            let result_bytes = u32_be_to_bytes(&result_u32);
            assert_eq!(
                result_bytes.as_slice(),
                plaintext.as_slice(),
                "Roundtrip mismatch"
            );
        });
    }

    #[test]
    fn test_aes_roundtrip_ttable() {
        run_aes_roundtrip_ttable();
    }
}
