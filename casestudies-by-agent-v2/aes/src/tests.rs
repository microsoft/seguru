use super::*;
use gpu_host::gpu_config;

const KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

/// Encrypt then decrypt `data` on the GPU and return both results.
fn roundtrip_gpu(key: &[u8; 16], data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n_bytes = data.len();
    let n_blocks = n_bytes / 16;
    let enc_rk = tables::key_expansion(key);
    let dec_rk = tables::inv_round_keys(&enc_rk);
    let h_in = bytes_to_blocks(data);
    let padded = h_in.len();
    let grid = grid_dim_for(n_blocks);

    gpu_host::cuda_ctx(0, |ctx, m| {
        let d_in = ctx.new_tensor_view::<[U32_4]>(&h_in).unwrap();
        let enc_rk_staged = staged_round_keys(&enc_rk);
        let dec_rk_staged = staged_round_keys(&dec_rk);
        let d_enc_rk = ctx.new_tensor_view(enc_rk_staged.as_slice()).unwrap();
        let d_dec_rk = ctx.new_tensor_view(dec_rk_staged.as_slice()).unwrap();
        let d_te0 = ctx.new_tensor_view(tables::TE0.as_slice()).unwrap();
        let d_td0 = ctx.new_tensor_view(tables::TD0.as_slice()).unwrap();
        let isb = inv_sbox_u32();
        let d_isb = ctx.new_tensor_view(isb.as_slice()).unwrap();

        let zeros = vec![U32_4::default(); padded];
        let mut d_ct = ctx.new_tensor_view::<[U32_4]>(&zeros).unwrap();
        let mut d_pt = ctx.new_tensor_view::<[U32_4]>(&zeros).unwrap();

        let cfg = gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
        aes128_encrypt::launch(cfg, ctx, m, &d_in, &mut d_ct, &d_enc_rk, &d_te0).unwrap();

        let cfg = gpu_config!(grid, 1, 1, @const BLOCK_DIM, 1, 1, 0);
        aes128_decrypt::launch(cfg, ctx, m, &d_ct, &mut d_pt, &d_dec_rk, &d_td0, &d_isb).unwrap();

        let mut h_ct = vec![U32_4::default(); padded];
        let mut h_pt = vec![U32_4::default(); padded];
        d_ct.copy_to_host(&mut h_ct).unwrap();
        d_pt.copy_to_host(&mut h_pt).unwrap();
        (blocks_to_bytes(&h_ct, n_bytes), blocks_to_bytes(&h_pt, n_bytes))
    })
}

#[test]
fn tables_match_known_values() {
    // Spot-check the compile-time generated tables against the AES standard.
    assert_eq!(tables::SBOX[0x00], 0x63);
    assert_eq!(tables::SBOX[0x53], 0xed);
    assert_eq!(tables::SBOX[0xff], 0x16);
    assert_eq!(tables::INV_SBOX[tables::SBOX[0x9a] as usize], 0x9a);
    // TE0[x] = [2*S(x), S(x), S(x), 3*S(x)]; S(1) = 0x7c, 2*0x7c = 0xf8, 3*0x7c = 0x84.
    assert_eq!(tables::TE0[0x01], 0xf87c_7c84);
    // TD0[x] = [14*IS(x), 9*IS(x), 13*IS(x), 11*IS(x)]; IS(0) = 0x52.
    assert_eq!(tables::TD0[0x00], 0x51f4_a750);
}

#[test]
fn key_expansion_matches_fips197() {
    let rk = tables::key_expansion(&KEY);
    assert_eq!(rk[0], 0x2b7e_1516);
    assert_eq!(rk[3], 0x09cf_4f3c);
    assert_eq!(rk[4], 0xa0fa_fe17);
    assert_eq!(rk[43], 0xb663_0ca6);
}

#[test]
fn cpu_matches_fips197_vector() {
    let pt: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
        0x34,
    ];
    let expected: [u8; 16] = [
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b,
        0x32,
    ];
    assert_eq!(cpu::encrypt_ecb(&KEY, &pt), expected.to_vec());
    assert_eq!(cpu::decrypt_ecb(&KEY, &expected), pt.to_vec());
}

#[test]
fn gpu_matches_fips197_vector() {
    let pt: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
        0x34,
    ];
    let expected: [u8; 16] = [
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b,
        0x32,
    ];
    let (ct, rt) = roundtrip_gpu(&KEY, &pt);
    assert_eq!(ct, expected.to_vec());
    assert_eq!(rt, pt.to_vec());
}

#[test]
fn gpu_matches_cpu_across_sizes() {
    // Cover a sub-CTA size, an exact multiple of the CTA tile, and a ragged
    // size that exercises the host-side padding.
    for &n_blocks in &[1usize, 7, BLOCKS_PER_CTA as usize, 3 * BLOCKS_PER_CTA as usize + 13] {
        let data: Vec<u8> = (0..n_blocks * 16).map(|i| (i * 31 + 7) as u8).collect();
        let (ct, rt) = roundtrip_gpu(&KEY, &data);
        assert_eq!(ct, cpu::encrypt_ecb(&KEY, &data), "ciphertext mismatch at {n_blocks} blocks");
        assert_eq!(rt, data, "roundtrip mismatch at {n_blocks} blocks");
    }
}

#[test]
fn gpu_decrypt_matches_cpu_decrypt() {
    let n_blocks = BLOCKS_PER_CTA as usize + 5;
    let data: Vec<u8> = (0..n_blocks * 16).map(|i| (i % 251) as u8).collect();
    let (ct, _) = roundtrip_gpu(&KEY, &data);
    assert_eq!(cpu::decrypt_ecb(&KEY, &ct), data);
}
