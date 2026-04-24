unsafe extern "C" {
    pub fn bench_aes128_encrypt_ttable(
        host_input: *const u8,
        host_output: *mut u8,
        host_round_keys: *const u32,
        num_blocks: u32,
        warmup: i32,
        iters: i32,
    ) -> f32;

    pub fn bench_aes128_decrypt_ttable(
        host_input: *const u8,
        host_output: *mut u8,
        host_round_keys: *const u32,
        num_blocks: u32,
        warmup: i32,
        iters: i32,
    ) -> f32;

    pub fn bench_aes128_encrypt_textbook(
        host_input: *const u8,
        host_output: *mut u8,
        host_round_keys: *const u32,
        num_blocks: u32,
        warmup: i32,
        iters: i32,
    ) -> f32;

    pub fn bench_aes128_decrypt_textbook(
        host_input: *const u8,
        host_output: *mut u8,
        host_round_keys: *const u32,
        num_blocks: u32,
        warmup: i32,
        iters: i32,
    ) -> f32;
}
