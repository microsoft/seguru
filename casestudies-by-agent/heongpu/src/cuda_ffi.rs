unsafe extern "C" {
    pub fn cuda_bench_addition(
        d_in1: *const u64,
        d_in2: *const u64,
        d_out: *mut u64,
        d_mod_values: *const u64,
        total_elements: u32,
        n_power: u32,
        rns_count: u32,
        block_size: u32,
        iters: i32,
    ) -> f64;

    pub fn cuda_bench_multiply(
        d_in1: *const u64,
        d_in2: *const u64,
        d_out: *mut u64,
        d_mod_values: *const u64,
        d_mod_bits: *const u64,
        d_mod_mus: *const u64,
        total_elements: u32,
        n_power: u32,
        rns_count: u32,
        block_size: u32,
        iters: i32,
    ) -> f64;

    pub fn cuda_bench_sk_multiply(
        d_ct1: *const u64,
        d_sk: *const u64,
        d_out: *mut u64,
        d_mod_values: *const u64,
        d_mod_bits: *const u64,
        d_mod_mus: *const u64,
        total_elements: u32,
        n_power: u32,
        rns_count: u32,
        block_size: u32,
        iters: i32,
    ) -> f64;

    pub fn cuda_bench_cipher_plain_mul(
        d_cipher: *const u64,
        d_plain: *const u64,
        d_out: *mut u64,
        d_mod_values: *const u64,
        d_mod_bits: *const u64,
        d_mod_mus: *const u64,
        total_elements: u32,
        n_power: u32,
        rns_count: u32,
        block_size: u32,
        iters: i32,
    ) -> f64;

    pub fn cuda_malloc(ptr: *mut *mut u8, size: usize);
    pub fn cuda_free(ptr: *mut u8);
    pub fn cuda_memcpy_h2d(dst: *mut u8, src: *const u8, size: usize);
    #[allow(dead_code)]
    pub fn cuda_memcpy_d2h(dst: *mut u8, src: *const u8, size: usize);
    #[allow(dead_code)]
    pub fn cuda_device_sync();
}
