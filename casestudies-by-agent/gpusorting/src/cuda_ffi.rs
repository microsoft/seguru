#[cfg(feature = "bench")]
extern "C" {
    pub fn cuda_bench_sort(
        h_keys: *const u32,
        h_out: *mut u32,
        size: u32,
        warmup: u32,
        iters: u32,
    ) -> f32;
}
