#![no_std]

#[allow(non_upper_case_globals)]
#[gpu_macros::shared_size]
pub static shared_size_kernel_arith: usize = 0;

#[gpu_macros::kernel]
#[no_mangle]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    b[0] = a[0];
    gpu::atomic_add::<u8>(&mut b[0], 1);
}
