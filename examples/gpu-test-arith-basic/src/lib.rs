#![no_std]

#[no_mangle]
#[gpu_macros::kernel_v2]
pub fn kernel_arith(a: &[u8], b: &mut [u8]) {
    let c = gpu::thread_id(gpu::DimType::X);
    b[c] = a[c];
}
