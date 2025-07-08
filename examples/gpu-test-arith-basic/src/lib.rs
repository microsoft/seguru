#![no_std]

/// # Safety
/// This kernel might be unsafe because it accesses mutable global memory
/// without using our chunking functions.
#[no_mangle]
#[gpu_macros::kernel_v2]
pub unsafe fn kernel_arith(a: &[u8], b: &mut [u8]) {
    let c = gpu::thread_id(gpu::DimType::X);
    b[c] = a[c];
}
