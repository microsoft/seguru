#![no_std]

#[allow(non_upper_case_globals)]
#[gpu_macros::shared_size]
pub static shared_size_kernel_arith: usize = 0;

#[gpu_macros::kernel]
#[no_mangle]
pub fn kernel_arith(a: &gpu::GpuChunkable<u8>, b: &gpu::GpuChunkableMut<u8>, c: &[u8]) {
    let thread_id = gpu::thread_id(gpu::DimType::X);
    b[0] = a[0] + c[thread_id];
    gpu::atomic_add::<u8>(&mut b[0], 1);
    let mut out: u32;
    let in32: u32 = b[0] as u32 + 30;
    unsafe {
        core::arch::asm!(
        "mov.u32 {0:e}, {1:e};",
        out(reg) out,
        in(reg) in32,
        );
    }
    b[0] = out as u8;
}
