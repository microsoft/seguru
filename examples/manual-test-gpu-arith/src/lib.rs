#![no_std]
#![allow(clippy::too_many_arguments)]

#[allow(non_upper_case_globals)]
#[gpu_macros::shared_size]
pub static shared_size_kernel_arith: usize = 0;

#[gpu_macros::kernel]
#[no_mangle]
pub fn kernel_arith(
    a: &gpu::GpuChunkable<u32>,
    b: &gpu::GpuChunkableMut<u32>,
    c: &[u32],
    f: &gpu::GpuChunkableMut<f32>,
    g: &[f32],
) {
    let thread_id = gpu::thread_id(gpu::DimType::X);
    b[0] = a[0] + c[thread_id];
    gpu::atomic_add::<u32>(&mut b[0], 1);
    let mut out: u32;
    let in32: u32 = b[0] + 30;
    unsafe {
        core::arch::asm!(
        "mov.u32 {0:e}, {1:e};",
        out(reg) out,
        in(reg) in32,
        );
    }
    b[0] = out;
    f[0] = gpu::__ldcs_f32(&g[thread_id]);
}
