#![no_std]
#![allow(clippy::too_many_arguments)]
#![feature(stmt_expr_attributes)]

use gpu::{GpuChunkIdx, GpuSharedChunkIdx};

#[allow(non_upper_case_globals)]
#[gpu_macros::shared_size]
pub static shared_size_kernel_arith: usize = 0;

/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[gpu_macros::kernel]
#[no_mangle]
pub fn kernel_arith(
    a: gpu::GpuChunkable2D<u32>,
    b: gpu::GpuChunkableMut2D<u32>,
    c: &[u32],
    f: gpu::GpuChunkableMut<f32>,
    g: &[f32],
) {
    let thread_id = gpu::thread_id(gpu::DimType::Y);
    let a_local = gpu::get_local_2d::<u32>(&a, 0, 0);
    let mut b = b;
    let b_local = gpu::get_local_mut_2d::<u32>(&mut b, 0, 0);
    *b_local = *a_local + c[thread_id];
    gpu::atomic_add::<u32>(b_local, 1);
    let mut out: u32;
    let in32: u32 = *b_local + 30;
    unsafe {
        core::arch::asm!(
        "mov.u32 {0:e}, {1:e};",
        out(reg) out,
        in(reg) in32,
        );
    }
    *b_local = out;

    let mut f = f;
    let f = gpu::get_mut_chunk(&mut f, GpuChunkIdx::new());
    let g_local = gpu::__ldcs_f32(&g[thread_id]);
    f[0] = g_local.sin();

    let mut shared = gpu::GpuShared::<[f32; 32]>::zero();
    let shared_chunk = shared.chunk_mut(1, GpuSharedChunkIdx::new());
    shared_chunk[0] = 1.1 * ((thread_id + 1) as f32);
    gpu::sync_threads(); // TODO: MIR analyzer to check/inject the correct sync.
    let mut i = 0;
    while i < 32 {
        f[0] += shared[i];
        i += 1;
    }

    let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    warp.run_on_thread_0::<f32>(f, |v| {
        *v += 1.5;
    });
}

/*
#[gpu_macros::kernel_v2]
#[no_mangle]
pub fn kernel_arith2(
    a: gpu::GpuChunkable2D<u32>,
    b: gpu::GpuChunkableMut2D<u32>,
    c: &[u32],
    f: gpu::GpuChunkableMut<f32>,
    g: &[f32],
) {
    /*let mut shared = gpu::GpuShared::<[f32; 32]>::zero();
    let thread_id = gpu::thread_id(gpu::DimType::Y);
    let a_local = gpu::get_local_2d::<u32>(&a, 0, 0);
    let mut b = b;
    let b_local = gpu::get_local_mut_2d::<u32>(&mut b, 0, 0);
    *b_local = *a_local + c[thread_id];
    gpu::atomic_add::<u32>(b_local, 1);
    let mut out: u32;
    let in32: u32 = *b_local + 30;
    unsafe {
        core::arch::asm!(
        "mov.u32 {0:e}, {1:e};",
        out(reg) out,
        in(reg) in32,
        );
    }
    *b_local = out;*/
    let thread_id = gpu::thread_id(gpu::DimType::Y);
    let mut shared = gpu::GpuShared::<[f32; 32]>::zero();
    let mut f = f;
    let f = gpu::get_mut_chunk(&mut f, GpuChunkIdx::new());
    f[0] = gpu::__ldcs_f32(&g[thread_id]);
    let shared_chunk = shared.chunk_mut(2, GpuChunkIdx::new());
    shared_chunk[0] = 1.23;
    gpu::sync_threads();
    let mut i = 0;
    while i < 2 {
        f[0] += shared.deref_call()[i];
        i += 1;
    }
}
*/
