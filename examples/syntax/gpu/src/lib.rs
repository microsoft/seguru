#![no_std]
#![allow(clippy::too_many_arguments)]
#![feature(stmt_expr_attributes)]

use gpu::GPUDeviceFloatIntrinsics;
use gpu::GpuSharedChunkIdx;

#[allow(non_upper_case_globals)]
pub static shared_size_kernel_arith: usize = 0;

type ThreadChunkMatrix2D<'a> = gpu::GlobalThreadChunk<'a, u32, 2, gpu::Map2D>;
/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[gpu_macros::kernel]
pub fn kernel_arith<const N: u32>(
    a: &[u32],
    b: ThreadChunkMatrix2D<'_>,
    c: &[u32],
    f: &mut [f32],
    f_width: usize,
    g: &[f32],
) {
    let thread_id = gpu::thread_id::<gpu::DimY>();
    let a_local = a[thread_id];
    let mut b = b;
    let b_local = &mut b[(0, 0)];
    *b_local = a_local + c[thread_id];
    gpu::sync::atomic_addi(b_local, 1);
    let mut out: u32;
    let in32: u32 = *b_local + N;
    unsafe {
        core::arch::asm!(
        "mov.u32 {0:e}, {1:e};",
        out(reg) out,
        in(reg) in32,
        );
    }
    *b_local = out;

    let f_chunk_param: gpu::MapLinearWithDim = gpu::MapLinearWithDim::new(f_width);
    let mut f = gpu::GlobalThreadChunk::new(f, f_chunk_param);
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
    if warp.lane_id() == 0 {
        f[0] += 1.5;
    }
    /*let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    warp.run_on_lane_0::<f32>(f, |v| {
        *v += 1.5;
    });*/
}

/*
#[gpu_macros::kernel_v2]
pub fn kernel_arith2(
    a: gpu::GpuChunkable2D<u32>,
    b: gpu::GpuChunkableMut2D<u32>,
    c: &[u32],
    f: gpu::GpuChunkableMut<f32>,
    g: &[f32],
) {
    /*let mut shared = gpu::GpuShared::<[f32; 32]>::zero();
    let thread_id = gpu::thread_id::<gpu::DimY>();
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
    let thread_id = gpu::thread_id::<gpu::DimY>();
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
