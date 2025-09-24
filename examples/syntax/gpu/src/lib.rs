#![no_std]
#![allow(clippy::too_many_arguments)]
#![feature(stmt_expr_attributes)]

use gpu::cg::CGOperations;
use gpu::{thread_id, CacheStreamLoadStore, DimX, GPUDeviceFloatIntrinsics};

type ThreadChunkMatrix2D<'a> = gpu::GlobalThreadChunk<'a, u32, gpu::Map2D>;
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
    h: &mut f32, // sum of g
) {
    let thread_id = gpu::thread_id::<gpu::DimY>();
    let a_local = a[thread_id];
    let mut b = b;
    let b_local = &mut b[(0, 0)];
    *b_local = a_local + c[thread_id];
    *b_local += 1;

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
    let mut f = gpu::chunk_mut(f, f_chunk_param);
    let g_local = g[thread_id].ldcs();
    f[0] = g_local.sin();

    let mut shared = gpu::GpuShared::<[f32; 32]>::zero();
    let mut shared_chunk = shared.chunk_mut(gpu::MapLinear::new(1));
    shared_chunk[0] = 1.1 * ((thread_id + 1) as f32);
    gpu::sync_threads(); // TODO: MIR analyzer to check/inject the correct sync.
    let mut i = 0;
    while i < 32 {
        f[0] += shared[i];
        i += 1;
    }
    let warp = gpu::cg::ThreadWarpTile::<32>;
    if warp.thread_rank() == 0 {
        f[0] += 1.5;
    }

    // Use atomic to update h
    let atomic_h = gpu::sync::Atomic::new(h);
    atomic_h.atomic_addf(g[thread_id]);
    /*let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    warp.run_on_lane_0::<f32>(f, |v| {
        *v += 1.5;
    });*/
}

#[gpu_macros::kernel]
pub fn oob1(a: f32, b: &mut [f32], width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::MapLinear::new(width));
    if thread_id::<DimX>() == 0 {
        b[1] = a;
    }
}

/// This function never fails even if the code has out-of-bound access.
/// Compiler optimized all code out since it thinks accessing NULL ptr
/// has no side effect and so the code below will actually has empty function body in ptx.
/// It is still safe since no memory access is actually performed.
/// But it is silent.
/// TODO: since the code is optimized out, static analysis should be able to capture that
/// it always fails and so we can warn the user.
#[gpu_macros::kernel]
pub fn oob_no_fails(a: f32, b: &mut [f32], _width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    if thread_id::<DimX>() == 0 {
        b[1] = a;
    }
}

#[gpu_macros::kernel]
pub fn oob2(a: f32, b: &mut [f32], width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::Map2D::new(width));
    if thread_id::<DimX>() == 0 {
        b[(4, 1)] = a;
    }
}

#[gpu_macros::kernel]
pub fn oob3(a: f32, b: &mut [f32], width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::MapLinear::new(width));
    if thread_id::<DimX>() == 1 {
        b[0] = a;
    }
}
