#![no_std]
#![allow(clippy::too_many_arguments)]
#![deny(clippy::cast_possible_truncation)]
#![feature(stmt_expr_attributes)]

use gpu::cg::CGOperations;
use gpu::{thread_id, CacheStreamLoadStore, DimX, GPUDeviceFloatIntrinsics};

pub enum TestEnum {
    A,
    B,
}

impl TestEnum {
    #[gpu::device]
    #[inline]
    pub fn test_enum_closure(&self, el: [u64; 4]) -> [u8; 32] {
        let mut res = [0u8; 32];
        match self {
            TestEnum::A => {
                el.iter().enumerate().for_each(|(i, limb)| {
                    let off = i * 8;
                    res[off..off + 8].copy_from_slice(&limb.to_le_bytes());
                });
            }
            TestEnum::B => {
                el.iter().enumerate().for_each(|(i, limb)| {
                    let off = i * 8;
                    res[off..off + 8].copy_from_slice(&limb.to_be_bytes());
                });
            }
        }
        res
    }
}

/// # Safety
/// This kernel might be unsafe because it uses Chunkable::new that is not defined as trusted chunking func.
#[gpu::kernel]
pub fn kernel_arith<const N: u32>(
    a: &[u32],
    b: &mut [u32],
    c: &[u32],
    f: &mut [f32],
    f_width: usize,
    g: &[f32],
    h: &mut f32, // sum of g
) {
    let thread_id = gpu::thread_id::<gpu::DimY>();
    let a_local = a[thread_id as usize];
    let mut b = gpu::chunk_mut(b, gpu::Map2D::new(f_width));
    let b_local = &mut b[(0, 0)];
    *b_local = a_local + c[thread_id as usize];
    *b_local += 1;

    let mut out: u32;
    let in32: u32 = *b_local + N;
    unsafe {
        gpu::asm!(
        "mov.u32 {0:reg32}, {1:reg32};",
        out(reg) out,
        in(reg) in32,
        );
    }
    *b_local = out;

    let f_chunk_param: gpu::MapLinearWithDim = gpu::MapLinearWithDim::new(f_width);
    let mut f = gpu::chunk_mut(f, f_chunk_param);
    let g_local = g[thread_id as usize].ldcs();
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
    atomic_h.atomic_addf(g[thread_id as usize]);
    let l = 1u64.to_le_bytes();
    let (l0, l7) = (l[0], l[7]);
    assert!(l0 == 1 && l7 == 0);
    let x = TestEnum::A.test_enum_closure([1, 2, 3, 4]);
    assert!(x[0] == 1 && x[8] == 2 && x[16] == 3 && x[24] == 4);
    /*let warp = gpu::cg::ThreadWarpTile::<32, 1>();
    warp.run_on_lane_0::<f32>(f, |v| {
        *v += 1.5;
    });*/
}

#[gpu::cuda_kernel]
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
#[gpu::cuda_kernel]
pub fn oob_no_fails(a: f32, b: &mut [f32], _width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::MapLinear::new(1));
    if thread_id::<DimX>() == 0 {
        b[1] = a;
    }
}

#[gpu::cuda_kernel]
pub fn oob2(a: f32, b: &mut [f32], width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::Map2D::new(width));
    if thread_id::<DimX>() == 0 {
        b[(4, 1)] = a;
    }
}

#[gpu::cuda_kernel]
pub fn oob3(a: f32, b: &mut [f32], width: usize) {
    let mut b = gpu::chunk_mut(b, gpu::MapLinear::new(width));
    if thread_id::<DimX>() == 1 {
        b[0] = a;
    }
}
