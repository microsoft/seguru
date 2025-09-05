#![feature(negative_impls)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![no_std]
#![feature(asm_experimental_arch)]
#![feature(core_intrinsics)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;
use core::arch::asm;

pub mod cg;
mod chunk;
mod chunk_impl;
mod device_intrinsic;
mod dim;
pub mod iter;
mod print;
mod shared;
mod thread;

pub use chunk::GlobalThreadChunk;
pub use chunk_impl::{Map2D, MapLinear, MapLinearWithDim, chunk_mut};
#[cfg(not(feature = "codegen_tests"))]
pub use cuda_bindings::GPUConfig;
pub use device_intrinsic::GPUDeviceFloatIntrinsics;
#[cfg(not(feature = "codegen_tests"))]
pub use dim::assume_dim_with_config;
pub use dim::{
    DimType, DimX, DimY, DimZ, GpuChunkIdx, GpuSharedChunkIdx, block_dim, block_id,
    block_thread_ids, dim, global_id, grid_dim, thread_id,
};
pub use print::{PushPrintfArg, printf};
pub use shared::{DynamicSharedAlloc, GpuShared};
pub use thread::sync_threads;

/// Add an extra assertion before indexing operation.
/// This is used to ensure that some indexing operation is safe,
/// allowing us to optimize the assertion using select.
/// If not followed by an indexing operation, it will be ignored.
#[inline(never)]
#[rustc_diagnostic_item = "gpu::build_sfi"]
#[gpu_codegen::device]
pub(crate) fn assert_before_index(_cond: bool, _idx: usize) {
    unimplemented!()
}

/// Add a string attribute to the MLIR module.
#[rustc_diagnostic_item = "gpu::add_mlir_string_attr"]
#[gpu_codegen::device]
#[inline(never)]
pub const fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

/// This is safe to use.
/// Not part of TCB, but defined in pair with subslice_mut.
/// Defined to make gpu_macros::kernel work fluently.
#[inline(never)]
#[rustc_diagnostic_item = "gpu::subslice"]
#[gpu_codegen::device]
#[allow(dead_code)]
pub(crate) fn subslice<T>(_original: &[T], _offset: usize, _window: usize) -> &[T] {
    unimplemented!()
}

/// # Safety
/// This function is unsafe because it assumes that the [_offset, _offset+ _window) is unique per GPU thread.
/// If it is not unique per GPU thread, it can cause racing on the same memory location.
#[inline(never)]
#[rustc_diagnostic_item = "gpu::subslice_mut"]
#[gpu_codegen::device]
#[gpu_codegen::sync_data(0, 2)]
pub(crate) unsafe fn subslice_mut<T>(
    _original: &mut [T],
    _offset: usize,
    _window: usize,
) -> &mut [T] {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.atomic_add)]
#[rustc_diagnostic_item = "gpu::atomic_add"]
pub fn atomic_add<T>(_slice: &mut T, _val: T) -> T {
    unimplemented!()
}

#[repr(C)]
pub struct float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[gpu_codegen::device]
#[inline(always)]
pub fn __ldcs_f32(val: &f32) -> f32 {
    let mut ret: f32;
    let ptr = val as *const f32;
    unsafe {
        asm!(
            "ld.global.cs.f32 {0:e}, [{1:r}];",
            out(reg) ret,
            in(reg) ptr,
        );
    }
    ret
}

/*  TODO: Define shared memory */
