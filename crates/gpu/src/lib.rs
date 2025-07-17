#![feature(negative_impls)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![no_std]
#![feature(asm_experimental_arch)]

use core::arch::asm;

pub mod cg;
mod dim;
mod print;
mod thread;

#[cfg(not(feature = "codegen_tests"))]
pub use cuda_bindings::{GpuChunkable, GpuChunkable2D, GpuChunkableMut, GpuChunkableMut2D};

#[inline(never)]
#[gpu_codegen::builtin(gpu.build_sfi)]
#[rustc_diagnostic_item = "gpu::build_sfi"]
pub fn build_sfi(_size: usize, _offset: usize) {
    unimplemented!()
}

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_mut_2d"]
#[gpu_codegen::device]
#[inline(always)]
pub fn get_local_mut_2d<'a, T>(
    a: &'a mut GpuChunkableMut2D<'a, T>,
    x: usize,
    y: usize,
) -> &'a mut T {
    // Must check if col is smaller than a.size_x
    let row = y * grid_dim(DimType::Y) * block_dim(DimType::Y)
        + block_dim(DimType::Y) * block_id(DimType::Y)
        + thread_id(DimType::Y);
    let col = x * grid_dim(DimType::X) * block_dim(DimType::X)
        + block_dim(DimType::X) * block_id(DimType::X)
        + thread_id(DimType::X);
    build_sfi(a.size_x(), col);

    // Here Rust will automatic generate an SFI
    unsafe { &mut (&mut *(a.as_ptr() as *mut [T]))[a.size_x() * row + col] }
}

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_2d"]
#[gpu_codegen::device]
#[inline(always)]
pub fn get_local_2d<'a, T>(a: &'a GpuChunkable2D<'a, T>, x: usize, y: usize) -> &'a T {
    // Must check if col is smaller than a.size_x
    let row = y * grid_dim(DimType::Y) * block_dim(DimType::Y)
        + block_dim(DimType::Y) * block_id(DimType::Y)
        + thread_id(DimType::Y);
    let col = x * grid_dim(DimType::X) * block_dim(DimType::X)
        + block_dim(DimType::X) * block_id(DimType::X)
        + thread_id(DimType::X);
    build_sfi(a.size_x(), col);

    // Here Rust will automatic generate an SFI
    unsafe { &(&*a.as_ptr())[a.size_x() * row + col] }
}

pub use dim::{DimType, GpuChunkIdx, block_dim, block_id, global_id, grid_dim, thread_id};
pub use print::{PushPrintfArg, printf};
pub use thread::{GpuChunksMut, GpuShared, chunk_mut, scope, sync_threads};

/// Add a string attribute to the MLIR module.
#[rustc_diagnostic_item = "gpu::add_mlir_string_attr"]
#[gpu_codegen::device]
#[inline(never)]
pub fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice)]
#[rustc_diagnostic_item = "gpu::subslice"]
pub fn subslice<T>(_original: &[T], _offset: usize, _window: usize) -> &[T] {
    unimplemented!()
}

#[inline(never)]
#[rustc_diagnostic_item = "gpu::subslice_mut"]
pub fn subslice_mut<T>(_original: &mut [T], _offset: usize, _window: usize) -> &mut [T] {
    unimplemented!()
}

#[cfg(not(feature = "codegen_tests"))]
#[inline(never)]
#[rustc_diagnostic_item = "gpu::get_chunk"]
#[gpu_codegen::device]
pub fn get_mut_chunk<'a, T>(
    chunkable: &mut GpuChunkableMut<'a, T>,
    idx_pattern: GpuChunkIdx,
) -> &'a mut [T] {
    let w = chunkable.window();
    let end_idx = w + idx_pattern.as_usize() * w;
    unsafe {
        // Here Rust will automatic generate an SFI
        let end = &(&*chunkable.as_ptr())[end_idx] as *const T as *mut T;
        core::slice::from_raw_parts_mut(end.sub(w), w)
    }
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
