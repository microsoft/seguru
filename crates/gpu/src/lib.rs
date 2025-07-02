#![feature(negative_impls)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

mod dim;
mod print;
mod thread;

pub use dim::{DimType, GpuChunkIdx, block_dim, global_id, grid_dim, thread_id};
pub use print::{PushPrintfArg, printf};
pub use thread::{GpuChunksMut, scope};

pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// Add a string attribute to the MLIR module.
#[gpu_codegen::builtin(add_mlir_string_attr)]
#[gpu_codegen::device]
#[inline(never)]
pub fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice)]
pub fn subslice<T>(_original: &[T], _offset: usize, _window: usize) -> &[T] {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice_mut)]
pub fn subslice_mut<T>(_original: &mut [T], _offset: usize, _window: usize) -> &mut [T] {
    unimplemented!()
}

/*  TODO: Define shared memory */
