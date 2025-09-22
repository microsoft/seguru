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

pub mod cg;
pub mod chunk;
mod chunk_impl;
pub mod chunk_scope;
mod device_intrinsic;
mod dim;
pub(crate) mod global;
mod host_dev;
pub mod iter;
mod ldst;
mod print;
mod shared;
pub mod sync;

pub use chunk::{GlobalGroupChunk, GlobalThreadChunk, chunk_mut};
pub use chunk_impl::{Map2D, MapLinear, MapLinearWithDim, MapReshape};
#[cfg(not(feature = "codegen_tests"))]
pub use cuda_bindings::SafeGpuConfig;
pub use device_intrinsic::GPUDeviceFloatIntrinsics;
#[cfg(not(feature = "codegen_tests"))]
pub use dim::assume_dim_with_config;
pub use dim::{
    DimType, DimX, DimY, DimZ, block_dim, block_id, dim, global_id, grid_dim, thread_id,
};
pub use global::GpuGlobal;
pub use host_dev::HostToDev;
pub use ldst::CacheStreamLoadStore;
pub use print::{PushPrintfArg, printf};
pub use shared::{DynamicSharedAlloc, DynamicSharedAllocBuilder, GpuShared};
pub use sync::sync_threads;

/// Add an extra assertion before indexing operation.
/// This is used to ensure that some indexing operation is safe,
/// allowing us to optimize the assertion using select.
/// If not followed by an indexing operation, it will be ignored.
#[inline(never)]
#[rustc_diagnostic_item = "gpu::build_sfi"]
#[gpu_codegen::device]
pub(crate) fn assert_ptr<T>(_cond: bool, _ptr: T) -> T {
    unimplemented!()
}

/// Add a string attribute to the MLIR module.
#[rustc_diagnostic_item = "gpu::add_mlir_string_attr"]
#[gpu_codegen::device]
#[inline(never)]
pub const fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
