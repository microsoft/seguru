#![cfg_attr(not(verus_keep_ghost), feature(negative_impls))]
#![cfg_attr(not(verus_keep_ghost), feature(register_tool))]
#![register_tool(gpu_codegen)]
#![allow(internal_features)]
#![cfg_attr(not(verus_keep_ghost), feature(rustc_attrs))]
#![no_std]
#![feature(asm_experimental_arch)]
#![feature(core_intrinsics)]

extern crate alloc;

#[cfg(feature = "codegen_tests")]
extern crate gpu_macros;

#[cfg(feature = "codegen_tests")]
extern crate num_traits;

pub mod arch;
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
pub mod prelude;
mod print;
mod shared;
pub mod sync;
pub mod tile;
pub mod vector;

pub use prelude::*;

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
