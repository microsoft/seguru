#![feature(rustc_private)]
#![allow(unused_variables)]
#![feature(box_patterns)]
#![allow(clippy::cast_possible_truncation)]
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_mir_dataflow;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_type_ir;

mod attr;
mod backend;
mod builder;
mod codegen;
mod context;
mod error;
mod mir_analysis;
mod mir_mut_arg_check;
mod mir_thread_sync_check;
mod mlir;
mod scope;
mod write;

#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn rustc_codegen_ssa_gpu::traits::CodegenBackend> {
    //let _ = tracing_subscriber::fmt::try_init();
    Box::new(backend::GPUCodegenBackend::new())
}
