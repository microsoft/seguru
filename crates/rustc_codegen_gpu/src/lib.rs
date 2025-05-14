#![feature(rustc_private)]
#![allow(unused_variables)]
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_llvm;
extern crate rustc_codegen_ssa;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

mod attr;
mod backend;
mod builder;
mod codegen;
mod context;
mod mlir;
mod write;

#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn rustc_codegen_ssa::traits::CodegenBackend> {
    env_logger::init();
    Box::new(backend::GPUCodegenBackend::new())
}
