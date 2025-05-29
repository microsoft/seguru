#![feature(rustc_private)]
#![allow(unused_variables)]
#![feature(box_patterns)]
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_llvm;
extern crate rustc_codegen_ssa_gpu;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;
extern crate rustc_type_ir;

mod attr;
mod backend;
mod builder;
mod codegen;
mod context;
mod mlir;
mod write;

#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn rustc_codegen_ssa_gpu::traits::CodegenBackend> {
    use std::io::Write;
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            writeln!(
                buf,
                "[{}:{}] {} - {}",
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.level(),
                record.args()
            )
        })
        .init();
    Box::new(backend::GPUCodegenBackend::new())
}
