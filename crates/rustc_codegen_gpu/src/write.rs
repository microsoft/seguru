use melior::ir::operation::{OperationLike, OperationMutLike, OperationRefMut};
use rustc_codegen_ssa_gpu::back::write::{CodegenContext, ModuleConfig};
use rustc_codegen_ssa_gpu::{CompiledModule, ModuleCodegen};
use rustc_errors::DiagCtxtHandle;
use tracing::{debug, trace};

use crate::backend::{GPUCodeGenModule, GPUCodegenBackend};

pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GPUCodeGenModule>,
    _config: &ModuleConfig,
) -> Result<rustc_codegen_ssa_gpu::CompiledModule, rustc_errors::FatalError> {
    let mod_name = module.name.clone();
    let module_name = Some(&mod_name[..]);
    let out = if let Some(mut m) = module.module_llvm.mlir_module {
        let out =
            cgcx.output_filenames.temp_path(rustc_session::config::OutputType::Mir, module_name);
        let out_object =
            cgcx.output_filenames.temp_path(rustc_session::config::OutputType::Object, module_name);
        debug!("write MLIR module to {:?}", out);
        let mut op = m.module.as_operation_mut();
        crate::mlir::visit::visit_ops_recursively(&mut op, &|op: &mut OperationRefMut<'_, '_>| {
            if op.attribute("to_remove").is_ok() {
                op.remove_from_parent();
            }
        });
        let content = op.to_string();
        if !m.module.as_operation().verify() {
            trace!("MLIR module verify failed.");
            std::fs::write(&out, &content).unwrap();
            Err(rustc_errors::FatalError)?;
        }
        let content = content.replace("attributes {kernel, ", "kernel attributes {");

        std::fs::write(&out, &content).unwrap();
        debug!("[Done]write MLIR module to {:?}", out);
        // mlir-opt must use "shell" in order to pass correct arguments.
        mlir_compile::CompileConfig::default()
            .mlir_compile(&out, &out_object)
            .expect("Failed to compile MLIR to object file");
        trace!("copy MLIR obj to {:?}", out_object);
        Some(out_object)
    } else {
        None
    };
    Ok(CompiledModule {
        name: mod_name,
        kind: module.kind,
        object: out,
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: vec![],
    })
}
