use melior::ir::operation::{
    OperationLike, OperationMutLike, OperationPrintingFlags, OperationRefMut,
};
use mlir_compile::CompileConfig;
use rustc_codegen_ssa_gpu::back::write::{CodegenContext, ModuleConfig};
use rustc_codegen_ssa_gpu::{CompiledModule, ModuleCodegen};
use tracing::{debug, trace};

use crate::backend::{GPUCodeGenModule, GPUCodegenBackend};

pub(crate) fn get_compile_config(config: &rustc_session::config::CodegenOptions) -> CompileConfig {
    let cpu = config.target_cpu.as_ref().map_or(
        rustc_session::config::host_tuple().split("-").next().unwrap().to_string(),
        |cpu| cpu.clone(),
    );
    CompileConfig::from_target_llvm_args(cpu.as_str(), config.llvm_args.iter().cloned())
}

pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    module: ModuleCodegen<GPUCodeGenModule>,
    config: &ModuleConfig,
) -> Result<rustc_codegen_ssa_gpu::CompiledModule, rustc_errors::FatalError> {
    let mod_name = module.name.clone();
    let module_name = &mod_name[..];
    let out = if let Some(mut m) = module.module_llvm.mlir_module {
        let out = cgcx.output_filenames.temp_path_for_cgu(
            rustc_session::config::OutputType::Mir,
            module_name,
            None,
        );
        let out_object = cgcx.output_filenames.temp_path_for_cgu(
            rustc_session::config::OutputType::Object,
            module_name,
            None,
        );
        debug!("write MLIR module to {:?}", out);
        let mut op = m.module.as_operation_mut();
        crate::mlir::visit::visit_ops_recursively(&mut op, &|op: &mut OperationRefMut<'_, '_>| {
            if op.attribute("to_remove").is_ok() {
                op.remove_from_parent();
            }
        });
        let flags = OperationPrintingFlags::new().enable_debug_info(true, false);
        let content = op.to_string_with_flags(flags).unwrap();
        if !m.module.as_operation().verify() {
            trace!("MLIR module verify failed.");
            std::fs::write(&out, &content).unwrap();
            Err(rustc_errors::FatalError)?;
        }
        let content = content.replace("attributes {kernel, ", "kernel attributes {");
        let content = content.replace("uniform", "uniform {}");
        let content = content.replace("#gpu.address_space<private>", "0");

        std::fs::write(&out, &content).unwrap();
        debug!("[Done]write MLIR module to {:?}", out);
        // mlir-opt must use "shell" in order to pass correct arguments.
        get_compile_config(&cgcx.opts.cg)
            .mlir_compile(&out, &out_object)
            .expect("Failed to compile MLIR to object file");

        // When emit-llvmir, we also emit ptx code.
        if config.emit_ir {
            let ptx_file = cgcx.output_filenames.with_extension("ptx");
            let tmp_ptx_file = out_object.with_extension("ptx");
            std::fs::copy(&tmp_ptx_file, &ptx_file)
                .expect("Failed to copy object file to ptx file");
        }
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
