use melior::ir::operation::{OperationLike, OperationMutLike, OperationRefMut};
use mlir_compile::CompileConfig;
use rustc_codegen_ssa_gpu::back::write::{CodegenContext, ModuleConfig};
use rustc_codegen_ssa_gpu::{CompiledModule, ModuleCodegen};
use rustc_errors::DiagCtxtHandle;
use tracing::{debug, trace};

use crate::backend::{GPUCodeGenModule, GPUCodegenBackend};

fn get_compile_config() -> CompileConfig {
    let mut cconfig = CompileConfig::default();
    std::env::var("USE_FAST").map(|v| cconfig.use_fast = v == "true").unwrap_or_default();
    std::env::var("USE_FTZ").map(|v| cconfig.use_ftz = v == "true").unwrap_or_default();
    std::env::var("NVPTX_ARCH").map(|v| cconfig.cubin_chip = v).unwrap_or_default();
    std::env::var("NVPTX_FEATURES").map(|v| cconfig.cubin_features = v).unwrap_or_default();
    std::env::var("PTXAS_OPT_LEVEL")
        .map(|v| cconfig.opt_level = v.parse().unwrap_or_default())
        .unwrap_or_default();
    cconfig
}

pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GPUCodeGenModule>,
    config: &ModuleConfig,
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
        let content = content.replace("uniform", "uniform {}");

        std::fs::write(&out, &content).unwrap();
        debug!("[Done]write MLIR module to {:?}", out);
        // mlir-opt must use "shell" in order to pass correct arguments.
        get_compile_config()
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
