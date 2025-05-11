use melior::ir::operation::OperationPrintingFlags;
use melior::ir::BlockLike;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::{
    back::write::{CodegenContext, ModuleConfig},
    CompiledModule,
};
use rustc_errors::DiagCtxtHandle;

use crate::attr::is_gpu_code;
use crate::{
    backend::{GPUCodeGenModule, GPUCodegenBackend},
    builder::GpuBuilder,
    context::GPUCodegenContext,
};
use rustc_span::Symbol;

pub(crate) fn codegen(
    cgcx: &CodegenContext<GPUCodegenBackend>,
    dcx: DiagCtxtHandle<'_>,
    module: ModuleCodegen<GPUCodeGenModule>,
    _config: &ModuleConfig,
) -> Result<rustc_codegen_ssa::CompiledModule, rustc_errors::FatalError> {
    let mod_name = module.name.clone();
    let module_name = Some(&mod_name[..]);
    log::trace!("try write MLIR module to");
    let out = if let Some(m) = module.module_llvm.mlir_module {
        let out = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Mir, module_name);
        log::trace!("write MLIR module to {:?}", out);
        let content = m
            .module
            .as_operation()
            .to_string_with_flags(
                OperationPrintingFlags::new()
                    .use_local_scope()
                    .print_generic_operation_form(),
            )
            .unwrap();
        std::fs::write(&out, &content).unwrap();
        if !m.module.as_operation().verify() {
            log::trace!("MLIR module verify failed: {}", content);
            Err(rustc_errors::FatalError)?;
        }
        log::trace!("MLIR module verified");

        Some(out)
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

pub(crate) fn module_codegen<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    cgu_name: Symbol,
) -> ModuleCodegen<GPUCodeGenModule> {
    let mlir_ctx = crate::mlir::create_mlir_ctx();
    let (mlir_module, cpu_block, gpu_block) = crate::mlir::create_top_module(mlir_ctx);
    let mut blocks = std::collections::HashMap::new();
    blocks.insert("host".to_string(), cpu_block);
    blocks.insert("gpu".to_string(), gpu_block);

    log::trace!("create MLIR module {}", cgu_name);
    let cgu = tcx.codegen_unit(cgu_name);
    {
        let cx: GPUCodegenContext<'_, '_, '_> =
            crate::context::GPUCodegenContext::new(tcx, mlir_ctx, &mlir_module, blocks);
        let mono_items = cgu.items_in_deterministic_order(tcx);
        for &(mono_item, data) in &mono_items {
            if is_gpu_code(&tcx, mono_item.def_id()) {
                log::trace!("predefine {}", mono_item);
            } else {
                log::trace!("skip {}", mono_item);
                continue;
            }
            mono_item.predefine::<GpuBuilder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
        }
        for (mono_item, mono_data) in mono_items {
            if is_gpu_code(&tcx, mono_item.def_id()) {
                log::trace!("define {}", mono_item);
            } else {
                log::trace!("skip {}", mono_item);
                continue;
            }
            mono_item.define::<GpuBuilder<'_, '_, '_>>(&cx);
        }
    }

    let module = GPUCodeGenModule {
        llvm_module: None,
        mlir_module: Some(crate::backend::MLIRModule {
            module: mlir_module,
        }),
    };
    let m = ModuleCodegen::new_regular(cgu_name.to_string(), module);
    eprintln!("compile_codegen_unit {}", cgu_name);
    m
}
