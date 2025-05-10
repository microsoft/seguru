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
        std::fs::write(&out, m.module.as_operation().to_string()).unwrap();
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
    let mlir_ctx: &'static melior::Context = Box::leak(Box::new(melior::Context::new()));
    let location = melior::ir::Location::unknown(mlir_ctx);
    let mlir_module = melior::ir::Module::new(location);
    let mlir_body = mlir_module.body();
    log::trace!("create MLIR module {}", cgu_name);
    let cgu = tcx.codegen_unit(cgu_name);
    {
        let cx: GPUCodegenContext<'_, '_, '_> =
            crate::context::GPUCodegenContext::new(tcx, mlir_ctx, &mlir_module, mlir_body);
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
