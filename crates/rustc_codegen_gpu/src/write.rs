use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::{
    back::write::{CodegenContext, ModuleConfig},
    CompiledModule,
};
use rustc_errors::DiagCtxtHandle;

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

    let out = if let Some(m) = module.module_llvm.mlir_module {
        let out = cgcx
            .output_filenames
            .temp_path(rustc_session::config::OutputType::Object, module_name);
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
    (cgu_name, cx): (Symbol, GPUCodegenContext<'tcx, 'static, 'static>),
) -> ModuleCodegen<GPUCodeGenModule> {
    let cgu = tcx.codegen_unit(cgu_name);
    let module = GPUCodeGenModule {
        llvm_module: None,
        mlir_module: None,
    };
    {
        let mono_items = cgu.items_in_deterministic_order(tcx);
        for &(mono_item, data) in &mono_items {
            mono_item.predefine::<GpuBuilder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
        }
        for (mono_item, mono_data) in mono_items {
            mono_item.define::<GpuBuilder<'_, '_, '_>>(&cx);
        }
    }
    let m = ModuleCodegen::new_regular(cgu_name.to_string(), module);
    eprintln!("compile_codegen_unit {}", cgu_name);
    m
}
