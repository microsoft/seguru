use rustc_codegen_ssa::{mono_item::MonoItemExt, ModuleCodegen};
use rustc_span::Symbol;

use crate::{
    attr::is_gpu_code, backend::GPUCodeGenModule, builder::GpuBuilder, context::GPUCodegenContext,
};

pub(crate) fn module_codegen<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    cgu_name: Symbol,
) -> ModuleCodegen<GPUCodeGenModule> {
    let mlir_ctx = crate::mlir::create_mlir_ctx();
    let (mlir_module, gpu_block, cpu_block) = crate::mlir::create_top_module(mlir_ctx);
    let mut blocks = std::collections::HashMap::new();
    blocks.insert("host".to_string(), mlir_module.body());
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
