use rustc_codegen_ssa_gpu::ModuleCodegen;
use rustc_codegen_ssa_gpu::mono_item::MonoItemExt;
use rustc_span::Symbol;
use tracing::trace;

use crate::backend::GPUCodeGenModule;
use crate::builder::GpuBuilder;
use crate::context::GPUCodegenContext;

pub(crate) fn module_codegen<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    cgu_name: Symbol,
) -> ModuleCodegen<GPUCodeGenModule> {
    let mlir_ctx = crate::mlir::create_mlir_ctx();
    let (mlir_module, gpu_block, cpu_block) = crate::mlir::create_top_module(mlir_ctx);
    let mut blocks = std::collections::HashMap::new();
    blocks.insert("host".to_string(), mlir_module.body());
    blocks.insert("gpu".to_string(), gpu_block);

    trace!("create MLIR module {}", cgu_name);
    let cgu = tcx.codegen_unit(cgu_name);
    {
        let cx: GPUCodegenContext<'_, '_, '_> = crate::context::GPUCodegenContext::new(
            cgu_name.as_str().to_string(),
            cgu,
            tcx,
            mlir_ctx,
            &mlir_module,
            blocks,
        );
        let mono_items = cgu.items_in_deterministic_order(tcx);
        for &(mono_item, data) in &mono_items {
            let attr = crate::attr::GpuAttributes::build(&tcx, mono_item.def_id());
            if attr.is_gpu_related() {
                trace!("define {}", mono_item);
            } else {
                trace!("skip {}", mono_item);
                continue;
            }
            mono_item.predefine::<GpuBuilder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
        }
        for (mono_item, mono_data) in mono_items {
            let attr = crate::attr::GpuAttributes::build(&tcx, mono_item.def_id());
            if attr.is_gpu_related() {
                trace!("define {}", mono_item);
            } else {
                trace!("skip {}", mono_item);
                continue;
            }
            match &mono_item {
                rustc_middle::mir::mono::MonoItem::Fn(instance) => {
                    /*let mir = tcx.optimized_mir(instance.def_id());
                    let output = String::new();
                    let mut out = Vec::new();
                    rustc_middle::mir::pretty::write_mir_pretty(
                        tcx,
                        Some(instance.def_id()),
                        &mut out,
                    )
                    .unwrap();
                    debug!("mir {}", String::from_utf8(out).unwrap());*/
                    if !attr.is_builtin() {
                        mono_item.define::<GpuBuilder<'_, '_, '_>>(&cx);
                    }
                }
                rustc_middle::mir::mono::MonoItem::Static(def_id) => todo!(),
                rustc_middle::mir::mono::MonoItem::GlobalAsm(item_id) => todo!(),
            }
        }
    }
    let module =
        GPUCodeGenModule { mlir_module: Some(crate::backend::MLIRModule { module: mlir_module }) };
    let m = ModuleCodegen::new_regular(cgu_name.to_string(), module);
    eprintln!("compile_codegen_unit {}", cgu_name);
    m
}
