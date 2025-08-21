use std::collections::VecDeque;

use rustc_codegen_ssa_gpu::ModuleCodegen;
use rustc_codegen_ssa_gpu::mono_item::MonoItemExt;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::Terminator;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::ty::Instance;
use rustc_middle::ty::layout::HasTypingEnv;
use rustc_span::Symbol;
use tracing::trace;

use crate::backend::GPUCodeGenModule;
use crate::builder::GpuBuilder;
use crate::context::GPUCodegenContext;

struct FindCallee<'tcx, 'ml, 'a, 'b> {
    cx: &'b GPUCodegenContext<'tcx, 'ml, 'a>,
    callees: Vec<Instance<'tcx>>,
}

impl<'tcx, 'ml, 'a> Visitor<'tcx> for FindCallee<'tcx, 'ml, 'a, '_> {
    fn visit_terminator(
        &mut self,
        terminator: &Terminator<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        let tcx = self.cx.tcx;
        // Visit the terminator to find reads/writes
        if let rustc_middle::mir::TerminatorKind::Call { func, .. } = &terminator.kind {
            if let Some((def_id, substs)) = func.const_fn_def() {
                if let Ok(Some(callee)) =
                    Instance::try_resolve(tcx, self.cx.typing_env(), def_id, substs)
                {
                    self.callees.push(callee);
                }
            }
        }
    }
}

fn callees_of_instance<'tcx>(
    cx: &GPUCodegenContext<'tcx, '_, '_>,
    instance: Instance<'tcx>,
) -> Vec<Instance<'tcx>> {
    let tcx = cx.tcx;
    let body = tcx.instance_mir(instance.def);
    let mut visit_calless = FindCallee { cx, callees: Vec::new() };
    visit_calless.visit_body(body);
    visit_calless.callees
}

fn find_gpu_related_mono_items<'tcx>(
    cx: &mut GPUCodegenContext<'tcx, '_, '_>,
    all_mono_items: &[(MonoItem<'tcx>, rustc_middle::mir::mono::MonoItemData)],
) -> Vec<(MonoItem<'tcx>, rustc_middle::mir::mono::MonoItemData)> {
    let tcx = cx.tcx;
    let mut gpu_items = FxHashSet::default();
    let mut results = vec![];
    let mut worklist = VecDeque::new();

    // Start from roots
    for (item, data) in all_mono_items {
        let attr = crate::attr::GpuAttributes::build(&tcx, item.def_id());
        if attr.is_gpu_related() {
            if !attr.is_builtin() {
                worklist.push_back(*item);
            }
            if let MonoItem::Fn(instance) = item {
                gpu_items.insert(*instance);
                cx.gpu_attrs.insert(*instance, attr);
            } else {
                // Static and GlobalAsm items are not traversed further
                results.push((*item, *data));
            }
        }
    }

    // Traverse dependencies
    while let Some(item) = worklist.pop_front() {
        match item {
            MonoItem::Fn(instance) => {
                for callee in callees_of_instance(cx, instance) {
                    if !gpu_items.contains(&callee) {
                        gpu_items.insert(callee);
                        let original_attr = cx.gpu_attrs(&callee);
                        if !original_attr.is_gpu_related() {
                            cx.gpu_attrs
                                .insert(callee, crate::attr::GpuAttributes::callee_device());
                            worklist.push_back(MonoItem::Fn(callee));
                        } else if !original_attr.is_builtin() {
                            // If the callee is a builtin, we don't need to traverse it further
                            worklist.push_back(MonoItem::Fn(callee));
                        }
                    }
                }
            }
            MonoItem::Static(_) | MonoItem::GlobalAsm(_) => {
                // Static and GlobalAsm items are not traversed further
            }
        }
    }

    for (item, data) in all_mono_items {
        if let MonoItem::Fn(instance) = item {
            if gpu_items.contains(instance) {
                results.push((*item, *data));
            }
        }
    }

    results
}

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
        let mut cx: GPUCodegenContext<'_, '_, '_> = crate::context::GPUCodegenContext::new(
            cgu_name.as_str().to_string(),
            cgu,
            tcx,
            mlir_ctx,
            &mlir_module,
            blocks,
        );
        let mono_items = cgu.items_in_deterministic_order(tcx);
        let mono_items = find_gpu_related_mono_items(&mut cx, &mono_items);
        for &(mono_item, data) in &mono_items {
            tracing::info!("is_gpu_related define {}", tcx.def_path_str(mono_item.def_id()));
            mono_item.predefine::<GpuBuilder<'_, '_, '_>>(&cx, data.linkage, data.visibility);
            cx.define_indirect_if_needed();
        }
        for (mono_item, mono_data) in mono_items {
            match &mono_item {
                rustc_middle::mir::mono::MonoItem::Fn(instance) => {
                    let attr = cx.gpu_attrs(instance);
                    if !attr.is_builtin() {
                        crate::mir_analysis::analyze_gpu_code(tcx, instance.def_id(), attr.kernel)
                            .unwrap_or_else(|err| {
                                err.fatal(tcx);
                            });
                        mono_item.define::<GpuBuilder<'_, '_, '_>>(&cx);
                        if attr.kernel {}
                    }
                }
                rustc_middle::mir::mono::MonoItem::Static(def_id) => {
                    continue;
                }
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
