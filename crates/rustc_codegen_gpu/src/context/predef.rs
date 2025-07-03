use rustc_codegen_ssa_gpu::traits::PreDefineCodegenMethods;
use rustc_middle::ty::Instance;
use tracing::trace;

use super::GPUCodegenContext;
use crate::mlir::MLIRVisibility;
use crate::rustc_middle::ty::layout::HasTypingEnv;

fn to_mlir_visibility(visibility: rustc_middle::mir::mono::Visibility) -> MLIRVisibility {
    match visibility {
        rustc_middle::mir::mono::Visibility::Default => MLIRVisibility::Private,
        rustc_middle::mir::mono::Visibility::Protected => MLIRVisibility::Public,
        rustc_middle::mir::mono::Visibility::Hidden => MLIRVisibility::Private,
    }
}
impl<'tcx, 'ml, 'a> PreDefineCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn predefine_static(
        &self,
        def_id: rustc_hir::def_id::DefId,
        linkage: rustc_middle::mir::mono::Linkage,
        visibility: rustc_middle::mir::mono::Visibility,
        symbol_name: &str,
    ) {
        let instance = Instance::mono(self.tcx, def_id);
        let def_path = self.tcx.def_path_str(def_id);
        let ty = instance.ty(self.tcx, self.typing_env());
        let attrs = self.tcx.get_attrs_unchecked(instance.def_id());
        let attr = crate::attr::GpuAttributes::build(&self.tcx, instance.def_id());
        if !attr.shared_size {
            self.emit_error(
                format!("Using static is only supported when used as shared memory size but {} is not marked with `#[gpu_codegen::shared_size]` {:?}", def_path, attrs),
                self.tcx.def_span(def_id),
            );
        }
        let span = self.tcx.def_span(def_id);
        if ty.kind() != &rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::Usize) {
            self.emit_error(
                format!("Shared memory size must be usize, found: {:?}", ty.kind()),
                span,
            );
        }

        let shared_size = match self.tcx.eval_static_initializer(def_id) {
            Ok(alloc) => alloc
                .inner()
                .read_scalar(
                    &self.tcx,
                    rustc_const_eval::interpret::AllocRange {
                        start: rustc_abi::Size::ZERO,
                        size: rustc_abi::Size::from_bytes(8),
                    },
                    false,
                )
                .unwrap()
                .to_u64()
                .unwrap() as usize,
            _ => {
                self.emit_error(
                    "Failed to evaluate static initializer".into(),
                    self.tcx.def_span(def_id),
                );
            }
        };
        let Some(name) = def_path.strip_prefix("shared_size_") else {
            self.emit_error(
                format!("Invalid symbol def_path: {}", def_path),
                self.tcx.def_span(def_id),
            );
        };
        trace!("Predefining shared memory size for `{}` with size {}", name, shared_size);
        self.expected_shared_memory_size.write().unwrap().insert(name.into(), (span, shared_size));
    }

    fn predefine_fn(
        &self,
        instance: rustc_middle::ty::Instance<'tcx>,
        linkage: rustc_middle::mir::mono::Linkage,
        visibility: rustc_middle::mir::mono::Visibility,
        symbol_name: &str,
    ) {
        let attr = crate::attr::GpuAttributes::build(&self.tcx, instance.def_id());
        if !attr.is_gpu_related() {
            return;
        }
        trace!(
            "Predefining function with name `{}` with linkage `{:?}` and attributes `{:?} {:?}`",
            symbol_name,
            linkage,
            self.tcx.codegen_fn_attrs(instance.def_id()),
            visibility
        );
        let visibility = to_mlir_visibility(visibility);
        let decl = self.to_mir_func_decl(instance, visibility);
    }
}
