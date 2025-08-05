use rustc_codegen_ssa_gpu::traits::PreDefineCodegenMethods;
use rustc_middle::ty::Instance;
use tracing::trace;

use super::GPUCodegenContext;
use crate::mlir::MLIRVisibility;
use crate::rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};

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
        let instance = Instance::mono(self.tcx, def_id);
        let ty = instance.ty(self.tcx, self.typing_env());
        let llty = self.mlir_type(self.layout_of(ty), false);
        if attr.shared_size {
            panic!(
                "static shared memory should be defined using SharedGpu::<T>::zero() inside a GPU kernel function"
            );
        }
        let span = self.tcx.def_span(def_id);
        let val = match self.tcx.eval_static_initializer(def_id) {
            Ok(alloc) => self.const_data_memref_from_alloc(alloc, symbol_name),
            _ => {
                self.emit_error(
                    "Failed to evaluate static initializer".into(),
                    self.tcx.def_span(def_id),
                );
            }
        };
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
