use rustc_codegen_ssa::traits::PreDefineCodegenMethods;
use rustc_middle::mir::mono::Linkage;

use crate::{attr::GpuAttributes, mlir::MLIRVisibility};

use super::GPUCodegenContext;

fn to_mlir_visibility(visibility: rustc_middle::mir::mono::Visibility) -> MLIRVisibility {
    match visibility {
        rustc_middle::mir::mono::Visibility::Default => MLIRVisibility::Public,
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
        todo!()
    }

    fn predefine_fn(
        &self,
        instance: rustc_middle::ty::Instance<'tcx>,
        linkage: rustc_middle::mir::mono::Linkage,
        visibility: rustc_middle::mir::mono::Visibility,
        symbol_name: &str,
    ) {
        log::trace!(
            "Predefining function with name `{}` with linkage `{:?}` and attributes `{:?}`",
            symbol_name,
            linkage,
            self.tcx.codegen_fn_attrs(instance.def_id())
        );
        let visibility = if linkage != Linkage::Internal
            && self
                .tcx
                .is_compiler_builtins(rustc_hir::def_id::LOCAL_CRATE)
        {
            MLIRVisibility::Private
        } else {
            to_mlir_visibility(visibility)
        };
        let decl = self.to_mir_func_decl(instance, visibility);
    }
}
