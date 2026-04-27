use rustc_codegen_ssa_gpu::traits::PreDefineCodegenMethods;
use rustc_middle::ty::Instance;
use tracing::trace;

use super::GPUCodegenContext;
use crate::rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub(crate) fn to_mlir_linkage(
        &self,
        linkage: rustc_middle::mir::mono::Linkage,
    ) -> melior::ir::Attribute<'ml> {
        let link = match linkage {
            rustc_middle::mir::mono::Linkage::External => "external",
            rustc_middle::mir::mono::Linkage::AvailableExternally => "available_externally",
            rustc_middle::mir::mono::Linkage::LinkOnceAny => "linkonce",
            rustc_middle::mir::mono::Linkage::LinkOnceODR => "linkonce_odr",
            rustc_middle::mir::mono::Linkage::WeakAny => "weak",
            rustc_middle::mir::mono::Linkage::WeakODR => "weak_odr",
            rustc_middle::mir::mono::Linkage::Internal => "internal",
            rustc_middle::mir::mono::Linkage::ExternalWeak => "extern_weak",
            rustc_middle::mir::mono::Linkage::Common => "common",
        };
        melior::ir::Attribute::parse(self.mlir_ctx, &format!("#llvm.linkage<{link}>")).unwrap()
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
        let attr = self.gpu_attrs(&instance);
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
        let attr = self.gpu_attrs(&instance);
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
        let decl = self.to_mir_func_decl(instance, self.to_mlir_linkage(linkage));
    }
}
