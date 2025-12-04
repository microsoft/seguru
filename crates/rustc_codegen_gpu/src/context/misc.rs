use rustc_codegen_ssa_gpu::traits::MiscCodegenMethods;
use rustc_middle::query::Key;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};
use tracing::trace;

use super::GPUCodegenContext;

impl<'tcx, 'ml, 'a> MiscCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a>
where
    'tcx: 'a,
{
    fn vtables(
        &self,
    ) -> &std::cell::RefCell<
        rustc_data_structures::fx::FxHashMap<
            (rustc_middle::ty::Ty<'tcx>, Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>),
            Self::Value,
        >,
    > {
        todo!()
    }

    fn get_fn(&self, instance: rustc_middle::ty::Instance<'tcx>) -> Self::Function {
        let (ret, _) = self.get_fn_with_gpu_indicator(instance);
        ret
    }

    fn get_fn_with_gpu_indicator(
        &self,
        instance: rustc_middle::ty::Instance<'tcx>,
    ) -> (Self::Function, bool) {
        let tcx = self.tcx();
        let mlir_ctx = self.mlir_ctx;
        let sym = self.sanitized_symbol_name(instance);
        let def_id: rustc_hir::def_id::DefId = instance.def_id();
        trace!("[{}]:get_fn({:?}) => {}", self.cgu_name, instance, sym);
        let location = self.to_mlir_loc(instance.def.default_span(tcx));
        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let gpu_attrs = self.gpu_attrs(&instance);
        (
            self.to_mir_func_decl(
                instance,
                self.to_mlir_linkage(rustc_hir::attrs::Linkage::ExternalWeak),
            ),
            gpu_attrs.kernel || gpu_attrs.device,
        )
    }

    fn get_fn_addr(&self, instance: rustc_middle::ty::Instance<'tcx>) -> Self::Value {
        self.to_mir_func_const(instance, None)
    }

    fn eh_personality(&self) -> Self::Function {
        todo!()
    }

    fn sess(&self) -> &rustc_session::Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, llfn: Self::Function) {
        todo!()
    }

    fn apply_target_cpu_attr(&self, llfn: Self::Function) {
        todo!()
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        todo!()
    }
}
