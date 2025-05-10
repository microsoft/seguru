use rustc_codegen_ssa::traits::MiscCodegenMethods;
use rustc_middle::{
    query::Key,
    ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv},
};

use super::GPUCodegenContext;

impl<'tcx, 'ml> MiscCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, '_> {
    fn vtables(
        &self,
    ) -> &std::cell::RefCell<
        rustc_data_structures::fx::FxHashMap<
            (
                rustc_middle::ty::Ty<'tcx>,
                Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>,
            ),
            Self::Value,
        >,
    > {
        todo!()
    }

    fn get_fn(&self, instance: rustc_middle::ty::Instance<'tcx>) -> Self::Function {
        let tcx = self.tcx();
        let mlir_ctx = self.mlir_ctx;
        let sym = tcx.symbol_name(instance).name;
        let def_id: rustc_hir::def_id::DefId = instance.def_id();
        log::trace!(
            "get_fn({:?}: {:?}) => {}",
            instance,
            instance.ty(tcx, self.typing_env()),
            sym
        );
        let location = self.to_mlir_loc(instance.def.default_span(tcx));
        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        if self.fn_db.contains_key(&def_id) {
            self.fn_db[&def_id]
        } else {
            self.to_mir_func_decl(instance, crate::mlir::MLIRVisibility::Public)
        }
    }

    fn get_fn_addr(&self, instance: rustc_middle::ty::Instance<'tcx>) -> Self::Value {
        self.to_mir_func_const(instance)
    }

    fn eh_personality(&self) -> Self::Value {
        todo!()
    }

    fn sess(&self) -> &rustc_session::Session {
        self.tcx.sess
    }

    fn codegen_unit(&self) -> &'tcx rustc_middle::mir::mono::CodegenUnit<'tcx> {
        todo!()
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
