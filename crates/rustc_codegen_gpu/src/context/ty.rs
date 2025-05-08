use rustc_codegen_ssa::traits::{
    LayoutTypeCodegenMethods, MiscCodegenMethods, TypeMembershipCodegenMethods,
};

use super::GPUCodegenContext;

use melior::ir as mlir_ir;
use melior::ir::TypeLike;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MLIRType<'ctx> {
    Raw(mlir_ir::Type<'ctx>),
    Array(mlir_ir::Type<'ctx>),
}

impl<'a> MLIRType<'a> {
    pub(crate) fn array(a: mlir_ir::Type<'a>) -> Self {
        MLIRType::Array(a)
    }
}

impl<'ctx> From<mlir_ir::Type<'ctx>> for MLIRType<'ctx> {
    fn from(ty: mlir_ir::Type<'ctx>) -> Self {
        MLIRType::Raw(ty)
    }
}

impl<'ctx> From<MLIRType<'ctx>> for mlir_ir::Type<'ctx> {
    fn from(ty: MLIRType<'ctx>) -> Self {
        match ty {
            MLIRType::Raw(ty) => ty,
            MLIRType::Array(ty) => ty,
        }
    }
}

pub(crate) fn array_element<'ctx>(ty: &mlir_ir::Type<'ctx>) -> mlir_ir::Type<'ctx> {
    unsafe {
        let raw = mlir_ir::Type::from(*ty).to_raw();
        mlir_ir::Type::from_raw(mlir_sys::mlirLLVMArrayTypeGetElementType(raw))
    }
}

impl<'tcx, 'ml, 'a> TypeMembershipCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {}

impl<'tcx, 'ml, 'a> LayoutTypeCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn backend_type(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> Self::Type {
        todo!()
    }

    fn cast_backend_type(&self, ty: &rustc_target::callconv::CastTarget) -> Self::Type {
        todo!()
    }

    fn fn_decl_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }

    fn fn_ptr_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }

    fn reg_backend_type(&self, ty: &rustc_abi::Reg) -> Self::Type {
        todo!()
    }

    fn immediate_backend_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
    ) -> Self::Type {
        todo!()
    }

    fn is_backend_immediate(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        todo!()
    }

    fn is_backend_scalar_pair(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        todo!()
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Self::Type {
        todo!()
    }
}

impl<'tcx, 'ml, 'a> MiscCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
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
        todo!()
    }

    fn get_fn_addr(&self, instance: rustc_middle::ty::Instance<'tcx>) -> Self::Value {
        todo!()
    }

    fn eh_personality(&self) -> Self::Value {
        todo!()
    }

    fn sess(&self) -> &rustc_session::Session {
        todo!()
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
