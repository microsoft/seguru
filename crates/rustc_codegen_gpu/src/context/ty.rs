use rustc_codegen_ssa::traits::{LayoutTypeCodegenMethods, TypeMembershipCodegenMethods};

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
