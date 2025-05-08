mod abi;
mod asm;
mod const_static;
mod coverage;
mod debug;
mod ty;

use core::panic;
use rustc_abi::{self, HasDataLayout};
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{
    BackendTypes, BaseTypeCodegenMethods, CoverageInfoBuilderMethods, DerivedTypeCodegenMethods,
    MiscCodegenMethods, PreDefineCodegenMethods,
};
use std::{marker::PhantomData, sync::Arc};

use melior::ir as mlir_ir;
use melior::ir::{r#type as mlir_type, TypeLike};
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};

use self::ty::MLIRType;

pub(crate) struct MLIRContext<'ml, 'a> {
    pub ctx: &'ml melior::Context,
    pub dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
}
pub(crate) struct GPUCodegenContext<'tcx, 'ml, 'a> {
    mlir_ctx: MLIRContext<'ml, 'a>,
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn new(tcx: rustc_middle::ty::TyCtxt<'tcx>, mlir_ctx: MLIRContext<'ml, 'a>) -> Self {
        Self { mlir_ctx, tcx }
    }
}

pub(crate) struct Funclet<'tcx, 'a> {
    cleanuppad: mlir_ir::Value<'tcx, 'a>,
    operand: mlir_ir::operation::OperationRef<'tcx, 'a>,
}

impl<'tcx, 'ml, 'a> HasTypingEnv<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn typing_env(&self) -> rustc_middle::ty::TypingEnv<'tcx> {
        rustc_middle::ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx, 'ml, 'a> HasTyCtxt<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx, 'ml, 'a> HasDataLayout for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
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
        todo!()
    }
}

impl<'tcx, 'ml, 'a> BackendTypes for GPUCodegenContext<'tcx, 'ml, 'a> {
    type Value = mlir_ir::Value<'ml, 'a>;

    type Metadata = ();

    type Function = mlir_ir::operation::OperationRef<'ml, 'a>;

    type BasicBlock = mlir_ir::block::BlockRef<'ml, 'a>;

    type Type = MLIRType<'ml>;

    /// Each Block may contain an instance of this, indicating whether the block is part of a landing pad or not. This is used to make
    /// decision about whether to emit invoke instructions (e.g., in a landing pad we don’t continue to use invoke) and also about
    /// various function call metadata.
    type Funclet = ();

    type DIScope = ();

    type DILocation = ();

    type DIVariable = ();
}

impl<'tcx, 'ml, 'a> BaseTypeCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn type_i8(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            8,
        )))
    }

    fn type_i16(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            16,
        )))
    }

    fn type_i32(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            32,
        )))
    }

    fn type_i64(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            64,
        )))
    }

    fn type_i128(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            128,
        )))
    }

    fn type_isize(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::IntegerType::new(
            self.mlir_ctx,
            size_of::<isize>() as u32 * 8,
        )))
    }

    fn type_f16(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::float16(self.mlir_ctx))
    }

    fn type_f32(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::float32(self.mlir_ctx))
    }

    fn type_f64(&self) -> Self::Type {
        MLIRType::from(mlir_ir::Type::float64(self.mlir_ctx))
    }

    fn type_f128(&self) -> Self::Type {
        todo!()
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        MLIRType::Array(melior::dialect::llvm::r#type::array(ty.into(), len as u32))
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(mlir_type::FunctionType::new(
            self.mlir_ctx,
            &args.iter().map(|a| (*a).into()).collect::<Vec<_>>(),
            &[ret.into()],
        )))
    }

    fn type_ptr(&self) -> Self::Type {
        self.type_ptr_ext(rustc_abi::AddressSpace::DATA)
    }

    fn element_type(&self, ty: Self::Type) -> Self::Type {
        match ty {
            MLIRType::Raw(ty) => unimplemented!(),
            MLIRType::Array(ty) => MLIRType::array(ty::array_element(&ty)),
        }
    }

    fn vector_length(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn float_width(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn int_width(&self, ty: Self::Type) -> u64 {
        todo!()
    }

    fn val_ty(&self, v: Self::Value) -> Self::Type {
        todo!()
    }

    fn type_ptr_ext(&self, address_space: rustc_abi::AddressSpace) -> Self::Type {
        MLIRType::from(mlir_ir::Type::from(melior::dialect::llvm::r#type::pointer(
            self.mlir_ctx,
            address_space.0,
        )))
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa::common::TypeKind {
        match ty {
            MLIRType::Array(_) => return rustc_codegen_ssa::common::TypeKind::Array,
            MLIRType::Raw(ty) => {
                if ty.is_float() {
                    rustc_codegen_ssa::common::TypeKind::Float
                } else if ty.is_integer() {
                    rustc_codegen_ssa::common::TypeKind::Integer
                } else if ty.is_vector() {
                    rustc_codegen_ssa::common::TypeKind::Vector
                } else if ty.is_llvm_pointer_type() {
                    rustc_codegen_ssa::common::TypeKind::Pointer
                } else if ty.is_function() {
                    rustc_codegen_ssa::common::TypeKind::Function
                } else if ty.is_vector() {
                    rustc_codegen_ssa::common::TypeKind::Vector
                } else {
                    panic!("Unsupported type: {:?}", ty);
                }
            }
        }
    }
}
