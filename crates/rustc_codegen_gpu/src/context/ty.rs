use log_derive::{logfn, logfn_inputs};
use melior::dialect::llvm::r#type as mlir_llvm_type;
use rustc_abi::{BackendRepr, Primitive};
use rustc_codegen_ssa::traits::{
    BackendTypes, BaseTypeCodegenMethods, DerivedTypeCodegenMethods, LayoutTypeCodegenMethods,
    TypeMembershipCodegenMethods,
};
use rustc_middle::ty::layout::LayoutOf;

use super::GPUCodegenContext;

use melior::ir::{self as mlir_ir, r#type as mlir_type};
use melior::ir::{ShapedTypeLike, TypeLike};

pub type MLIRType<'ctx> = mlir_ir::Type<'ctx>;

impl<'tcx, 'ml, 'a> TypeMembershipCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    fn type_empty(&self) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::TupleType::new(self.mlir_ctx, &[]))
    }

    pub(crate) fn type_index(&self) -> MLIRType<'ml> {
        mlir_ir::Type::index(self.mlir_ctx)
    }

    pub(crate) fn type_i1(&self) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 1))
    }

    pub(crate) fn type_tensor(&self, ty: MLIRType<'ml>, len: u64) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::RankedTensorType::new(&[len], ty.into(), None))
    }

    pub fn type_padding_filler(
        &self,
        size: rustc_abi::Size,
        align: rustc_abi::Align,
    ) -> MLIRType<'ml> {
        let unit = rustc_abi::Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    /// TODO(delete): this should never be called
    fn type_struct(
        &self,
        fields: &[<GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type],
        packed: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        panic!();
        /*MLIRType::from(mlir_llvm_type::r#struct(
            self.mlir_ctx,
            &fields.iter().map(|a| (*a).into()).collect::<Vec<_>>(),
            packed,
        ))*/
    }

    fn scalar_mlir_type(
        &self,
        scalar: &rustc_abi::Scalar,
        ptr_ty: Option<&rustc_middle::ty::Ty<'tcx>>,
        immediate: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        if immediate && scalar.is_bool() {
            return self.type_i1();
        }
        match scalar.primitive() {
            Primitive::Int(i, _signed) => self.type_from_integer(i),
            Primitive::Float(f) => self.type_from_float(f),
            Primitive::Pointer(a) => {
                let ty = ptr_ty.unwrap().builtin_deref(true).unwrap();
                let layout = self.layout_of(ty);
                MLIRType::from(mlir_type::MemRefType::new(
                    self.mlir_type(layout, immediate).into(),
                    &[1],
                    None,
                    None,
                ))
            }
        }
    }

    fn struct_fields(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
    ) -> (Vec<MLIRType<'ml>>, bool) {
        let field_count = layout.fields.count();

        let mut packed = false;
        let mut offset = rustc_abi::Size::ZERO;
        let mut prev_effective_align = layout.align.abi;
        let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
        for i in layout.fields.index_by_increasing_offset() {
            let target_offset = layout.fields.offset(i);
            let field = layout.field(self, i);
            let effective_field_align = layout
                .align
                .abi
                .min(field.align.abi)
                .restrict_for_offset(target_offset);
            packed |= effective_field_align < field.align.abi;

            assert!(target_offset >= offset);
            let padding = target_offset - offset;
            if padding != rustc_abi::Size::ZERO {
                let padding_align = prev_effective_align.min(effective_field_align);
                assert_eq!(offset.align_to(padding_align) + padding, target_offset);
                result.push(self.type_padding_filler(padding, padding_align));
            }

            result.push(self.mlir_type(field, false));
            offset = target_offset + field.size;
            prev_effective_align = effective_field_align;
        }
        if layout.is_sized() && field_count > 0 {
            if offset > layout.size {
                panic!(
                    "layout: {:#?} stride: {:?} offset: {:?}",
                    layout, layout.size, offset
                );
            }
            let padding = layout.size - offset;
            if padding != rustc_abi::Size::ZERO {
                let padding_align = prev_effective_align;
                assert_eq!(offset.align_to(padding_align) + padding, layout.size);
                result.push(self.type_padding_filler(padding, padding_align));
            }
        }
        (result, packed)
    }

    pub(crate) fn mlir_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
        immediate: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        let cx = self.mlir_ctx;
        match layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                self.scalar_mlir_type(&scalar, Some(&layout.ty), immediate)
            }
            BackendRepr::ScalarPair(s1, s2) => {
                // An immediate pair always contains just the two elements, without any padding
                // filler, as it should never be stored to memory.
                let t1 = self
                    .scalar_pair_element_backend_type(layout, 0, false)
                    .into();
                let t2 = self
                    .scalar_pair_element_backend_type(layout, 1, false)
                    .into();
                MLIRType::from(mlir_type::TupleType::new(self.mlir_ctx, &[t1, t2]))
            }
            BackendRepr::SimdVector { element, count } => todo!(),
            BackendRepr::Memory { .. } if !layout.is_zst() => match &layout.fields {
                rustc_abi::FieldsShape::Primitive => todo!(),
                rustc_abi::FieldsShape::Union(non_zero) => todo!(),
                rustc_abi::FieldsShape::Array { stride, count } => {
                    let elem = self.mlir_type(layout.field(self, 0), immediate);
                    if *count == 0 {
                        dbg!(layout.ty);
                    }
                    self.type_array(elem, *count)
                }
                rustc_abi::FieldsShape::Arbitrary {
                    offsets,
                    memory_index,
                } => {
                    let (fields, packed) = self.struct_fields(layout);
                    dbg!(layout.ty);
                    panic!("GPU programming model does not support struct");
                    //self.type_struct(&fields, packed);
                }
            },
            _ => self.type_empty(),
        }
    }
}

impl<'tcx, 'ml, 'a> BaseTypeCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn type_i8(&self) -> Self::Type {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 8))
    }

    fn type_i16(&self) -> Self::Type {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 16))
    }

    fn type_i32(&self) -> Self::Type {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 32))
    }

    fn type_i64(&self) -> Self::Type {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 64))
    }

    fn type_i128(&self) -> Self::Type {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 128))
    }

    fn type_isize(&self) -> Self::Type {
        self.type_index()
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
        if len == 0 {
            return ty; // &[T] is the same as &T
        }
        self.type_tensor(ty, len)
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        MLIRType::from(mlir_type::FunctionType::new(
            self.mlir_ctx,
            &args.iter().map(|a| (*a).into()).collect::<Vec<_>>(),
            &[ret.into()],
        ))
    }

    fn type_ptr(&self) -> Self::Type {
        self.type_ptr_ext(rustc_abi::AddressSpace::DATA)
    }

    fn element_type(&self, ty: Self::Type) -> Self::Type {
        if ty.is_ranked_tensor() {
            mlir_type::RankedTensorType::try_from(ty).unwrap().element()
        } else if ty.is_mem_ref() {
            mlir_type::MemRefType::try_from(ty).unwrap().element()
        } else {
            panic!("Unsupported type: {:?}", ty);
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
        MLIRType::from(melior::dialect::llvm::r#type::pointer(
            self.mlir_ctx,
            address_space.0,
        ))
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa::common::TypeKind {
        if ty.is_float() {
            rustc_codegen_ssa::common::TypeKind::Float
        } else if ty.is_integer() {
            rustc_codegen_ssa::common::TypeKind::Integer
        } else if ty.is_ranked_tensor() {
            rustc_codegen_ssa::common::TypeKind::Array
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

impl<'tcx, 'ml, 'a> LayoutTypeCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn backend_type(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> Self::Type {
        let ty = self.mlir_type(layout, false);
        ty
    }

    fn cast_backend_type(&self, ty: &rustc_target::callconv::CastTarget) -> Self::Type {
        todo!()
    }

    fn fn_decl_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        self.fn_abi_to_fn_type(fn_abi).into()
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
        self.mlir_type(layout, true)
    }

    fn is_backend_immediate(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        match layout.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => true,
            BackendRepr::ScalarPair(..) => false,
            BackendRepr::Memory { .. } => layout.is_zst(),
        }
    }

    fn is_backend_scalar_pair(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        matches!(layout.backend_repr, BackendRepr::ScalarPair(..))
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Self::Type {
        let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
            panic!(
                "TyAndLayout::scalar_pair_element_llty({:?}): not applicable",
                self
            );
        };
        let pair_typs = match layout.ty.kind() {
            rustc_middle::ty::TyKind::Slice(ty) => [Some(ty), None],
            rustc_middle::ty::TyKind::Ref(_, ty, _) => [Some(&layout.ty), None],
            _ => {
                todo!("{:?}", layout.ty.kind());
            }
        };
        let scalar = [a, b][index];
        let ty = pair_typs[index];
        self.scalar_mlir_type(&scalar, ty, immediate)
    }
}
