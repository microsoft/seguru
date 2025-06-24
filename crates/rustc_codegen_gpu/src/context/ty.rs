use melior::ir::{self as mlir_ir, ShapedTypeLike, TypeLike, r#type as mlir_type};
use rustc_abi::{BackendRepr, Primitive};
use rustc_codegen_ssa_gpu::traits::{
    BackendTypes, BaseTypeCodegenMethods, DerivedTypeCodegenMethods, LayoutTypeCodegenMethods,
    TypeMembershipCodegenMethods,
};
use rustc_middle::ty::layout::LayoutOf;
use tracing::debug;

use super::GPUCodegenContext;

pub type MLIRType<'ctx> = mlir_ir::Type<'ctx>;

impl<'tcx, 'ml, 'a> TypeMembershipCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub(crate) fn align_to_attr(
        &self,
        align: rustc_abi::Align,
    ) -> mlir_ir::attribute::IntegerAttribute<'ml> {
        mlir_ir::attribute::IntegerAttribute::new(self.type_i64(), align.bytes() as i64)
    }

    pub(crate) fn mlir_integer_width(&self, ty: MLIRType<'_>) -> usize {
        let int_ty = mlir_type::IntegerType::try_from(ty).unwrap();
        int_ty.width() as usize
    }

    pub(crate) fn mlir_float_width(&self, ty: MLIRType<'_>) -> usize {
        assert!(ty.is_float());
        unsafe { mlir_sys::mlirFloatTypeGetWidth(ty.to_raw()) as usize }
    }

    pub(crate) fn mlir_element_type<'b>(&self, ty: MLIRType<'b>) -> MLIRType<'b> {
        if ty.is_ranked_tensor() {
            mlir_type::RankedTensorType::try_from(ty).unwrap().element()
        } else if ty.is_mem_ref() {
            mlir_type::MemRefType::try_from(ty).unwrap().element()
        } else {
            panic!("Unsupported type: {:?}", ty);
        }
    }

    fn type_empty(&self) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::Type::none(self.mlir_ctx))
    }

    /// Note: Tuple is an abstract type in MLIR,
    /// so we need to use self.expand_type to get the real types.
    pub(crate) fn type_tuple(&self, types: &[MLIRType<'ml>]) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::TupleType::new(self.mlir_ctx, types))
    }

    pub(crate) fn type_memref(&self, eletype: MLIRType<'ml>, dim: &[i64]) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::MemRefType::new(
            eletype,
            dim,
            Some(crate::mlir::memref::default_memref_layout(self.mlir_ctx)),
            None,
        ))
    }

    pub(crate) fn type_memref_single(&self, eletype: MLIRType<'ml>) -> MLIRType<'ml> {
        self.type_memref(eletype, &[1])
    }

    pub(crate) fn type_index(&self) -> MLIRType<'ml> {
        mlir_ir::Type::index(self.mlir_ctx)
    }

    pub(crate) fn type_i1(&self) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, 1))
    }

    pub(crate) fn type_tensor(&self, ty: MLIRType<'ml>, len: u64) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::RankedTensorType::new(&[len], ty, None))
    }

    pub(crate) fn expand_type(&self, ty: MLIRType<'ml>) -> Vec<MLIRType<'ml>> {
        let mut rets: Vec<MLIRType<'ml>> = vec![];
        if let Ok(tuple) = mlir_type::TupleType::<'ml>::try_from(ty) {
            for i in 0..tuple.type_count() {
                let t = self.use_raw_type(tuple.r#type(i).unwrap());
                if t.is_tuple() {
                    rets.append(&mut self.expand_type(t));
                } else {
                    rets.push(t);
                }
            }
            rets
        } else {
            vec![ty]
        }
    }

    /// In Rust, a closure is zero-sized. A function will be translated to
    /// multiple functions specilized with the closure if a function takes a closure as an argument.
    pub fn ty_to_closure(
        &self,
        ty: &rustc_middle::ty::Ty<'tcx>,
    ) -> Option<rustc_middle::ty::Instance<'tcx>> {
        if let rustc_middle::ty::Closure(closure_def_id, closure_substs) = *ty.kind() {
            // ✅ You’ve found the closure passed to this call.
            // Closure type is represented as a type(ClosureArgs) that implements fn trait. Thus.
            // the call only see the ClosureArgs as inputs and need to explicitly resolve the closure.
            let closure_inst = rustc_middle::ty::Instance::resolve_closure(
                self.tcx,
                closure_def_id,
                closure_substs,
                rustc_middle::ty::ClosureKind::FnOnce,
            );
            debug!("Closure def_id: {:?}", closure_def_id);
            Some(closure_inst)
        } else {
            None
        }
    }

    fn arbitrary_mlir_type(
        &self,
        layout: &rustc_middle::ty::layout::TyAndLayout<'tcx>,
        immediate: bool,
    ) -> MLIRType<'ml> {
        // This is used for types that are not directly supported by the backend,
        // such as `dyn Trait` or `impl Trait`.
        // We use a tuple type to represent these types.
        let mut mlir_types = vec![];
        let field_count = layout.fields.count();
        let mut offset = rustc_abi::Size::ZERO;
        for i in layout.fields.index_by_increasing_offset() {
            let field = layout.field(self, i);

            let target_offset = layout.fields.offset(i);
            let padding_size = target_offset - offset;

            if padding_size != rustc_abi::Size::ZERO {
                // Add padding
                mlir_types.push(self.type_array(self.type_i8(), padding_size.bytes()));
            }
            mlir_types.push(self.mlir_type(field, immediate));
            offset = target_offset + field.size;
        }
        if mlir_types.len() > 1 {
            use crate::rustc_middle::query::Key;
            self.emit_error(
                "Does not support tuple with more than 1 elements".into(),
                self.tcx.def_span(layout.ty.def_id_for_ty_in_cycle().unwrap()),
            )
        }
        self.type_tuple(&mlir_types)
    }

    fn pointer_to_mlir_type(
        &self,
        ty: &rustc_middle::ty::Ty<'tcx>,
        immediate: bool,
    ) -> MLIRType<'ml> {
        match ty.kind() {
            rustc_middle::ty::TyKind::Str => {
                let ty = ty.builtin_deref(true).unwrap_or_else(|| panic!("{:?}", ty));
                let layout = self.layout_of(ty);
                self.type_memref_single(self.type_i8())
            }
            rustc_middle::ty::TyKind::RawPtr(inner_type, _)
            | rustc_middle::ty::TyKind::Ref(_, inner_type, _)
            | rustc_middle::ty::TyKind::Slice(inner_type) => {
                let layout = self.layout_of(*inner_type);
                let bytes = layout.size.bytes() as i64;
                if bytes == 0 {
                    self.type_memref_single(self.type_i8())
                } else {
                    self.type_memref(self.type_i8(), &[bytes])
                }
            }
            _ => {
                panic!("Unsupported pointer type: {:?}", ty);
            }
        }
    }

    fn scalar_mlir_type(
        &self,
        scalar: &rustc_abi::Scalar,
        ty: Option<&rustc_middle::ty::Ty<'tcx>>,
        immediate: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        if immediate && scalar.is_bool() {
            return self.type_i1();
        }
        if let Some(ty) = ty {
            match ty.kind() {
                rustc_middle::ty::TyKind::Int(rustc_middle::ty::IntTy::Isize)
                | rustc_middle::ty::TyKind::Uint(rustc_middle::ty::UintTy::Usize) => {
                    return self.type_index();
                }
                _ => {}
            }
        }
        match scalar.primitive() {
            Primitive::Int(i, _signed) => self.type_from_integer(i),
            Primitive::Float(f) => self.type_from_float(f),
            Primitive::Pointer(a) => {
                if let Some(ptr_ty) = ty {
                    let ty = self.pointer_to_mlir_type(ptr_ty, immediate);
                    assert!(ty.is_mem_ref());
                    ty
                } else {
                    self.type_memref_single(self.type_i8())
                }
            }
        }
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
                let t1 = self.scalar_pair_element_backend_type(layout, 0, false);
                let t2 = self.scalar_pair_element_backend_type(layout, 1, false);
                MLIRType::from(mlir_type::TupleType::new(self.mlir_ctx, &[t1, t2]))
            }
            BackendRepr::SimdVector { element, count } => todo!(),
            BackendRepr::Memory { .. } if !layout.is_zst() => match &layout.fields {
                rustc_abi::FieldsShape::Primitive => todo!(),
                rustc_abi::FieldsShape::Union(non_zero) => todo!(),
                rustc_abi::FieldsShape::Array { stride, count } => {
                    let elem = self.mlir_type(layout.field(self, 0), immediate);
                    if *count == 0 {
                        return elem;
                    }
                    self.type_array(elem, *count)
                }
                rustc_abi::FieldsShape::Arbitrary { offsets, memory_index } => {
                    dbg!(layout.ty);
                    self.arbitrary_mlir_type(&layout, immediate)
                }
            },
            _ => {
                let empty_ty = layout.ty;
                self.type_empty()
            }
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
        MLIRType::from(mlir_type::FunctionType::new(self.mlir_ctx, args, &[ret]))
    }

    fn type_ptr(&self) -> Self::Type {
        self.type_ptr_ext(rustc_abi::AddressSpace::DATA)
    }

    fn element_type(&self, ty: Self::Type) -> Self::Type {
        self.mlir_element_type(ty)
    }

    fn vector_length(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn float_width(&self, ty: Self::Type) -> usize {
        self.mlir_float_width(ty)
    }

    fn int_width(&self, ty: Self::Type) -> u64 {
        self.mlir_integer_width(ty) as _
    }

    fn val_ty(&self, v: Self::Value) -> Self::Type {
        todo!()
    }

    fn type_ptr_ext(&self, address_space: rustc_abi::AddressSpace) -> Self::Type {
        MLIRType::from(melior::dialect::llvm::r#type::pointer(self.mlir_ctx, address_space.0))
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa_gpu::common::TypeKind {
        if ty.is_float() {
            rustc_codegen_ssa_gpu::common::TypeKind::Float
        } else if ty.is_integer() {
            rustc_codegen_ssa_gpu::common::TypeKind::Integer
        } else if ty.is_ranked_tensor() {
            rustc_codegen_ssa_gpu::common::TypeKind::Array
        } else if ty.is_llvm_pointer_type() {
            rustc_codegen_ssa_gpu::common::TypeKind::Pointer
        } else if ty.is_function() {
            rustc_codegen_ssa_gpu::common::TypeKind::Function
        } else if ty.is_vector() {
            rustc_codegen_ssa_gpu::common::TypeKind::Vector
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
        self.fn_abi_to_fn_type(fn_abi)
            .unwrap_or_else(|msg| {
                panic!();
            })
            .into()
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
            panic!("TyAndLayout::scalar_pair_element_llty({:?}): not applicable", self);
        };
        let scalar: rustc_abi::Scalar = [a, b][index];
        self.scalar_mlir_type(&scalar, None, immediate)
    }
}
