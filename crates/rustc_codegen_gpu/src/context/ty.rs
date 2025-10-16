use melior::ir::r#type::MemRefType;
use melior::ir::{
    self as mlir_ir, Attribute, ShapedTypeLike, TypeLike, ValueLike, r#type as mlir_type,
};
use rustc_abi::{BackendRepr, Primitive};
use rustc_codegen_ssa_gpu::traits::{
    BackendTypes, BaseTypeCodegenMethods, DerivedTypeCodegenMethods, LayoutTypeCodegenMethods,
    MiscCodegenMethods, TypeMembershipCodegenMethods,
};
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use tracing::debug;

use super::GPUCodegenContext;
use crate::mlir::memref::MemorySpace;
use crate::mlir::{VectorType, float_width};

pub type MLIRType<'ctx> = mlir_ir::Type<'ctx>;

impl<'tcx, 'ml, 'a> TypeMembershipCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {}

fn flat_tuple<'tcx>(t: &rustc_middle::ty::Ty<'tcx>) -> Vec<rustc_middle::ty::Ty<'tcx>> {
    let mut type_list = Vec::new();
    let rustc_middle::ty::TyKind::Tuple(t) = t.kind() else { unreachable!() };
    for ty in t.as_slice() {
        if matches!(ty.kind(), rustc_middle::ty::TyKind::Tuple(t2)) {
            type_list.extend(flat_tuple(ty));
        } else {
            type_list.push(*ty);
        }
    }
    type_list
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub(crate) fn align_to_attr(
        &self,
        align: rustc_abi::Align,
    ) -> mlir_ir::attribute::IntegerAttribute<'ml> {
        mlir_ir::attribute::IntegerAttribute::new(self.type_i64(), align.bytes() as i64)
    }

    pub(crate) fn mlir_integer_width(&self, ty: MLIRType<'_>) -> usize {
        if ty.is_index() {
            return self.mlir_integer_width(self.type_i64());
        }
        let int_ty = mlir_type::IntegerType::try_from(ty).unwrap();
        int_ty.width() as usize
    }

    pub(crate) fn local_mem_space(&self) -> Attribute<'ml> {
        MemorySpace::Local.to_attr(self.mlir_ctx)
    }

    pub(crate) fn mlir_float_width(&self, ty: MLIRType<'_>) -> usize {
        assert!(ty.is_float());
        float_width(ty).unwrap()
    }

    pub(crate) fn mlir_element_type<'b>(&self, ty: MLIRType<'b>) -> MLIRType<'b> {
        if ty.is_ranked_tensor() {
            mlir_type::RankedTensorType::try_from(ty).unwrap().element()
        } else if ty.is_vector() {
            VectorType::try_from(ty).unwrap().element()
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

    pub(crate) fn memref_set_memory_space(
        &self,
        memref_ty: MemRefType<'ml>,
        memory_space: Option<Attribute<'ml>>,
    ) -> MemRefType<'ml> {
        self.type_memref(
            memref_ty.element(),
            &(0..memref_ty.rank())
                .map(|i| memref_ty.dim_size(i).unwrap() as i64)
                .collect::<Vec<_>>(),
            Some(memref_ty.layout()),
            memory_space,
        )
        .try_into()
        .unwrap()
    }

    /// Safety:
    ///   This is safe if eletype is not a memref type.
    ///   MemRef is not 8bytes and so we should be careful when using creating MemRef<nxMemRef<..>>.
    pub(crate) unsafe fn _type_memref(
        &self,
        eletype: MLIRType<'ml>,
        dim: &[i64],
        layout: Option<Attribute<'ml>>,
        memory_space: Option<Attribute<'ml>>,
    ) -> MemRefType<'ml> {
        crate::mlir::type_memref(self.mlir_ctx, eletype, dim, layout, memory_space)
    }

    pub(crate) fn type_shared_memref(
        &self,
        eletype: MLIRType<'ml>,
        dim: &[i64],
        layout: Option<Attribute<'ml>>,
    ) -> MemRefType<'ml> {
        assert!(!eletype.is_mem_ref());
        unsafe {
            self._type_memref(
                eletype,
                dim,
                layout,
                Some(MemorySpace::Shared.to_attr(self.mlir_ctx)),
            )
        }
    }

    pub(crate) fn type_memref(
        &self,
        eletype: MLIRType<'ml>,
        dim: &[i64],
        layout: Option<Attribute<'ml>>,
        memory_space: Option<Attribute<'ml>>,
    ) -> MLIRType<'ml> {
        if eletype.is_mem_ref() {
            // type_memref should be treated as 8bytes instead of the real memref type (>40bytes)
            return unsafe {
                self._type_memref(
                    self.type_i8(),
                    dim.iter().map(|&d| d * 8).collect::<Vec<_>>().as_slice(),
                    layout,
                    memory_space,
                )
                .into()
            };
        }
        unsafe { self._type_memref(eletype, dim, layout, memory_space).into() }
    }

    pub(crate) fn type_memref_single(
        &self,
        eletype: MLIRType<'ml>,
        memory_space: Option<Attribute<'ml>>,
    ) -> MLIRType<'ml> {
        self.type_memref(eletype, &[1], None, memory_space)
    }

    pub(crate) fn type_vector(&self, dimensions: &[u64], elem: MLIRType<'ml>) -> MLIRType<'ml> {
        VectorType::new(dimensions, elem).into()
    }

    pub(crate) fn type_index(&self) -> MLIRType<'ml> {
        mlir_ir::Type::index(self.mlir_ctx)
    }

    fn type_i_bits(&self, n: u32) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, n))
    }

    pub(crate) fn type_i1(&self) -> MLIRType<'ml> {
        self.type_i_bits(1)
    }

    pub(crate) fn type_tensor(&self, ty: MLIRType<'ml>, len: u64) -> MLIRType<'ml> {
        MLIRType::from(mlir_type::RankedTensorType::new(&[len], ty, None))
    }

    pub(crate) fn type_llvm_ptr(&self) -> MLIRType<'ml> {
        // This is a pointer to the LLVM dialect, which is used for LLVM intrinsics.
        // We use the `llvm.ptr` type from the LLVM dialect.
        MLIRType::from(melior::dialect::llvm::r#type::pointer(self.mlir_ctx, 0))
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
            if field.size.bytes() == 0 {
                // Skip zero-sized types
                continue;
            }
            mlir_types.push(self.mlir_type(field, immediate));
            offset = target_offset + field.size;
        }
        match mlir_types.len() {
            0 => self.type_empty(),
            1 => mlir_types[0],
            _ => self.type_tuple(&mlir_types),
            /*use crate::rustc_middle::query::Key;
            self.emit_error(
                "Does not support tuple with more than 1 elements".into(),
                self.tcx.def_span(layout.ty.def_id_for_ty_in_cycle().unwrap()),
            )*/
        }
    }

    fn pointer_to_mlir_type(
        &self,
        ty: &rustc_middle::ty::Ty<'tcx>,
        pair_idx: Option<usize>,
        _immediate: bool,
        memory_space: Option<Attribute<'ml>>,
    ) -> MLIRType<'ml> {
        match ty.kind() {
            rustc_middle::ty::TyKind::Str => self.type_memref_single(self.type_i8(), memory_space),
            rustc_middle::ty::TyKind::RawPtr(inner_type, _)
            | rustc_middle::ty::TyKind::Ref(_, inner_type, _)
            | rustc_middle::ty::TyKind::Slice(inner_type)
            | rustc_middle::ty::TyKind::Array(inner_type, _) => {
                let layout = self.layout_of(*inner_type);
                let bytes = layout.size.bytes() as i64;
                if inner_type.is_primitive() {
                    if bytes == 0 {
                        self.type_memref_single(self.type_i8(), memory_space)
                    } else {
                        self.type_memref(self.type_i8(), &[bytes], None, memory_space)
                    }
                } else {
                    let mut memory_space = memory_space;
                    if let Some(def) = inner_type.ty_adt_def() {
                        tracing::debug!("Using adt {:?} as scalar type", def);
                        if Some(def.did())
                            == self
                                .tcx
                                .get_diagnostic_item(rustc_span::Symbol::intern("gpu::GpuShared"))
                        {
                            memory_space = Some(MemorySpace::Shared.to_attr(self.mlir_ctx));
                        }
                    }
                    self.type_memref(self.type_i8(), &[bytes], None, memory_space)
                }
            }
            rustc_middle::ty::TyKind::Closure(_, _) => {
                let layout = self.layout_of(*ty);
                let bytes = layout.size.bytes() as i64;
                if bytes == 0 {
                    // Closure takes zero arguments
                    self.type_memref_single(self.type_i8(), memory_space)
                } else {
                    // Closure takes some arguments via pointer.
                    self.type_memref(self.type_i8(), &[bytes], None, memory_space)
                }
            }
            rustc_middle::ty::TyKind::Adt(def, substs) => {
                let layout = self.layout_of(*ty);
                let bytes = layout.size.bytes() as i64;
                if bytes == 0 {
                    self.type_memref_single(self.type_i8(), memory_space)
                } else {
                    self.type_memref(self.type_i8(), &[bytes], None, memory_space)
                }
            }
            rustc_middle::ty::TyKind::Tuple(list) => {
                let type_list = flat_tuple(ty);
                if let Some(idx) = pair_idx {
                    assert!(type_list.len() > idx);
                    self.pointer_to_mlir_type(&type_list[idx], None, _immediate, memory_space)
                } else {
                    assert!(!type_list.is_empty());
                    self.pointer_to_mlir_type(&type_list[0], None, _immediate, memory_space)
                }
            }
            rustc_middle::ty::TyKind::FnDef(def_id, substs) => {
                let instance = rustc_middle::ty::Instance::resolve_for_fn_ptr(
                    self.tcx,
                    self.typing_env(),
                    *def_id,
                    substs,
                )
                .unwrap();
                self.get_fn_addr(instance).r#type()
            }
            _ => {
                panic!("Unsupported pointer type: {:?}", ty);
            }
        }
    }

    pub(crate) fn cast_to_mlir_type(
        &self,
        cast: &rustc_target::callconv::CastTarget,
    ) -> MLIRType<'ml> {
        let unit_size = cast.rest.unit.size;
        let unit_ty = match cast.rest.unit.kind {
            rustc_abi::RegKind::Integer => {
                MLIRType::from(mlir_type::IntegerType::new(self.mlir_ctx, unit_size.bits() as u32))
            }
            rustc_abi::RegKind::Float => match unit_size.bytes() {
                16 => self.type_f16(),
                32 => self.type_f32(),
                64 => self.type_f64(),
                128 => self.type_f128(),
                _ => panic!("Unsupported float size: {}", unit_size.bytes()),
            },
            rustc_abi::RegKind::Vector => todo!("Vector cast is not supported"),
        };
        let len = cast.rest.total.bytes() / unit_size.bytes();
        assert!(len <= 1);
        unit_ty
    }

    fn scalar_mlir_type(
        &self,
        scalar: &rustc_abi::Scalar,
        layout: Option<&rustc_middle::ty::layout::TyAndLayout<'tcx>>,
        pair_idx: Option<usize>,
        immediate: bool,
        memory_space: Option<Attribute<'ml>>,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        if immediate && scalar.is_bool() {
            return self.type_i1();
        }
        match scalar.primitive() {
            Primitive::Int(i, _signed) => self.type_from_integer(i),
            Primitive::Float(f) => self.type_from_float(f),
            Primitive::Pointer(a) => {
                if let Some(ptr_ty) = layout.map(|l| &l.ty) {
                    debug!("Pointer type: {:?}", ptr_ty);
                    let ty = self.pointer_to_mlir_type(ptr_ty, pair_idx, immediate, memory_space);
                    assert!(ty.is_mem_ref());
                    ty
                } else {
                    self.type_memref_single(self.type_i8(), memory_space)
                }
            }
        }
    }

    pub(crate) fn type_to_mlir_type(
        &self,
        ty: &rustc_middle::ty::Ty<'tcx>,
        immediate: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        let layout = self.layout_of(*ty);
        self.mlir_type(layout, immediate)
    }

    pub(crate) fn mlir_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
        immediate: bool,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type {
        let cx = self.mlir_ctx;

        match layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                self.scalar_mlir_type(&scalar, Some(&layout), None, immediate, None)
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
        self.type_i_bits(8)
    }

    fn type_i16(&self) -> Self::Type {
        self.type_i_bits(16)
    }

    fn type_i32(&self) -> Self::Type {
        self.type_i_bits(32)
    }

    fn type_i64(&self) -> Self::Type {
        self.type_i_bits(64)
    }

    fn type_i128(&self) -> Self::Type {
        self.type_i_bits(128)
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
        if len <= 1 {
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
        use melior::ir::ValueLike;
        v.r#type()
    }

    fn type_ptr_ext(&self, address_space: rustc_abi::AddressSpace) -> Self::Type {
        self.type_memref_single(self.type_i8(), None)
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa_gpu::common::TypeKind {
        if ty.is_float() {
            rustc_codegen_ssa_gpu::common::TypeKind::Float
        } else if ty.is_integer() || ty.is_index() {
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
        assert!(ty.prefix == [None; 8]);
        assert!(ty.rest.unit.kind == rustc_abi::RegKind::Integer);
        let size = ty.rest.total;
        self.type_i_bits(size.bits() as u32)
    }

    fn fn_decl_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        self.fn_abi_to_fn_type(fn_abi, false, &[])
            .unwrap_or_else(|msg| {
                panic!();
            })
            .0
            .into()
    }

    fn fn_ptr_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        self.fn_abi_to_fn_type(fn_abi, false, &[]).unwrap().0.into()
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
        debug!("scalar_pair_element_backend_type: {:?} {:?} {:?}", a, b, scalar);
        if index == 1 {
            self.scalar_mlir_type(&scalar, Some(&layout), Some(index), immediate, None)
        } else {
            // Check if the type is adt gpu::GpuShared<T>, if so, it it a shared memory type.
            let memory_space = if let Some(def) = layout.ty.ty_adt_def() {
                if Some(def.did())
                    == self.tcx.get_diagnostic_item(rustc_span::Symbol::intern("gpu::GpuShared"))
                {
                    Some(MemorySpace::Shared.to_attr(self.mlir_ctx))
                } else {
                    None
                }
            } else {
                None
            };
            self.scalar_mlir_type(&scalar, Some(&layout), Some(index), immediate, memory_space)
        }
    }
}
