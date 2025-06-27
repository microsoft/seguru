use melior::dialect::memref as mlir_memref;
use melior::helpers::ArithBlockExt;
use melior::ir::{self as mlir_ir, BlockLike, r#type as mlir_type};
use rustc_abi::Size;
use rustc_codegen_ssa_gpu::traits::{
    BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_const_eval::interpret::{GlobalAlloc, alloc_range};
use tracing::{debug, trace};

use super::GPUCodegenContext;

fn get_alloc_name(alloc_id: rustc_const_eval::interpret::AllocId) -> String {
    format!("memory_alloc_{:?}", alloc_id.0)
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub(crate) fn mlir_const_val_from_type(
        &self,
        i: impl std::fmt::Display,
        typ: <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type,
        block: &'a mlir_ir::Block<'ml>,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        block
            .const_int_from_type(self.mlir_ctx, self.unknown_loc(), i, typ)
            .expect("failed to create const int")
    }

    fn mlir_global_const_int_from_type(
        &self,
        i: impl std::fmt::Display,
        typ: <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        self.mlir_const_val_from_type(i, typ, self.mlir_body(false))
    }

    pub(crate) fn const_data_memref_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
        name: &str,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        assert!(alloc.inner().size().bytes() > 0);
        let ty = self.type_array(self.type_i8(), alloc.inner().len() as u64);
        let ref_ty =
            mlir_type::MemRefType::new(self.type_i8(), &[alloc.inner().len() as i64], None, None);
        let bytes =
            alloc.inner().get_bytes_unchecked(alloc_range(Size::ZERO, alloc.inner().size()));
        debug!(
            "const_data_memref_from_alloc: {} {} {:?} {}",
            alloc.inner().len(),
            alloc.inner().size().bytes(),
            bytes,
            ty
        );
        let value = mlir_ir::attribute::DenseElementsAttribute::new(
            ty,
            &bytes
                .iter()
                .map(|v| {
                    mlir_ir::attribute::IntegerAttribute::new(self.type_i8(), *v as i64).into()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .into();
        let op = mlir_memref::global(
            self.mlir_ctx,
            name,
            None,
            ref_ty,
            Some(value),
            true,
            None,
            self.unknown_loc(),
        );
        let op = self.mlir_body(true).append_operation(op);
        let op = mlir_memref::get_global(self.mlir_ctx, name, ref_ty, self.unknown_loc());
        let op = self.mlir_body(true).append_operation(op);
        op.result(0).unwrap().into()
    }

    pub(crate) fn const_data_memref_from_alloc_id(
        &self,
        alloc_id: rustc_const_eval::interpret::AllocId,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        let alloc = self.tcx.global_alloc(alloc_id);
        {
            let const_alloc = self.const_alloc.read().unwrap();
            if const_alloc.contains_key(&alloc_id) {
                return const_alloc[&alloc_id];
            }
        }
        let name = get_alloc_name(alloc_id);
        let v = match alloc {
            GlobalAlloc::Memory(alloc) => {
                self.const_name_to_allocid.write().unwrap().insert(name.clone(), alloc_id);
                self.const_data_memref_from_alloc(alloc, name.as_str())
            }
            GlobalAlloc::VTable(ty, dyn_ty) => {
                let alloc_id = self.tcx.vtable_allocation((
                    ty,
                    dyn_ty
                        .principal()
                        .map(|principal| self.tcx.instantiate_bound_regions_with_erased(principal)),
                ));
                self.const_data_memref_from_alloc_id(alloc_id)
            }
            _ => todo!(),
        };
        self.const_alloc.write().unwrap().insert(alloc_id, v);
        trace!("const_data_memref_from_alloc_id: {:?}", v);
        v
    }
}
impl<'tcx, 'ml, 'a> ConstCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn const_null(&self, t: Self::Type) -> Self::Value {
        self.const_int(self.type_i64(), 0)
    }

    fn const_undef(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        let op =
            crate::mlir::poison::const_poison(self.mlir_ctx, self.type_i64(), self.unknown_loc());
        let op = self.mlir_body(true).append_operation(op);
        op.result(0).unwrap().into()
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        todo!()
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        self.mlir_global_const_int_from_type(i, t)
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_index())
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i, t)
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        self.mlir_global_const_int_from_type(u, t)
    }

    fn const_real(&self, t: Self::Type, val: f64) -> Self::Value {
        todo!()
    }

    fn const_str(&self, s: &str) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn const_struct(&self, elts: &[Self::Value], packed: bool) -> Self::Value {
        todo!()
    }

    fn const_vector(&self, elts: &[Self::Value]) -> Self::Value {
        todo!()
    }

    fn const_to_opt_uint(&self, v: Self::Value) -> Option<u64> {
        self.const_to_opt_u128(v, false).map(|v| v as u64)
    }

    fn const_to_opt_u128(&self, v: Self::Value, sign_ext: bool) -> Option<u128> {
        dbg!(v);
        crate::mlir::mlir_val_to_const_int(v)
    }

    fn const_data_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
    ) -> Self::Value {
        todo!();
    }

    fn scalar_to_backend(
        &self,
        cv: rustc_const_eval::interpret::Scalar,
        layout: rustc_abi::Scalar,
        ty: Self::Type,
    ) -> Self::Value {
        match cv {
            rustc_const_eval::interpret::Scalar::Int(int) => {
                assert_eq!(int.size(), layout.primitive().size(self));
                let data = int.to_uint(int.size());

                if let rustc_abi::Primitive::Pointer(_) = layout.primitive() {
                    if data == 0 { self.const_null(ty) } else { self.const_undef(ty) }
                } else {
                    self.mlir_global_const_int_from_type(
                        cv.assert_scalar_int().to_int(int.size()),
                        ty,
                    )
                }
            }
            rustc_const_eval::interpret::Scalar::Ptr(ptr, s) => {
                let (prov, offset) = ptr.into_parts();
                let alloc_id = prov.alloc_id();
                trace!("scalar_to_backend ptr: {:?}", self.tcx.global_alloc(alloc_id));
                self.const_data_memref_from_alloc_id(alloc_id)
            }
        }
    }

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: rustc_abi::Size) -> Self::Value {
        todo!()
    }
}

impl<'tcx, 'ml, 'a> StaticCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn static_addr_of(
        &self,
        cv: Self::Value,
        align: rustc_abi::Align,
        kind: Option<&str>,
    ) -> Self::Value {
        todo!()
    }

    fn codegen_static(&self, def_id: rustc_hir::def_id::DefId) {
        todo!()
    }

    fn add_used_global(&self, global: Self::Value) {
        todo!()
    }

    fn add_compiler_used_global(&self, global: Self::Value) {
        todo!()
    }
}
