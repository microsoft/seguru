use melior::dialect::memref as mlir_memref;
use melior::ir::{self as mlir_ir, r#type as mlir_type};
use melior::{dialect::ods::memref::GlobalOperation, helpers::ArithBlockExt, ir::BlockLike};
use rustc_abi::Size;
use rustc_codegen_ssa::traits::{
    BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_const_eval::interpret::{alloc_range, AllocRange, GlobalAlloc};

use crate::mlir;

use super::GPUCodegenContext;
impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn mlir_const_int_from_type(
        &self,
        i: impl std::fmt::Display,
        typ: <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        self.mlir_body()
            .const_int_from_type(self.mlir_ctx, self.unknown_loc(), i, typ)
            .expect("failed to create const int")
    }

    pub fn mlir_const_int<T>(
        &self,
        i: impl std::fmt::Display,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        self.mlir_body()
            .const_int(
                self.mlir_ctx,
                self.unknown_loc(),
                i,
                size_of::<T>() as u32 * 8,
            )
            .expect("failed to create const int")
    }

    fn const_data_memref_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        let name = GlobalOperation::name();
        let ty =
            mlir_type::MemRefType::new(self.type_i8(), &[alloc.inner().len() as i64], None, None);
        let bytes = alloc
            .inner()
            .get_bytes_unchecked(alloc_range(Size::ZERO, alloc.inner().size()));
        log::trace!("const_data_memref_from_alloc: {:?}", bytes);

        let value = mlir_ir::attribute::ArrayAttribute::new(
            self.mlir_ctx,
            &bytes
                .iter()
                .map(|v| {
                    mlir_ir::attribute::IntegerAttribute::new(self.type_i8(), *v as i64).into()
                })
                .collect::<Vec<_>>(),
        )
        .into();
        let op = mlir_memref::global(
            self.mlir_ctx,
            name,
            None,
            ty,
            Some(value),
            true,
            None,
            self.unknown_loc(),
        );
        let op = self.mlir_body().append_operation(op);
        let op = mlir_memref::get_global(self.mlir_ctx, name, ty, self.unknown_loc());
        let op = self.mlir_body().append_operation(op);
        op.result(0).unwrap().into()
    }

    fn const_data_memref_from_alloc_id(
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
        let v = match alloc {
            GlobalAlloc::Memory(alloc) => self.const_data_memref_from_alloc(alloc),
            GlobalAlloc::VTable(ty, dyn_ty) => {
                let alloc = self
                    .tcx
                    .global_alloc(self.tcx.vtable_allocation((
                        ty,
                        dyn_ty.principal().map(|principal| {
                            self.tcx.instantiate_bound_regions_with_erased(principal)
                        }),
                    )))
                    .unwrap_memory();
                self.const_data_memref_from_alloc(alloc)
            }
            _ => todo!(),
        };
        self.const_alloc.write().unwrap().insert(alloc_id, v);
        log::trace!("const_data_memref_from_alloc_id: {:?}", v);
        v
    }
}
impl<'tcx, 'ml, 'a> ConstCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn const_null(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_undef(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        todo!()
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        todo!()
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        todo!()
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        todo!()
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        todo!()
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        todo!()
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        todo!()
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        todo!()
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        todo!()
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        self.mlir_const_int::<usize>(i)
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        self.mlir_const_int_from_type(i, t)
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        self.mlir_const_int_from_type(u, t)
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
        todo!()
    }

    fn const_to_opt_u128(&self, v: Self::Value, sign_ext: bool) -> Option<u128> {
        todo!()
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
                    if data == 0 {
                        self.const_null(ty)
                    } else {
                        let result = self.const_undef(ty);
                        result
                    }
                } else {
                    self.mlir_const_int_from_type(cv.assert_scalar_int().to_u128(), ty)
                }
            }
            rustc_const_eval::interpret::Scalar::Ptr(ptr, s) => {
                let (prov, offset) = ptr.into_parts();
                let alloc_id = prov.alloc_id();
                log::trace!(
                    "scalar_to_backend ptr: {:?}",
                    self.tcx.global_alloc(alloc_id)
                );
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
