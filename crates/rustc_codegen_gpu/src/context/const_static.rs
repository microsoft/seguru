use melior::dialect::memref as mlir_memref;
use melior::helpers::ArithBlockExt;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationMutLike;
use melior::ir::{self as mlir_ir, BlockLike, Location, r#type as mlir_type};
use rustc_abi::{HasDataLayout, Size};
use rustc_codegen_ssa_gpu::traits::{
    BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_const_eval::interpret::{GlobalAlloc, PointerArithmetic, alloc_range};
use tracing::trace;

use super::GPUCodegenContext;

fn get_alloc_name(alloc_id: rustc_const_eval::interpret::AllocId) -> String {
    format!("memory_alloc_{:?}", alloc_id.0)
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    fn get_name_by_alloc(
        &self,
        alloc: &rustc_const_eval::interpret::ConstAllocation<'_>,
    ) -> String {
        format!(
            "const_alloc_{:?}",
            self.const_alloc_count_no_id.load(std::sync::atomic::Ordering::SeqCst)
        )
    }

    fn append_op_outside(
        &self,
        mut op: mlir_ir::Operation<'ml>,
        to_remove_hint: &str,
    ) -> mlir_ir::OperationRef<'ml, 'a> {
        op.set_attribute("to_remove", StringAttribute::new(self.mlir_ctx, to_remove_hint).into());
        self.mlir_body(true).append_operation(op)
    }
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

    pub(crate) fn define_static_shared_mem(
        &self,
        size: rustc_abi::Size,
        align: rustc_abi::Align,
        loc: Location<'ml>,
    ) -> String {
        let ret_final_type = self.type_shared_memref(self.type_i8(), &[size.bytes() as i64], None);
        let idx = self.static_shared_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let name = format!("static_shared_{}", idx);
        let val_ty = self.type_array(self.type_i8(), size.bytes());
        let value = mlir_ir::attribute::DenseElementsAttribute::new(
            val_ty,
            &vec![0; size.bytes() as usize]
                .iter()
                .map(|v| {
                    mlir_ir::attribute::IntegerAttribute::new(self.type_i8(), *v as i64).into()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .into();
        let static_shared = mlir_memref::global(
            self.mlir_ctx,
            &name,
            None,
            ret_final_type,
            Some(value),
            false,
            Some(self.align_to_attr(align)),
            loc,
        );
        self.mlir_body(true).append_operation(static_shared);
        name
    }

    pub(crate) fn const_data_memref_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
        name: &str,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        let dl = self.data_layout();
        let pointer_size = dl.pointer_size();
        let mem_len = pointer_size.bytes() as usize;
        let ty = self.type_array(self.type_i8(), mem_len as _);
        let value = if alloc.inner().size().bytes() == 0 {
            None
        } else {
            let bytes =
                alloc.inner().get_bytes_unchecked(alloc_range(Size::ZERO, alloc.inner().size()));
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
            Some(value)
        };
        self.static_data_memref(value, name)
    }

    fn static_data_memref(
        &self,
        value: Option<mlir_ir::Attribute<'ml>>,
        name: &str,
    ) -> <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value {
        let dl = self.data_layout();
        let pointer_size = dl.pointer_size();
        let align = dl.pointer_align.pref;
        let mem_len = pointer_size.bytes() as usize;
        let ty = self.type_array(self.type_i8(), mem_len as _);
        let ref_ty = mlir_type::MemRefType::new(self.type_i8(), &[mem_len as _], None, None);
        let op = mlir_memref::global(
            self.mlir_ctx,
            name,
            None,
            ref_ty,
            value,
            true,
            Some(self.align_to_attr(align)),
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
            GlobalAlloc::Static(def_id) => {
                //let instance = Instance::mono(self.tcx, def_id);
                //use crate::rustc_middle::ty::layout::HasTypingEnv;
                //let ty = instance.ty(self.tcx, self.typing_env());
                //let llty = self.mlir_type(self.layout_of(ty), false);
                self.static_data_memref(None, &format!("static_{}", self.tcx.def_path_str(def_id)))
            }
            _ => todo!("{:?}", alloc),
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
        self.const_poison(t)
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        let op = crate::mlir::poison::const_poison(self.mlir_ctx, t, self.unknown_loc());
        let op = self.mlir_body(true).append_operation(op);
        op.result(0).unwrap().into()
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        self.mlir_global_const_int_from_type(val as i64, self.type_i1())
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_i8())
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_i16())
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        self.mlir_global_const_int_from_type(i, self.type_i32())
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        self.mlir_global_const_int_from_type(i, t)
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i8, self.type_i8())
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i32, self.type_i32())
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i64, self.type_i64())
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i128, self.type_i128())
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i64, self.type_i64())
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        self.mlir_global_const_int_from_type(i as i64, t)
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        self.mlir_global_const_int_from_type(u as i128, t)
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
        crate::mlir::mlir_val_to_const_int(v)
    }

    fn const_data_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
    ) -> Self::Value {
        let ret = self.const_data_memref_from_alloc(alloc, &self.get_name_by_alloc(&alloc));
        self.const_alloc_count_no_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ret
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
        let op = crate::mlir::poison::const_ptr_offset(
            self.mlir_ctx,
            val,
            offset.bytes() as usize,
            self.unknown_loc(),
        );
        self.append_op_outside(op, "const_offset").result(0).unwrap().into()
    }
}

impl<'tcx, 'ml, 'a> StaticCodegenMethods for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn static_addr_of(
        &self,
        cv: Self::Value,
        align: rustc_abi::Align,
        kind: Option<&str>,
    ) -> Self::Value {
        cv
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
