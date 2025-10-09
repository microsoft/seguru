use melior::ir::r#type::MemRefType;
use melior::ir::{self as mlir_ir, ShapedTypeLike, ValueLike};
use rustc_codegen_ssa_gpu::traits::{BaseTypeCodegenMethods, BuilderMethods};

use crate::builder::GpuBuilder;

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    /// In MLIR, MemRef<N*f32> will not be treated as a vector type.
    /// This function converts a memref<N*f32> to memref<1xvec<Nxf32>> to use optimized behavior for vectors.
    /// For example, load/store of memref<1xvec<4*f32>> will become ld.v4.f32 instead of executing ld.f32 for 4 times
    pub(crate) fn use_memref_as_vector_memref(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
        vec_ty: melior::ir::Type<'ml>,
    ) -> melior::ir::Value<'ml, 'a> {
        let memref_ty = MemRefType::try_from(val.r#type()).unwrap();
        self.use_value_as_ty(val, self.type_memref(vec_ty, &[1], None, memref_ty.memory_space()))
    }

    pub(crate) fn load_vector(
        &mut self,
        memref: mlir_ir::Value<'ml, 'a>,
        index: mlir_ir::Value<'ml, 'a>,
        vector_ty: mlir_ir::r#type::Type<'ml>,
    ) -> mlir_ir::Value<'ml, 'a> {
        let loc = self.cur_loc();
        let memref = self.use_memref_as_vector_memref(memref, vector_ty);
        let op =
            melior::dialect::ods::vector::load(self.mlir_ctx, vector_ty, memref, &[index], loc);
        self.append_op_res(op.into())
    }

    pub(crate) fn store_vector(
        &mut self,
        memref: mlir_ir::Value<'ml, 'a>,
        index: mlir_ir::Value<'ml, 'a>,
        invec: mlir_ir::Value<'ml, 'a>,
    ) -> mlir_ir::OperationRef<'ml, 'a> {
        let loc = self.cur_loc();
        let vector_ty = invec.r#type();
        let memref = self.use_memref_as_vector_memref(memref, vector_ty);
        let op = melior::dialect::ods::vector::store(self.mlir_ctx, invec, memref, &[index], loc);
        self.append_op(op.into())
    }

    fn scalar_memcpy(
        &mut self,
        dst: mlir_ir::Value<'ml, 'a>,
        dst_align: rustc_abi::Align,
        src: mlir_ir::Value<'ml, 'a>,
        src_align: rustc_abi::Align,
        size: mlir_ir::Value<'ml, 'a>,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) {
        let dst_ty = MemRefType::try_from(dst.r#type()).unwrap();
        let src_ty = MemRefType::try_from(src.r#type()).unwrap();
        let const_size = crate::mlir::mlir_val_to_const_int(size);
        let (dst_ty, dynamic_size) =
            if let Some(const_size) = crate::mlir::mlir_val_to_const_int(size) {
                (
                    self.type_memref(
                        dst_ty.element(),
                        &[const_size as i64 / self.static_size_of(dst_ty.element()) as i64],
                        None,
                        dst_ty.memory_space(),
                    ),
                    None,
                )
            } else {
                (
                    self.type_memref(
                        dst_ty.element(),
                        &[crate::mlir::memref::dynamic_size()],
                        None,
                        dst_ty.memory_space(),
                    ),
                    Some(size),
                )
            };
        let src = self.mlir_memref_view(src, dst_ty, None, dynamic_size);
        let dst = self.mlir_memref_view(dst, dst_ty, None, dynamic_size);
        self.append_op(
            melior::dialect::ods::memref::copy(self.mlir_ctx, src, dst, self.cur_loc()).into(),
        );
    }

    pub(crate) fn vector_memcpy(
        &mut self,
        dst: mlir_ir::Value<'ml, 'a>,
        dst_align: rustc_abi::Align,
        src: mlir_ir::Value<'ml, 'a>,
        src_align: rustc_abi::Align,
        size: mlir_ir::Value<'ml, 'a>,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) {
        let dst_ty = MemRefType::try_from(dst.r#type()).unwrap();
        let src_ty = MemRefType::try_from(src.r#type()).unwrap();
        // If both src and dst are in local memory, we do scalar copy
        if dst_ty.memory_space() == Some(self.local_mem_space())
            && src_ty.memory_space() == Some(self.local_mem_space())
        {
            self.scalar_memcpy(dst, dst_align, src, src_align, size, flags);
            return;
        }
        let const_size = crate::mlir::mlir_val_to_const_int(size);
        if let Some(const_size) = crate::mlir::mlir_val_to_const_int(size) {
            let const_size = const_size as u64;
            let vec_ty = if const_size % 4 == 0 {
                self.type_vector(&[const_size / 4], self.type_i32())
            } else if const_size % 2 == 0 {
                self.type_vector(&[const_size / 2], self.type_i16())
            } else {
                self.type_vector(&[const_size], self.type_i8())
            };
            let zero = self.const_value(0, self.type_index());
            let src = self.use_memref_as_vector_memref(src, vec_ty);
            let dst_memref_ty = self.type_memref(
                dst_ty.element(),
                &[const_size as i64 / self.static_size_of(dst_ty.element()) as i64],
                None,
                src_ty.memory_space(),
            );
            //let dst = self.use_value_as_ty(dst, dst_memref_ty);
            let dst = self.use_memref_as_vector_memref(dst, vec_ty);
            let src_vec = self.load_vector(src, zero, vec_ty);
            self.store_vector(dst, zero, src_vec);
            return;
        }
        assert!(src_align.bytes() == dst_align.bytes());
        let vec_ty = if src_align.bytes() % 4 == 0 {
            self.type_vector(&[src_align.bytes() / 4], self.type_i32())
        } else if src_align.bytes() % 2 == 0 {
            self.type_vector(&[src_align.bytes() / 2], self.type_i16())
        } else {
            self.type_vector(&[src_align.bytes()], self.type_i8())
        };
        let align = self.const_value(src_align.bytes(), self.type_index());
        let len = self.udiv(size, align);
        let src = self.mlir_memref_view(src, vec_ty, None, Some(len));
        let dst = self.mlir_memref_view(dst, vec_ty, None, Some(len));
        let op = self.append_op(crate::mlir::linalg::linalg_copy_op(
            self.mlir_ctx,
            src,
            dst,
            vec_ty,
            self.cur_loc(),
        ));
    }
}
