use super::GpuBuilder;
use crate::rustc_codegen_ssa::traits::BaseTypeCodegenMethods;
use crate::rustc_codegen_ssa::traits::BuilderMethods;
use melior::ir::TypeLike;
use melior::{
    ir::{Location, Value, ValueLike},
    Context,
};

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub(crate) fn memref_to_llvm_ptr(&mut self, memref: Value<'ml, '_>) -> Value<'ml, 'a> {
        let op = melior::dialect::ods::memref::extract_aligned_pointer_as_index(
            self.cx.mlir_ctx,
            self.type_index(),
            memref,
            self.cur_loc(),
        )
        .into();
        let index = self.append_op_res(op);
        self.inttoptr(index, self.type_ptr())
    }

    pub(crate) fn val_to_llvm_value(&mut self, value: Value<'ml, 'a>) -> Value<'ml, 'a> {
        let ty = value.r#type();
        if ty.is_mem_ref() {
            self.memref_to_llvm_ptr(value)
        } else {
            value
        }
    }
}
