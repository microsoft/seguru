use rustc_codegen_ssa::traits::IntrinsicCallBuilderMethods;

use super::GpuBuilder;

impl<'tcx, 'ml, 'a> IntrinsicCallBuilderMethods<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        args: &[rustc_codegen_ssa::mir::operand::OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: rustc_span::Span,
    ) -> Result<(), rustc_middle::ty::Instance<'tcx>> {
        todo!()
    }

    fn abort(&mut self) {
        todo!()
    }

    fn assume(&mut self, val: Self::Value) {
        todo!()
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        todo!()
    }

    fn type_test(&mut self, pointer: Self::Value, typeid: Self::Metadata) -> Self::Value {
        todo!()
    }

    fn type_checked_load(
        &mut self,
        llvtable: Self::Value,
        vtable_byte_offset: u64,
        typeid: Self::Metadata,
    ) -> Self::Value {
        todo!()
    }

    fn va_start(&mut self, val: Self::Value) -> Self::Value {
        todo!()
    }

    fn va_end(&mut self, val: Self::Value) -> Self::Value {
        todo!()
    }
}
