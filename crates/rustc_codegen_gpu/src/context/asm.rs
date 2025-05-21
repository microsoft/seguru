use rustc_codegen_ssa::traits::AsmCodegenMethods;

use super::GPUCodegenContext;

impl<'tcx, 'ml, 'a> AsmCodegenMethods<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    fn codegen_global_asm(
        &self,
        template: &[rustc_ast::InlineAsmTemplatePiece],
        operands: &[rustc_codegen_ssa::traits::GlobalAsmOperandRef<'tcx>],
        options: rustc_ast::InlineAsmOptions,
        line_spans: &[rustc_span::Span],
    ) {
        todo!()
    }

    fn mangled_name(&self, instance: rustc_middle::ty::Instance<'tcx>) -> String {
        todo!()
    }
}
