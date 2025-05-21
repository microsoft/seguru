use rustc_codegen_ssa::traits::CoverageInfoBuilderMethods;

use super::GpuBuilder;

impl<'tcx, 'ml, 'a> CoverageInfoBuilderMethods<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn add_coverage(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        kind: &rustc_middle::mir::coverage::CoverageKind,
    ) {
        log::trace!("add_coverage {:?} {:?}", instance, kind);
    }
}
