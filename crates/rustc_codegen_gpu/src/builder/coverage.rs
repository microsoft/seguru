use rustc_codegen_ssa_gpu::traits::CoverageInfoBuilderMethods;
use tracing::trace;

use super::GpuBuilder;

impl<'cx, 'tcx, 'ml, 'a> CoverageInfoBuilderMethods<'tcx> for GpuBuilder<'cx, 'tcx, 'ml, 'a> {
    fn add_coverage(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        kind: &rustc_middle::mir::coverage::CoverageKind,
    ) {
        trace!("add_coverage {:?} {:?}", instance, kind);
    }
}
