use rustc_middle::ty::layout::{FnAbiOfHelpers, LayoutOfHelpers};

use super::GPUCodegenContext;

impl<'tcx, 'ml, 'a> LayoutOfHelpers<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    type LayoutOfResult = rustc_middle::ty::layout::TyAndLayout<'tcx>;

    fn handle_layout_err(
        &self,
        err: rustc_middle::ty::layout::LayoutError<'tcx>,
        span: rustc_span::Span,
        ty: rustc_middle::ty::Ty<'tcx>,
    ) -> <Self::LayoutOfResult as rustc_middle::ty::layout::MaybeResult<
        rustc_middle::ty::layout::TyAndLayout<'tcx>,
    >>::Error {
        todo!()
    }
}

impl<'tcx, 'ml, 'a> FnAbiOfHelpers<'tcx> for GPUCodegenContext<'tcx, 'ml, 'a> {
    type FnAbiOfResult = &'tcx rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: rustc_middle::ty::layout::FnAbiError<'tcx>,
        span: rustc_span::Span,
        fn_abi_request: rustc_middle::ty::layout::FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as rustc_middle::ty::layout::MaybeResult<
        &'tcx rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    >>::Error {
        todo!()
    }
}
