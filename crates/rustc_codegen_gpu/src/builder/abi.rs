use rustc_abi::HasDataLayout;
use rustc_codegen_ssa::traits::{AbiBuilderMethods, ArgAbiBuilderMethods};
use rustc_middle::ty::layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers};

use super::GpuBuilder;

impl<'tcx, 'ml, 'a> HasDataLayout for GpuBuilder<'tcx, 'ml, 'a> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'tcx, 'ml, 'a> HasTypingEnv<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn typing_env(&self) -> rustc_middle::ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl<'tcx, 'ml, 'a> HasTyCtxt<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.cx.tcx()
    }
}

impl<'tcx, 'ml, 'a> LayoutOfHelpers<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
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

impl<'tcx, 'ml, 'a> FnAbiOfHelpers<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
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

impl<'tcx, 'ml, 'a> AbiBuilderMethods for GpuBuilder<'tcx, 'ml, 'a> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        let cur_fn = self.cur_operation().unwrap();
        log::trace!("get_param({})", index);
        if index >= cur_fn.operand_count() {
            panic!("index out of bounds");
        }
        let val = cur_fn.operand(index);
        val.expect("failed to get params")
    }
}

impl<'tcx, 'ml, 'a> ArgAbiBuilderMethods<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        idx: &mut usize,
        dst: rustc_codegen_ssa::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        todo!()
    }

    fn store_arg(
        &mut self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        val: Self::Value,
        dst: rustc_codegen_ssa::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        todo!()
    }

    fn arg_memory_ty(
        &self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }
}
