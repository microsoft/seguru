use melior::ir::BlockLike;
use rustc_abi::HasDataLayout;
use rustc_codegen_ssa::traits::{AbiBuilderMethods, ArgAbiBuilderMethods};
use rustc_middle::ty::layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers};
use rustc_target::callconv::PassMode;

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
        log::trace!("get_param({})", index);
        if index >= self.cur_block.argument_count() {
            panic!("index out of bounds");
        }
        let val = self.cur_block.argument(index).unwrap();
        Self::Value::from(val)
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
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) | PassMode::Pair(..) => {
                rustc_codegen_ssa::mir::operand::OperandRef::from_immediate_or_packed_pair(
                    self,
                    val,
                    arg_abi.layout,
                )
                .val
                .store(self, dst);
            }
            PassMode::Cast { .. } | PassMode::Indirect { .. } => todo!(),
        }
    }

    fn arg_memory_ty(
        &self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }
}
