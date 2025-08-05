use melior::ir::BlockLike;
use rustc_abi::HasDataLayout;
use rustc_codegen_ssa_gpu::traits::{AbiBuilderMethods, ArgAbiBuilderMethods};
use rustc_middle::ty::layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers};
use rustc_target::callconv::PassMode;
use tracing::{trace, warn};

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
        trace!("get_param({})", index);
        if index >= self.cur_block.argument_count() {
            warn!("{}", self.cx.mlir_module.as_operation());
            panic!(
                "{:?} get_param({}) out of bounds at {:?}",
                self.cur_block.parent_operation(),
                index,
                self.cur_span
            );
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
        dst: rustc_codegen_ssa_gpu::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        if self.is_unreachable() {
            return;
        }
        warn!("store_fn_arg {:?} {} {:?} {:?}", arg_abi, idx, dst, self.cur_span);
        fn next<'ml, 'a>(
            bx: &GpuBuilder<'_, 'ml, 'a>,
            idx: &mut usize,
        ) -> melior::ir::Value<'ml, 'a> {
            if *idx >= bx.cur_block.argument_count() {
                dbg!(bx.cur_block().parent_operation());
                dbg!(*idx);
                dbg!(bx.cur_block.argument_count());
                panic!();
            }
            let val = bx.cur_block.argument(*idx).unwrap();
            *idx += 1;
            val.into()
        }
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) | PassMode::Cast { .. } => {
                self.store_arg(arg_abi, next(self, idx), dst);
            }
            PassMode::Pair(..) => {
                rustc_codegen_ssa_gpu::mir::operand::OperandValue::Pair(
                    next(self, idx),
                    next(self, idx),
                )
                .store(self, dst);
            }
            PassMode::Indirect { .. } => {
                panic!("query hooks should've made this `PassMode` impossible: {:#?}", arg_abi)
            }
        }
    }

    fn store_arg(
        &mut self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        val: Self::Value,
        dst: rustc_codegen_ssa_gpu::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        if self.is_unreachable() {
            return;
        }
        warn!("store_arg {:?} {} {:?}", arg_abi.mode, val, dst);
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) | PassMode::Pair(..) => {
                rustc_codegen_ssa_gpu::mir::operand::OperandRef::from_immediate_or_packed_pair(
                    self,
                    val,
                    arg_abi.layout,
                )
                .val
                .store(self, dst);
            }
            PassMode::Cast { .. } | PassMode::Indirect { .. } => {
                rustc_codegen_ssa_gpu::mir::operand::OperandRef::from_immediate_or_packed_pair(
                    self,
                    val,
                    arg_abi.layout,
                )
                .val
                .store(self, dst);
            }
        }
    }

    fn arg_memory_ty(
        &self,
        arg_abi: &rustc_target::callconv::ArgAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }
}
