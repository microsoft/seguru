use melior::dialect::ods::{arith as melior_arith, math as melior_math};
use melior::ir::Value;
use melior::ir::attribute::DenseI32ArrayAttribute;
use rustc_codegen_ssa_gpu::traits::{BuilderMethods, IntrinsicCallBuilderMethods};
use rustc_middle::ty::layout::HasTypingEnv;

use super::GpuBuilder;

macro_rules! call_intrinsic {
    (1, $func:path, $args:expr, $ctx:expr, $loc:expr) => {
        $func($ctx, $args[0], $loc)
    };
    (2, $func:path, $args:expr, $ctx:expr, $loc:expr) => {
        $func($ctx, $args[0], $args[1], $loc)
    };
    (3, $func:path, $args:expr, $ctx:expr, $loc:expr) => {
        $func($ctx, $args[0], $args[1], $args[2], $loc)
    };
}

macro_rules! device_intrinsic_match {
    ($name:expr, $args:expr, $builder:expr, $ctx:expr, $loc:expr,
    {$($intrinsic:literal => $func:path, $num_args:tt),* $(,)?}
    ) => {
        match $name {
            $(
                $intrinsic => {
                    if $args.len() != $num_args {
                        panic!("Intrinsic `{}` expects {} args, got {}", $intrinsic, $num_args, $args.len());
                    }
                    let op = call_intrinsic!($num_args, $func, $args, $ctx, $loc);
                    $builder.append_op(op.into())
                }
            )*
            _ => panic!("GPU intrinsic `{}` not supported.", $name),
        }
    };
}

macro_rules! intrinsic_match {
    ($name:expr, $llresult:expr, $args:expr, $builder:expr, $ctx:expr, $loc:expr,
    {$($intrinsic:path => $func:path, $num_args:tt),* $(,)?}
    ) => {
        match $name {
            $(
                $intrinsic => {
                    if $args.len() != $num_args {
                        panic!("Intrinsic `{}` expects {} args, got {}", $intrinsic, $num_args, $args.len());
                    }
                    let op = call_intrinsic!($num_args, $func, $args, $ctx, $loc);
                    let ret = $builder.append_op_res(op.into());
                    $builder.store(ret, $llresult, rustc_abi::Align::ONE);
                }
            )*
            _ => panic!("GPU intrinsic `{}` not supported.", $name),
        }
    };
}

pub(crate) fn device_intrinsic<'tcx, 'ml, 'a>(
    builder: &mut GpuBuilder<'tcx, 'ml, 'a>,
    name: &str,
    args: &[Value<'ml, 'a>],
    loc: melior::ir::Location<'ml>,
) -> melior::ir::OperationRef<'ml, 'a> {
    let intrinsic_name = format!("gpu::device_intrinsics::{}", name);
    let mlir_ctx = builder.mlir_ctx;
    device_intrinsic_match! {name, args, builder, mlir_ctx, loc,
        {
            "fma" => melior_math::fma, 3,
            "max" => melior_arith::maxnumf, 2,
            "min" => melior_arith::minnumf, 2,
            "sqrt" => melior_math::sqrt, 1,
            "rsqrt" => melior_math::rsqrt, 1,
            "ceil" => melior_math::ceil, 1,
            "exp" => melior_math::exp, 1,
            "exp2" => melior_math::exp_2, 1,
            "expm1" => melior_math::expm_1, 1,
            "sin" => melior_math::sin, 1,
            "sinh" => melior_math::sinh, 1,
            "tan" => melior_math::tan, 1,
            "tanh" => melior_math::tanh, 1,
            "cos" => melior_math::cos, 1,
            "cosh" => melior_math::cosh, 1,
            "pow" => melior_math::powf, 2,
            "log" => melior_math::log, 1,
            "log10" => melior_math::log_10, 1,
            "log1p" => melior_math::log_1_p, 1,
            "log2" => melior_math::log_2, 1,
            // Add more intrinsics as needed
        }
    }
}

impl<'tcx, 'ml, 'a> IntrinsicCallBuilderMethods<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        args: &[rustc_codegen_ssa_gpu::mir::operand::OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: rustc_span::Span,
    ) -> Result<(), rustc_middle::ty::Instance<'tcx>> {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let rustc_middle::ty::FnDef(def_id, fn_args) = *callee_ty.kind() else {
            panic!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let name = tcx.item_name(def_id);
        let loc = self.to_mlir_loc(span);
        let ctx = self.mlir_ctx;
        let args_imm = args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>();
        use rustc_span::sym;
        intrinsic_match! {name, llresult, args_imm, self, ctx, loc,
            {
                sym::sinf32 => melior_math::sin, 1,
                sym::sinf64 => melior_math::sin, 1,
                sym::cosf32 => melior_math::cos, 1,
                sym::cosf64 => melior_math::cos, 1,
                sym::sqrtf32 => melior_math::sqrt, 1,
                sym::sqrtf64 => melior_math::sqrt, 1,
                sym::ceilf32 => melior_math::ceil, 1,
                sym::ceilf64 => melior_math::ceil, 1,
                sym::expf32 => melior_math::exp, 1,
                sym::expf64 => melior_math::exp, 1,
                sym::exp2f32 => melior_math::exp_2, 1,
                sym::exp2f64 => melior_math::exp_2, 1,
                sym::log10f32 => melior_math::log_10, 1,
                sym::log10f64 => melior_math::log_10, 1,
                sym::logf32 => melior_math::log, 1,
                sym::logf64 => melior_math::log, 1,
                sym::log2f32 => melior_math::log_2, 1,
                sym::log2f64 => melior_math::log_2, 1,
            }
        }
        Ok(())
    }

    fn abort(&mut self) {
        self.append_op(melior::dialect::ods::llvm::intr_trap(self.mlir_ctx, self.cur_loc()).into());
    }

    fn assume(&mut self, val: Self::Value) {
        self.append_op(
            melior::dialect::ods::llvm::intr_assume(
                self.mlir_ctx,
                val,
                &[],
                DenseI32ArrayAttribute::new(self.mlir_ctx, &[]),
                self.cur_loc(),
            )
            .into(),
        );
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        cond
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
