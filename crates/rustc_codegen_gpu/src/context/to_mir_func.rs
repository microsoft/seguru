use melior::ir::{BlockLike, Operation};
use rustc_codegen_ssa::traits::LayoutTypeCodegenMethods;
use rustc_middle::{
    query::Key,
    ty::{
        layout::{FnAbiOf, HasTyCtxt, HasTypingEnv},
        Instance,
    },
};

use crate::mlir::{MLIROpHelpers, MLIRVisibility};

use super::GPUCodegenContext;

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn to_mir_func_decl(
        &self,
        instance: Instance<'tcx>,
        visibility: MLIRVisibility,
    ) -> melior::ir::OperationRef<'ml, 'a> {
        let tcx = self.tcx();
        let mlir_ctx = self.mlir_ctx;
        let sym = tcx.symbol_name(instance).name.to_string();
        let def_id: rustc_hir::def_id::DefId = instance.def_id();
        log::trace!(
            "get_fn({:?}: {:?}) => {}",
            instance,
            instance.ty(tcx, self.typing_env()),
            sym
        );
        let span = instance.def.default_span(tcx);
        let location: melior::ir::Location<'ml> = self.to_mlir_loc(span);
        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let mut args = vec![];
        for arg in &fn_abi.args {
            let mlir_arg = match &arg.mode {
                rustc_target::callconv::PassMode::Ignore => {
                    continue;
                }
                rustc_target::callconv::PassMode::Direct(arg_attributes) => {
                    self.immediate_backend_type(arg.layout)
                }
                rustc_target::callconv::PassMode::Pair(arg_attributes, arg_attributes1) => {
                    let mlir_arg = self.scalar_pair_element_backend_type(arg.layout, 0, true);
                    args.push(mlir_arg.into());
                    self.scalar_pair_element_backend_type(arg.layout, 1, true)
                }
                rustc_target::callconv::PassMode::Cast { pad_i32, cast } => todo!(),
                rustc_target::callconv::PassMode::Indirect {
                    attrs,
                    meta_attrs,
                    on_stack,
                } => todo!(),
            };
            args.push(mlir_arg.into());
        }
        let mut ret = vec![];
        if !fn_abi.ret.layout.is_zst() {
            ret.push(self.mlir_type(fn_abi.ret.layout).into())
        }
        let mut operation: Operation<'ml> = melior::dialect::func::func(
            mlir_ctx,
            // accepts a StringAttribute which is the function name.
            melior::ir::attribute::StringAttribute::new(mlir_ctx, sym.as_str()),
            // A type attribute, defining the function signature.
            melior::ir::attribute::TypeAttribute::new(
                melior::ir::r#type::FunctionType::new(&mlir_ctx, &args, &ret).into(),
            ),
            melior::ir::Region::new(),
            &[],
            location,
        );
        operation.set_op_visible(self.mlir_ctx, visibility);
        let body: melior::ir::BlockRef<'ml, 'ml> = self.mlir_module.body();
        let op: melior::ir::OperationRef<'ml, 'ml> = body.append_operation(operation);
        op
    }
}
