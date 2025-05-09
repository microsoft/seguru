use melior::ir::{BlockLike, Operation};
use rustc_middle::{
    query::Key,
    ty::{
        layout::{HasTyCtxt, HasTypingEnv},
        Instance,
    },
};

use super::GPUCodegenContext;

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn to_mir_func(&self, instance: Instance<'tcx>) -> melior::ir::OperationRef<'ml, 'a> {
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
        //let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let operation: Operation<'ml> = melior::dialect::func::func(
            mlir_ctx,
            // accepts a StringAttribute which is the function name.
            melior::ir::attribute::StringAttribute::new(mlir_ctx, sym.as_str()),
            // A type attribute, defining the function signature.
            melior::ir::attribute::TypeAttribute::new(
                melior::ir::r#type::FunctionType::new(&mlir_ctx, &[], &[]).into(),
            ),
            {
                // The first block within the region, blocks accept arguments
                // In regions with control flow, MLIR leverages
                // this structure to implicitly represent
                // the passage of control-flow dependent values without the complex nuances
                // of PHI nodes in traditional SSA representations.
                //let block = melior::ir::Block::new(&[]);

                // Return the result using the "func" dialect return operation.
                //block.append_operation(melior::dialect::func::r#return(&[], location));

                // The Func operation requires a region,
                // we add the block we created to the region and return it,
                // which is passed as an argument to the `func::func` function.
                let region = melior::ir::Region::new();
                //region.append_block(block);
                region
            },
            &[],
            location,
        );
        let body: melior::ir::BlockRef<'ml, 'ml> = self.mlir_module.body();
        let op: melior::ir::OperationRef<'ml, 'ml> = body.append_operation(operation);
        op
    }
}
