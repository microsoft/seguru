use melior::ir::{self as mlir_ir, Attribute};
use rustc_codegen_ssa_gpu::traits::{
    BaseTypeCodegenMethods, BuilderMethods, IntrinsicCallBuilderMethods,
};
use rustc_middle::query::Key;
use rustc_span::Symbol;

use super::GpuBuilder;
use crate::mlir::gpu::{DimFn, DimType};

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub(crate) fn dim_fn_res(&mut self, f: DimFn, ty: DimType) -> mlir_ir::Value<'ml, 'a> {
        let attr = Attribute::parse(self.mlir_ctx, &format!("#gpu<dim {}>", ty.to_str())).unwrap();
        self.append_op_res(f.build(self.mlir_ctx, attr, self.cur_loc()))
    }

    pub(crate) fn add_dim_assumptions(&mut self) {
        if !self.inside_kernel_func() {
            return;
        }
        let name = &self.name;
        let instance = self.fn_db.read().unwrap()[name].instance;
        let tcx = self.tcx;
        let span = instance.def.default_span(tcx);
        for arg in instance.args.iter() {
            if let rustc_type_ir::GenericArgKind::Type(ty) = arg.unpack() {
                if let rustc_middle::ty::TyKind::Adt(adt_def, substs) = ty.kind() {
                    let def_id = adt_def.did();
                    let safe_config_def_id = match tcx
                        .get_diagnostic_item(Symbol::intern("gpu::Config"))
                    {
                        Some(id) => id,
                        None => {
                            self.emit_error(
                                    "Cannot find gpu::Config trait. Please make sure you have the used gpu crate.".to_string(),
                                    span,
                                );
                        }
                    };

                    const DIM_TUPLES: [(DimFn, DimFn, DimType, &str); 6] = [
                        (DimFn::BlockDim, DimFn::ThreadId, DimType::X, "gpu::Config::BDIM_X"),
                        (DimFn::BlockDim, DimFn::ThreadId, DimType::Y, "gpu::Config::BDIM_Y"),
                        (DimFn::BlockDim, DimFn::ThreadId, DimType::Z, "gpu::Config::BDIM_Z"),
                        (DimFn::GridDim, DimFn::BlockId, DimType::X, "gpu::Config::GDIM_X"),
                        (DimFn::GridDim, DimFn::BlockId, DimType::Y, "gpu::Config::GDIM_Y"),
                        (DimFn::GridDim, DimFn::BlockId, DimType::Z, "gpu::Config::GDIM_Z"),
                    ];
                    let mut dim_tuples = vec![];
                    for (dim, tid, dim_type, symbol) in DIM_TUPLES.iter() {
                        match tcx.get_diagnostic_item(Symbol::intern(symbol)) {
                            Some(id) => dim_tuples.push((dim, tid, dim_type, id)),
                            None => {
                                self.emit_error(
                                    "Cannot find gpu::Config trait. Please make sure you have the used gpu crate.".to_string(),
                                    span,
                                );
                            }
                        };
                    }
                    // Check if this type implements this trait
                    tcx.for_each_relevant_impl(safe_config_def_id, ty, |impl_def_id| {
                        let associated_items = tcx.associated_items(impl_def_id);
                        associated_items.in_definition_order().for_each(|item| {
                            for (&dim, &tid, &dim_type, diag_id) in dim_tuples.iter() {
                                if item.trait_item_def_id == Some(*diag_id) {
                                    let val = tcx
                                        .const_eval_poly(item.def_id)
                                        .ok()
                                        .unwrap()
                                        .try_to_scalar_int()
                                        .unwrap()
                                        .to_u32();
                                    let dim_val = self.dim_fn_res(dim, dim_type);
                                    let tid_val = self.dim_fn_res(tid, dim_type);
                                    let cond_dynamic = self.icmp(
                                        rustc_codegen_ssa_gpu::common::IntPredicate::IntULT,
                                        tid_val,
                                        dim_val,
                                    );
                                    self.assume(cond_dynamic);
                                    if val != 0 {
                                        let val = self.const_value(val, self.type_i32());
                                        let cond_dim = self.icmp(
                                            rustc_codegen_ssa_gpu::common::IntPredicate::IntEQ,
                                            dim_val,
                                            val,
                                        );
                                        let cond_id = self.icmp(
                                            rustc_codegen_ssa_gpu::common::IntPredicate::IntULT,
                                            tid_val,
                                            val,
                                        );
                                        self.assume(cond_id);
                                        self.assume(cond_dim);
                                    }
                                }
                            }
                        });
                    });
                }
            }
        }
    }
}
