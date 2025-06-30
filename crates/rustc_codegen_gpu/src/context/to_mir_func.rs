use melior::ir::operation::OperationMutLike;
use melior::ir::r#type::FunctionType;
use melior::ir::{BlockLike, Location, Operation};
use rustc_codegen_ssa_gpu::traits::{LayoutTypeCodegenMethods, MiscCodegenMethods};
use rustc_hir::def_id::DefId;
use rustc_middle::query::Key;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, LayoutOf};
use rustc_middle::ty::{AssocItemContainer, Instance};
use tracing::{trace, warn};

use super::GPUCodegenContext;
use crate::attr::{GpuAttributes, GpuItem};
use crate::context::CodegenGPUError;
use crate::mlir::{MLIRMutOpHelpers, MLIROpHelpers, MLIRVisibility};

fn is_impl_of_trait_method(
    tcx: &rustc_middle::ty::TyCtxt<'_>,
    def_id: DefId,
    trait_sym: rustc_span::Symbol,
    method_sym: rustc_span::Symbol,
) -> bool {
    // Get the trait DefId for core::iter::traits::IntoIterator
    let into_iter_trait = match tcx.get_diagnostic_item(trait_sym) {
        Some(id) => id,
        None => return false,
    };

    // Check if the provided DefId is an associated function in an impl of IntoIterator
    if let Some(assoc_item) = tcx.opt_associated_item(def_id) {
        if assoc_item.container == AssocItemContainer::Impl && assoc_item.name == method_sym {
            // Get the trait ref for the impl (if it's an impl of a trait)
            if let Some(trait_ref) =
                tcx.impl_of_method(def_id).and_then(|impl_def_id| tcx.impl_trait_ref(impl_def_id))
            {
                return trait_ref.skip_binder().def_id == into_iter_trait;
            }
        }
    }

    false
}

impl<'tcx, 'ml, 'a> GPUCodegenContext<'tcx, 'ml, 'a> {
    pub fn gpu_return(
        &self,
        ret: &[melior::ir::Value<'ml, 'a>],
        loc: Location<'ml>,
    ) -> melior::ir::Operation<'ml> {
        melior::dialect::ods::gpu::r#return(self.mlir_ctx, ret, loc).into()
    }

    pub fn cpu_return(
        &self,
        ret: &[melior::ir::Value<'ml, 'a>],
        loc: Location<'ml>,
    ) -> melior::ir::Operation<'ml> {
        melior::dialect::func::r#return(ret, loc)
    }

    /// Get the function pointer value for the given instance.
    pub fn to_mir_func_const(
        &self,
        instance: Instance<'tcx>,
        block: Option<melior::ir::BlockRef<'ml, 'a>>,
    ) -> melior::ir::Value<'ml, 'a> {
        let def_id: rustc_hir::def_id::DefId = instance.def_id();

        let sym = self.tcx.symbol_name(instance).name.to_string();
        if self.fn_ptr_db.read().unwrap().contains_key(&sym) {
            return self.fn_ptr_db.read().unwrap()[&sym].1;
        }
        let gpu_attrs = self.get_gpu_attrs(instance.def_id());
        let function =
            melior::ir::attribute::FlatSymbolRefAttribute::new(self.mlir_ctx, sym.as_str());
        let op = self.get_fn(instance);
        let ftyp = op.get_func_type().unwrap();
        let is_gpu = op.is_gpu_func();
        let mut const_op =
            melior::dialect::func::constant(self.mlir_ctx, function, ftyp, self.unknown_loc());
        if let Some(extra_attr) = gpu_attrs.to_mlir_attribute(self.mlir_ctx) {
            const_op.set_attribute(crate::mlir::BUILTIN_SYM, extra_attr);
        }
        let op = if let Some(b) = block {
            b.append_operation(const_op)
        } else {
            self.mlir_body(is_gpu).append_operation(const_op)
        };
        let val = op.result(0).unwrap().into();
        self.fn_ptr_db.write().unwrap().insert(sym, (instance, val));
        val
    }

    pub fn fn_abi_to_inputs(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        is_kernel_entry: bool,
    ) -> Result<Vec<melior::ir::Type<'ml>>, CodegenGPUError> {
        let mut args = vec![];
        //let mut closures = vec![];
        for arg in &abi.args {
            let mlir_arg = match &arg.mode {
                rustc_target::callconv::PassMode::Ignore => {
                    continue;
                }
                rustc_target::callconv::PassMode::Direct(arg_attributes) => {
                    self.immediate_backend_type(arg.layout)
                }
                rustc_target::callconv::PassMode::Pair(arg_attributes, arg_attributes1) => {
                    let mlir_arg = self.scalar_pair_element_backend_type(arg.layout, 0, true);
                    args.append(&mut self.expand_type(mlir_arg));
                    self.scalar_pair_element_backend_type(arg.layout, 1, true)
                }
                rustc_target::callconv::PassMode::Cast { pad_i32, cast } => {
                    return Err(format!(
                        "Does not support fn abi cast: {:?} pad_i32: {}",
                        cast, pad_i32
                    ));
                }
                rustc_target::callconv::PassMode::Indirect { attrs, meta_attrs, on_stack } => {
                    if is_kernel_entry {
                        return Err("Kernel entry does not support fn abi indirect".to_string());
                    }
                    warn!(
                        "indirect arg: {:?} attrs: {:?} meta_attrs: {:?} on_stack: {}",
                        arg.layout, attrs, meta_attrs, on_stack
                    );
                    let ptr_ty = rustc_middle::ty::Ty::new_mut_ptr(self.tcx, arg.layout.ty);
                    let ptr_layout = self.layout_of(ptr_ty);
                    self.mlir_type(ptr_layout, false)
                }
            };
            args.append(&mut self.expand_type(mlir_arg));
        }
        Ok(args)
    }

    pub fn fn_abi_to_fn_type(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        is_kernel_entry: bool,
    ) -> Result<FunctionType<'ml>, CodegenGPUError> {
        let mut args = self.fn_abi_to_inputs(abi, is_kernel_entry)?;
        let mut ret = vec![];
        if !abi.ret.layout.is_zst() {
            match abi.ret.mode {
                rustc_target::callconv::PassMode::Indirect { .. } => {
                    if is_kernel_entry {
                        return Err(
                            "Kernel entry does not support fn abi indirect return".to_string()
                        );
                    }
                    let ptr_ty = rustc_middle::ty::Ty::new_mut_ptr(self.tcx, abi.ret.layout.ty);
                    let ptr_layout = self.layout_of(ptr_ty);
                    // Indirect return type becomes an arguments to function.
                    args.insert(0, self.mlir_type(ptr_layout, false));
                }
                rustc_target::callconv::PassMode::Direct(_)
                | rustc_target::callconv::PassMode::Pair(..) => {
                    let t = self.mlir_type(abi.ret.layout, false);
                    let mut t = self.expand_type(t);
                    ret.append(&mut t);
                }
                rustc_target::callconv::PassMode::Ignore => {
                    warn!("Function return is ignored: {:?}", abi.ret.layout);
                }
                rustc_target::callconv::PassMode::Cast { .. } => {
                    panic!("Function return is cast: {:?} for {:?}", abi.ret.layout, abi.ret.mode);
                }
            }
        }
        if args.is_empty() && ret.is_empty() {
            warn!("function has no args and no ret");
        }
        let ftype: FunctionType<'ml> = FunctionType::new(self.mlir_ctx, &args, &ret);
        Ok(ftype)
    }

    pub fn get_gpu_attrs(&self, def_id: rustc_hir::def_id::DefId) -> GpuAttributes {
        let attrs = self.tcx().get_attrs_unchecked(def_id);
        GpuAttributes::parse(attrs)
    }

    pub fn to_mir_func_decl(
        &self,
        instance: Instance<'tcx>,
        visibility: MLIRVisibility,
    ) -> melior::ir::OperationRef<'ml, 'a> {
        let tcx = self.tcx();
        let mlir_ctx = self.mlir_ctx;
        let sym = tcx.symbol_name(instance).name.to_string();
        let def_id: rustc_hir::def_id::DefId = instance.def_id();
        {
            let fn_db = self.fn_db.read().unwrap();
            if fn_db.contains_key(&sym) {
                return fn_db[&sym];
            }
        }
        let mut gpu_attrs = self.get_gpu_attrs(def_id);
        let need_def = gpu_attrs.is_gpu_related();
        let span = instance.def.default_span(tcx);
        let location: melior::ir::Location<'ml> = self.to_mlir_loc(span);
        let fn_sym = melior::ir::attribute::StringAttribute::new(mlir_ctx, sym.as_str());
        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let ftype = self
            .fn_abi_to_fn_type(fn_abi, gpu_attrs.kernel)
            .unwrap_or_else(|e| self.emit_error(e, span));
        if is_impl_of_trait_method(
            &tcx,
            def_id,
            rustc_span::sym::IntoIterator,
            rustc_span::sym::into_iter,
        ) {
            gpu_attrs.gpu_item = Some(GpuItem::IntoIter);
            // TODO: fill the elemenent type
            //self.type_func(&[], self.type_empty())
        } else if is_impl_of_trait_method(
            &tcx,
            def_id,
            rustc_span::sym::Iterator,
            rustc_span::sym::next,
        ) {
            gpu_attrs.gpu_item = Some(GpuItem::IterNext);
            //self.type_func(&[], self.type_empty())
        }
        let fn_type = melior::ir::attribute::TypeAttribute::new(ftype.into());
        let mut operation: Operation<'ml> = if !gpu_attrs.kernel {
            melior::dialect::ods::func::func(
                mlir_ctx,
                melior::ir::Region::new(),
                // accepts a StringAttribute which is the function name.
                fn_sym,
                // A type attribute, defining the function signature.
                fn_type,
                location,
            )
            .into()
        } else {
            if ftype.result_count() != 0 {
                self.emit_error(
                    "GPU kernel entry function must not return a value".to_string(),
                    span,
                );
            }
            let gpu_op = melior::dialect::ods::gpu::func(
                mlir_ctx,
                melior::ir::Region::new(),
                fn_type,
                location,
            );
            let unit_attr = melior::ir::Attribute::unit(self.mlir_ctx);
            let mut op: Operation<'ml> = gpu_op.into();
            op.set_attribute("kernel", unit_attr);
            op.set_attribute(crate::mlir::SYM_NAME_SYM, fn_sym.into());

            op
        };
        if let Some(extra_attr) = gpu_attrs.to_mlir_attribute(self.mlir_ctx) {
            operation.set_attribute(crate::mlir::BUILTIN_SYM, extra_attr);
        }
        let visibility = if !gpu_attrs.host || !gpu_attrs.kernel {
            MLIRVisibility::Private
        } else {
            MLIRVisibility::Public
        };
        let in_gpu_mod = gpu_attrs.kernel || gpu_attrs.device;
        operation.set_op_visible(self.mlir_ctx, visibility);
        let body = self.mlir_body(in_gpu_mod);
        trace!("append operation to block {} {:?}", operation, fn_sym);
        let op = body.append_operation(operation);
        self.fn_db.write().unwrap().insert(sym, op);
        op
    }
}
