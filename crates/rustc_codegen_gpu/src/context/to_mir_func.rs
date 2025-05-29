use melior::ir::attribute::StringAttribute;
use melior::ir::r#type::FunctionType;
use melior::ir::{BlockLike, Location, Operation};
use rustc_codegen_ssa_gpu::traits::{LayoutTypeCodegenMethods, MiscCodegenMethods};
use rustc_hir::def_id::DefId;
use rustc_middle::query::Key;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt};
use rustc_middle::ty::{AssocItemContainer, Instance};
use rustc_span::Span;

use super::GPUCodegenContext;
use crate::attr::{GpuAttributes, GpuItem};
use crate::builder::GpuBuilderState;
use crate::mlir::{BUILTIN_SYM, MLIRMutOpHelpers, MLIROpHelpers, MLIRVisibility, ValueToOpRef};

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

    pub fn call_op(
        &self,
        fn_ptr_value: melior::ir::Value<'ml, 'a>,
        args: &[melior::ir::Value<'ml, 'a>],
        ftype: Option<FunctionType<'ml>>,
        extra: &mut crate::builder::GpuBuilderState<'ml, 'a>,
        span: Span,
    ) -> Result<Option<melior::ir::Operation<'ml>>, melior::Error> {
        let mut return_type = vec![];
        let fn_sym_ptr = fn_ptr_value.to_func_sym();
        let builtin_sym = fn_ptr_value.get_op_attr::<StringAttribute>(BUILTIN_SYM);
        if let Some(ftype) = ftype {
            for i in 0..ftype.result_count() {
                return_type.push(ftype.result(i).unwrap());
            }
        } else {
            panic!("")
        }
        let loc = self.to_mlir_loc(span);
        if let Ok(builtin_sym) = builtin_sym {
            log::trace!("call_op fn_sym_ptr: {:?}", builtin_sym.value());
            if let Ok(gpu_item) = GpuItem::try_from(builtin_sym.value()) {
                return self.call_gpu_builtin_operation(
                    gpu_item,
                    fn_ptr_value,
                    args,
                    &return_type,
                    extra,
                    span,
                );
            } else {
                panic!();
            }
        }
        if let Ok(fn_sym_ptr) = fn_sym_ptr {
            Ok(Some(melior::dialect::func::call(
                self.mlir_ctx,
                fn_sym_ptr,
                args,
                &return_type,
                loc,
            )))
        } else {
            Ok(Some(melior::dialect::func::call_indirect(fn_ptr_value, args, &return_type, loc)))
        }
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
    ) -> Vec<melior::ir::Type<'ml>> {
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
                rustc_target::callconv::PassMode::Cast { pad_i32, cast } => todo!(),
                rustc_target::callconv::PassMode::Indirect { attrs, meta_attrs, on_stack } => {
                    assert!(!on_stack);
                    self.type_memref(self.immediate_backend_type(arg.layout))
                }
            };
            args.append(&mut self.expand_type(mlir_arg));
        }
        args
    }
    pub fn fn_abi_to_fn_type(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> FunctionType<'ml> {
        let args = self.fn_abi_to_inputs(abi);
        let mut ret = vec![];
        if !abi.ret.layout.is_zst() {
            let t = self.mlir_type(abi.ret.layout, false);
            let mut t = self.expand_type(t);
            ret.append(&mut t);
        }
        if args.is_empty() && ret.is_empty() {
            log::warn!("function has no args and no ret");
        }
        let ftype: FunctionType<'ml> = FunctionType::new(self.mlir_ctx, &args, &ret);
        ftype
    }

    fn call_gpu_builtin_operation(
        &self,
        gpu_item: GpuItem,
        fn_ptr_value: melior::ir::Value<'ml, 'a>,
        args: &[melior::ir::Value<'ml, 'a>],
        return_types: &[melior::ir::Type<'ml>],
        extra: &mut GpuBuilderState<'ml, 'a>,
        span: Span,
    ) -> Result<Option<melior::ir::Operation<'ml>>, melior::Error> {
        let loc = self.to_mlir_loc(span);
        match gpu_item {
            GpuItem::ThreadId => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(extra.attrs.len() == 1);
                assert!(return_types.len() == 1);
                let dimention = extra.attrs.pop().unwrap();
                Ok(Some(crate::mlir::gpu::thread_id(self.mlir_ctx, dimention, loc)))
            }
            GpuItem::GlobalThreadId => {
                let dimention = extra.attrs.pop().unwrap();
                Ok(Some(crate::mlir::gpu::global_id(self.mlir_ctx, dimention, loc)))
            }
            GpuItem::Printf => {
                assert!(extra.attrs.len() == 1);
                let Ok(format) = extra.attrs.pop().unwrap().try_into() else {
                    let err =
                        format!("{:?} must take a single StringAttribute as format", gpu_item);
                    self.emit_error(err.clone(), span);
                    return Err(melior::Error::AttributeNotFound(err));
                };
                Ok(Some(melior::dialect::ods::gpu::printf(self.mlir_ctx, args, format, loc).into()))
            }
            GpuItem::AddStringAttr => {
                // args must be a const string.
                let arg = args[0];
                let err_msg =
                    || format!("{:?} must take valid string as MLIR attritutes", gpu_item);
                let name = arg.to_get_global_name();
                if name.is_err() {
                    self.emit_error(err_msg(), span);
                }
                let global_name = name.unwrap().value().to_string();
                let bytes = self.get_const_bytes_by_name(global_name.as_str());
                if let Ok(bytes) = std::str::from_utf8(bytes) {
                    let attr = melior::ir::Attribute::parse(self.mlir_ctx, bytes).unwrap();
                    extra.attrs.push(attr);
                } else {
                    self.emit_error(err_msg(), span);
                }
                Ok(None)
            }
            GpuItem::Scope => {
                log::trace!("gpu.scope args: {:?}", args);
                extra.inside_gpu_scope = true;
                Ok(Some(melior::dialect::func::call(
                    self.mlir_ctx,
                    fn_ptr_value.to_func_sym().unwrap(),
                    args,
                    return_types,
                    loc,
                )))
            }
            GpuItem::Grid => {
                log::trace!("gpu.grid args: {:?}", args);
                extra.args.insert(gpu_item, args.to_vec());
                Ok(Some(melior::dialect::func::call(
                    self.mlir_ctx,
                    fn_ptr_value.to_func_sym().unwrap(),
                    args,
                    return_types,
                    loc,
                )))
            }
            GpuItem::Block => {
                log::trace!("gpu.block args: {:?}", args);
                extra.args.insert(gpu_item, args.to_vec());
                Ok(Some(melior::dialect::func::call(
                    self.mlir_ctx,
                    fn_ptr_value.to_func_sym().unwrap(),
                    args,
                    return_types,
                    loc,
                )))
            }
            GpuItem::Launch => {
                log::trace!("gpu.launch args: {:?}", args);
                extra.args.insert(gpu_item, args.to_vec());
                /*let op = melior::ir::operation::OperationBuilder::new(
                    "gpu.launch",
                    self.to_mlir_loc(span),
                )
                .add_results(return_types)
                .build()
                .unwrap();*/
                Ok(None)
            }
            GpuItem::IntoIter => todo!(),
            GpuItem::IterNext => todo!(),
        }
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
        let ftype = self.fn_abi_to_fn_type(fn_abi);
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
        log::trace!("append operation to block {} {:?}", operation, fn_sym);
        let op = body.append_operation(operation);
        self.fn_db.write().unwrap().insert(sym, op);
        op
    }
}
