use melior::{
    ir::{
        attribute::StringAttribute, r#type::FunctionType, Attribute, BlockLike, Location, Operation,
    },
    pass::gpu,
};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, LayoutTypeCodegenMethods, MiscCodegenMethods,
};
use rustc_middle::{
    query::Key,
    ty::{
        layout::{FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv},
        Instance,
    },
};
use rustc_span::{sym::assert, Span};

use crate::{
    attr::{GpuAttributes, GpuItem},
    mlir::{MLIRMutOpHelpers, MLIROpHelpers, MLIRVisibility, ValueToOpRef, BUILTIN_SYM},
};

use super::GPUCodegenContext;

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
        melior::dialect::func::r#return(ret, loc).into()
    }

    pub fn call_op(
        &self,
        fn_ptr_value: melior::ir::Value<'ml, 'a>,
        args: &[melior::ir::Value<'ml, 'a>],
        ftype: Option<FunctionType<'ml>>,
        attrs: &mut Vec<melior::ir::Attribute<'ml>>,
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
            assert!(false);
        }
        let loc = self.to_mlir_loc(span);
        if let Ok(builtin_sym) = builtin_sym {
            log::trace!("call_op fn_sym_ptr: {:?}", builtin_sym.value());
            if let Ok(gpu_item) = GpuItem::try_from(builtin_sym.value()) {
                let ret =
                    self.call_gpu_builtin_operation(gpu_item, args, &return_type, attrs, span);
                return ret;
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
            Ok(Some(melior::dialect::func::call_indirect(
                fn_ptr_value,
                args,
                &return_type,
                loc,
            )))
        }
    }
    /// Get the function pointer value for the given instance.
    pub fn to_mir_func_const(&self, instance: Instance<'tcx>) -> melior::ir::Value<'ml, 'a> {
        let sym = self.tcx.symbol_name(instance).name.to_string();
        let gpu_attrs = self.get_gpu_attrs(instance.def_id());
        let function =
            melior::ir::attribute::FlatSymbolRefAttribute::new(&self.mlir_ctx, sym.as_str());
        println!("FlatSymbolRefAttribute: {}", function);
        let op = self.get_fn(instance);
        let ftyp = op.get_func_type().unwrap();
        let is_gpu = op.is_gpu_func();

        let mut const_op =
            melior::dialect::func::constant(self.mlir_ctx, function, ftyp, self.unknown_loc());
        if let Some(extra_attr) = gpu_attrs.to_mlir_attribute(&self.mlir_ctx) {
            const_op.set_attribute(crate::mlir::BUILTIN_SYM, extra_attr);
        }
        let const_op: melior::ir::OperationRef<'ml, 'a> =
            self.mlir_body(is_gpu).append_operation(const_op);
        let r = const_op.result(0).unwrap();
        r.into()
    }

    pub fn fn_abi_to_fn_type(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
    ) -> FunctionType<'ml> {
        let mut args = vec![];
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
        if !abi.ret.layout.is_zst() {
            ret.push(self.mlir_type(abi.ret.layout, false).into())
        }
        FunctionType::new(&self.mlir_ctx, &args, &ret)
    }

    fn call_gpu_builtin_operation(
        &self,
        gpu_item: GpuItem,
        args: &[melior::ir::Value<'ml, 'a>],
        return_types: &[melior::ir::Type<'ml>],
        attrs: &mut Vec<melior::ir::Attribute<'ml>>,
        span: Span,
    ) -> Result<Option<melior::ir::Operation<'ml>>, melior::Error> {
        match gpu_item {
            GpuItem::ThreadId => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(attrs.len() == 1);
                assert!(return_types.len() == 1);
                let dimention = attrs.pop().unwrap();
                Ok(Some(
                    crate::mlir::gpu::thread_id(self.mlir_ctx, dimention, self.to_mlir_loc(span))
                        .into(),
                ))
            }
            GpuItem::GlobalThreadId => {
                let dimention = attrs.pop().unwrap();
                Ok(Some(
                    crate::mlir::gpu::global_id(self.mlir_ctx, dimention, self.to_mlir_loc(span))
                        .into(),
                ))
            }
            GpuItem::Printf => {
                assert!(attrs.len() == 1);
                let Ok(format) = attrs.pop().unwrap().try_into() else {
                    let err = format!(
                        "{:?} must take a single StringAttribute as format",
                        gpu_item
                    );
                    self.emit_error(err.clone(), span);
                    return Err(melior::Error::AttributeNotFound(err));
                };
                Ok(Some(
                    melior::dialect::ods::gpu::printf(
                        self.mlir_ctx,
                        args,
                        format,
                        self.to_mlir_loc(span),
                    )
                    .into(),
                ))
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
                    attrs.push(attr);
                } else {
                    self.emit_error(err_msg(), span);
                }
                Ok(None)
            }
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
            if fn_db.contains_key(&def_id) {
                return fn_db[&def_id];
            }
        }
        let gpu_attrs = self.get_gpu_attrs(def_id);
        assert!(
            gpu_attrs.kernel || gpu_attrs.host || gpu_attrs.device || gpu_attrs.gpu_item.is_some(),
            r"function {sym} is not a kernel or host function",
        );
        dbg!(&sym);
        let span = instance.def.default_span(tcx);
        let location: melior::ir::Location<'ml> = self.to_mlir_loc(span);
        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let fn_type = self.fn_abi_to_fn_type(fn_abi);
        let fn_type = melior::ir::attribute::TypeAttribute::new(fn_type.into());
        let fn_sym = melior::ir::attribute::StringAttribute::new(mlir_ctx, sym.as_str());
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
            let mut op: Operation<'ml> = gpu_op.into();
            op.set_attribute(crate::mlir::SYM_NAME_SYM, fn_sym.into());
            op
        };
        if let Some(extra_attr) = gpu_attrs.to_mlir_attribute(&self.mlir_ctx) {
            operation.set_attribute(crate::mlir::BUILTIN_SYM, extra_attr);
        }
        let visibility = MLIRVisibility::Private;
        operation.set_op_visible(self.mlir_ctx, visibility);
        let body = self.mlir_body(!gpu_attrs.host);
        log::trace!("append operation to block {}", operation);
        let op = body.append_operation(operation);
        self.fn_db.write().unwrap().insert(def_id, op);
        op
    }
}
