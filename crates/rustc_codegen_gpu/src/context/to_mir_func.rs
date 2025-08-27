use std::collections::HashMap;

use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{OperationLike, OperationMutLike};
use melior::ir::r#type::FunctionType;
use melior::ir::{BlockLike, Location, Operation, TypeLike, ValueLike};
use rustc_codegen_ssa_gpu::traits::{
    ConstCodegenMethods, LayoutTypeCodegenMethods, MiscCodegenMethods,
};
use rustc_hir::def_id::DefId;
use rustc_middle::query::Key;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{AssocItemContainer, Instance};
use tracing::debug;

use super::GPUCodegenContext;
use crate::builder::GpuBuilder;
use crate::context::CodegenGPUError;
use crate::mlir::{MLIRMutOpHelpers, MLIROpHelpers, MLIRVisibility};

const INDIRECT: &str = "__indirect_";

pub struct IndirectEntry<'tcx, 'ml> {
    entry_sym: String,
    fn_type: FunctionType<'ml>,
    indirect_args: HashMap<usize, TyAndLayout<'tcx>>,
    dev_fn_type: FunctionType<'ml>,
    dev_instance: Instance<'tcx>,
    location: Location<'ml>,
}

type TyAndLayout<'tcx> = rustc_abi::TyAndLayout<'tcx, rustc_middle::ty::Ty<'tcx>>;
type IndirectArgsTypes<'tcx> = HashMap<usize, TyAndLayout<'tcx>>;
#[allow(dead_code)]
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
    pub(crate) fn sanitized_symbol_name(&self, instance: Instance<'tcx>) -> String {
        let ty = self.tcx.type_of(instance.def_id());
        let mono_ty = self
            .tcx
            .try_instantiate_and_normalize_erasing_regions(instance.args, self.typing_env(), ty)
            .expect("failed to instantiate and normalize");
        gpu_name::convert_def_path_to_gpu_sym_name(&rustc_const_eval::util::type_name(
            self.tcx, mono_ty,
        ))
    }

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

        let instance_sym = self.sanitized_symbol_name(instance);
        if self.fn_ptr_db.read().unwrap().contains_key(&instance_sym) {
            return self.fn_ptr_db.read().unwrap()[&instance_sym].1;
        }
        let gpu_attrs = self.gpu_attrs(&instance);
        let op = self.get_fn(instance);
        let sym = StringAttribute::try_from(op.attribute("sym_name").unwrap()).unwrap().value();
        let function = melior::ir::attribute::FlatSymbolRefAttribute::new(self.mlir_ctx, sym);
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
        self.fn_ptr_db.write().unwrap().insert(instance_sym, (instance, val));
        val
    }

    fn fn_abi_to_inputs(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        is_kernel_entry: bool,
    ) -> Result<(Vec<melior::ir::Type<'ml>>, Option<IndirectArgsTypes<'tcx>>), CodegenGPUError>
    {
        let mut args = vec![];
        let mut real_args = HashMap::new();
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
                    self.cast_to_mlir_type(cast)
                }
                rustc_target::callconv::PassMode::Indirect { attrs, meta_attrs, on_stack } => {
                    if is_kernel_entry {
                        real_args.insert(args.len(), arg.layout);
                        //return Err("Kernel entry does not support fn abi indirect".to_string());
                    }
                    debug!(
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
        let real_args = if real_args.is_empty() { None } else { Some(real_args) };
        Ok((args, real_args))
    }

    pub(crate) fn fn_abi_to_fn_type(
        &self,
        abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        is_kernel_entry: bool,
        ret_shared: bool,
    ) -> Result<
        (FunctionType<'ml>, Option<(FunctionType<'ml>, IndirectArgsTypes<'tcx>)>),
        CodegenGPUError,
    > {
        let (mut args, indirect_types) = self.fn_abi_to_inputs(abi, is_kernel_entry)?;

        // In rust, if struct size > 16bytes, it is passed as a pointer via indirect mode.
        // We need to convert it to a pointer type to translate code correctly.
        // However, we should not do this for kernel entry function.
        // If we have indirect arguments for kernel entry, we need to make this
        // func as a dev function with indirect arguments
        // and create a new kernel entry function using direct arguments.
        let mut new_entry_args = None;
        if let Some(indirect_types) = &indirect_types {
            let mut tmp_args = args.clone();
            // If we have indirect_types, we need to expand the types for
            // a crafted kernel entry.
            let mut keys = indirect_types.keys().collect::<Vec<_>>();
            keys.sort_by(|a, b| b.cmp(a));
            for i in keys {
                tmp_args.remove(*i);
                let mut idx = *i;
                for t in self.expand_type(self.mlir_type(indirect_types[i], false)) {
                    tmp_args.insert(idx, t);
                    idx += 1;
                }
            }
            new_entry_args = Some(tmp_args);
        }
        let mut ret = vec![];
        if !abi.ret.layout.is_zst() {
            match &abi.ret.mode {
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
                    debug!("Function return is ignored: {:?}", abi.ret.layout);
                }
                rustc_target::callconv::PassMode::Cast { box cast, .. } => {
                    ret.extend(self.expand_type(self.cast_to_mlir_type(cast)));
                }
            }
        }
        if args.is_empty() && ret.is_empty() {
            debug!("function has no args and no ret");
        }

        // If we specify the return memory space to shared,
        // we need to set the memory space for all memref return types.
        if ret_shared {
            ret.iter_mut().for_each(|t| {
                if t.is_mem_ref() {
                    *t = self
                        .memref_set_memory_space(
                            (*t).try_into().unwrap(),
                            Some(crate::mlir::memref::MemorySpace::Shared.to_attr(self.mlir_ctx)),
                        )
                        .into();
                }
            });
        }
        let ftype: FunctionType<'ml> = FunctionType::new(self.mlir_ctx, &args, &ret);
        let new_ftype = new_entry_args.map(|new_args| {
            (FunctionType::new(self.mlir_ctx, &new_args, &ret), indirect_types.unwrap())
        });
        Ok((ftype, new_ftype))
    }

    pub(crate) fn define_indirect_if_needed(&'ml self) {
        let Some(entry) = self.indirect_entry.lock().unwrap().take() else {
            return;
        };
        let IndirectEntry {
            entry_sym,
            fn_type,
            indirect_args,
            dev_fn_type,
            dev_instance,
            location,
        } = entry;
        use rustc_codegen_ssa_gpu::traits::{AbiBuilderMethods, BuilderMethods};
        let mlir_ctx = self.mlir_ctx;
        let fn_type_attr = melior::ir::attribute::TypeAttribute::new(fn_type.into());
        let body = self.mlir_body(true);
        let block = melior::ir::Block::new(&[]);
        let fn_sym = melior::ir::attribute::StringAttribute::new(mlir_ctx, entry_sym.as_str());
        let mut gpu_op: Operation<'_> = melior::dialect::ods::gpu::func(
            mlir_ctx,
            melior::ir::Region::new(),
            fn_type_attr,
            location,
        )
        .into();
        gpu_op.set_attribute("kernel", melior::ir::Attribute::unit(self.mlir_ctx));
        gpu_op.set_op_visible(self.mlir_ctx, MLIRVisibility::Public);
        gpu_op.set_attribute(crate::mlir::SYM_NAME_SYM, fn_sym.into());
        let op: melior::ir::OperationRef<'_, '_> = body.append_operation(gpu_op);

        let bb = GpuBuilder::append_block(self, op, entry_sym.as_str());
        let mut builder = GpuBuilder::build(self, bb);
        let mut dev_arg_idx = 0;
        let mut call_args = vec![];
        let mut kernel_arg_idx = 0;
        let mut indirect_idx = indirect_args.keys().collect::<Vec<_>>();
        indirect_idx.sort();
        for i in indirect_idx {
            let ty_layout = indirect_args[i];
            let i = *i;
            if i > dev_arg_idx {
                for _ in dev_arg_idx..i {
                    call_args.push(builder.get_param(kernel_arg_idx));
                    kernel_arg_idx += 1;
                }
            }
            let layout = ty_layout.layout;
            let size = layout.size;
            let align = layout.align.abi;
            let ty = self.mlir_type(ty_layout, false);
            let arg_ptr = builder.alloca(size, align);
            let tupe_ty = self.expand_type(ty);
            let mut ele_offset = 0;
            for j in 0..tupe_ty.len() {
                let arg = builder.get_param(kernel_arg_idx);
                let ele_ty = arg.r#type();
                let ele_size = crate::mlir::static_size_of(ele_ty) as u64;
                if ele_offset % ele_size != 0 {
                    self.emit_error(
                        format!(
                            "Unsupported indirect argument of type {} has invalid offset {} and size {}. Please reorder fields from largest alignment to smallest.",
                            ty_layout.ty, ele_offset, ele_size
                        ),
                        dev_instance.def.default_span(self.tcx),
                    );
                }
                let ele_ptr = builder.inbounds_gep(
                    ele_ty,
                    arg_ptr,
                    &[self.const_usize(ele_offset / ele_size)],
                );

                let elem = builder.store(arg, ele_ptr, align);
                kernel_arg_idx += 1;
                ele_offset += ele_size;
            }
            call_args.push(arg_ptr);
            dev_arg_idx = i + 1;
        }
        for i in kernel_arg_idx..fn_type.input_count() {
            call_args.push(builder.get_param(i));
        }
        debug!(
            "define_indirect_if_needed fn_type = {:#?}, dev_fn_type = {:#?}, indirect_args = {:#?} call_args = {:#?}",
            fn_type, dev_fn_type, indirect_args, call_args
        );
        let fn_abi = self.fn_abi_of_instance(dev_instance, rustc_middle::ty::List::empty());
        let llfn = self.to_mir_func_const(dev_instance, None);
        builder.call(
            dev_fn_type.into(),
            None,
            Some(fn_abi),
            llfn,
            &call_args,
            None,
            Some(dev_instance),
        );
        builder.ret_void();

        // self.fn_shared_memory_size.write().unwrap().insert(entry_sym, 0);
        // Indirect kernel should refer to the device function to get the shared memory size.
    }

    pub fn to_mir_func_decl(
        &self,
        instance: Instance<'tcx>,
        visibility: MLIRVisibility,
    ) -> melior::ir::OperationRef<'ml, 'a> {
        let tcx = self.tcx();
        let mlir_ctx = self.mlir_ctx;
        let sym = self.sanitized_symbol_name(instance);
        let def_id: rustc_hir::def_id::DefId = instance.def_id();
        {
            let fn_db = self.fn_db.read().unwrap();
            if fn_db.contains_key(&sym) {
                return fn_db[&sym];
            }
        }
        let mut gpu_attrs = self.gpu_attrs(&instance);
        let need_def = gpu_attrs.is_gpu_related();
        let span = instance.def.default_span(tcx);
        let location: melior::ir::Location<'ml> = self.to_mlir_loc(span);

        let fn_abi = self.fn_abi_of_instance(instance, rustc_middle::ty::List::empty());
        let (ftype, real_entry_ftype) = self
            .fn_abi_to_fn_type(fn_abi, gpu_attrs.kernel, gpu_attrs.ret_shared)
            .unwrap_or_else(|e| self.emit_error(e, span));
        let fn_sym = if real_entry_ftype.is_some() && gpu_attrs.kernel {
            gpu_attrs.kernel = false;
            gpu_attrs.device = true;
            let dev_sym = format!("{INDIRECT}{}", sym);
            debug!(
                "Function `{}` is a kernel entry function, but it has indirect arguments. \
                It will be defined as a device function with name `{}`",
                sym, dev_sym
            );
            let (new_ftype, indirect_args) = real_entry_ftype.unwrap();
            *self.indirect_entry.lock().unwrap() = Some(IndirectEntry {
                entry_sym: sym.clone(),
                fn_type: new_ftype,
                indirect_args,
                dev_fn_type: ftype,
                dev_instance: instance,
                location,
            });
            let fn_db = self.fn_db.read().unwrap();
            if fn_db.contains_key(&dev_sym) {
                return fn_db[&dev_sym];
            }
            dev_sym
        } else {
            sym.clone()
        };
        let fn_sym_attr = melior::ir::attribute::StringAttribute::new(mlir_ctx, &fn_sym);
        /*if is_impl_of_trait_method(
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
            gpu_attrs.gpu_item = Some(GpuItem::UniqueChunk);
            //self.type_func(&[], self.type_empty())
        }*/
        let fn_type = melior::ir::attribute::TypeAttribute::new(ftype.into());
        let mut operation: Operation<'ml> = if !gpu_attrs.kernel {
            melior::dialect::ods::func::func(
                mlir_ctx,
                melior::ir::Region::new(),
                // accepts a StringAttribute which is the function name.
                fn_sym_attr,
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
            let mut op: Operation<'ml> = gpu_op.into();
            op.set_attribute("kernel", melior::ir::Attribute::unit(self.mlir_ctx));
            op.set_attribute(crate::mlir::SYM_NAME_SYM, fn_sym_attr.into());

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
        debug!("append operation to block {} {:?}", operation, fn_sym);
        let op = body.append_operation(operation);
        self.fn_db.write().unwrap().insert(sym, op);
        self.fn_db.write().unwrap().insert(fn_sym.clone(), op);
        self.fn_shared_memory_size.write().unwrap().insert(fn_sym.clone(), 0);
        op
    }
}

/*
pub(crate) fn build_shared_size<'tcx: 'a, 'ml, 'a, 'val: 'a>(
    cx: &'val GPUCodegenContext<'tcx, 'ml, 'a>,
    instance: &Instance<'tcx>,
) {
    let name = cx.tcx.symbol_name(*instance).name.to_string();
    let shared = cx.fn_shared_memory_size.read().unwrap()[&name];
    use rustc_codegen_ssa_gpu::traits::BaseTypeCodegenMethods;
    let fn_type = melior::ir::attribute::TypeAttribute::new(
        FunctionType::new(cx.mlir_ctx, &[], &[cx.type_i32()]).into(),
    );
    let fn_sym = StringAttribute::new(cx.mlir_ctx, format!("shared_memory_{}", name).as_str());
    let region = melior::ir::Region::new();
    let block = melior::ir::Block::new(&[]);

    use melior::ir::RegionLike;
    use rustc_codegen_ssa_gpu::traits::BuilderMethods;
    let llbb = region.append_block(melior::ir::Block::new(&[]));
    let new_fn = melior::dialect::ods::func::func(
        cx.mlir_ctx,
        region,
        // accepts a StringAttribute which is the function name.
        fn_sym,
        // A type attribute, defining the function signature.
        fn_type,
        cx.unknown_loc(),
    );
    let op = cx.mlir_body(false).append_operation(new_fn.into());
    let mut builder: crate::builder::GpuBuilder<'tcx, 'ml, 'val> =
        crate::builder::GpuBuilder::build(cx, llbb);
    let size = builder.const_value(shared, cx.type_i32());
    builder.ret(size);
}
*/
