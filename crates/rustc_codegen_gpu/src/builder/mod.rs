mod abi;
mod asm;
mod coverage;
mod debug;
mod dim;
mod intrinsic;
mod print;
mod vector;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;

use melior::dialect::memref as mlir_memref;
use melior::ir::attribute::{DenseI64ArrayAttribute, IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike};
use melior::ir::r#type::{self as mlir_type, FunctionType, MemRefType};
use melior::ir::{
    self as mlir_ir, Attribute, BlockLike, Location, RegionLike, RegionRef, ShapedTypeLike,
    TypeLike, Value, ValueLike,
};
use rustc_abi::BackendRepr;
use rustc_codegen_ssa_gpu::common::IntPredicate;
use rustc_codegen_ssa_gpu::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa_gpu::traits::{
    BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, OverflowOp, StaticBuilderMethods,
};
use rustc_middle::ty::{AtomicOrdering, Instance};
use rustc_span::Span;
use tracing::{debug, trace};

use crate::attr::GpuItem;
use crate::context::GPUCodegenContext;
use crate::mlir::gpu::{DimFn, NonDimFn, all_reduce, subgroup_reduce};
use crate::mlir::memref::{
    StridedMetaDataResults, extract_strided_metadata, extract_strided_metadata_results,
};
use crate::mlir::{BUILTIN_SYM, MLIROpHelpers, ValueToOpRef, int_width};
use crate::rustc_middle::ty::layout::LayoutOf;

#[derive(Debug)]
pub(crate) struct GpuBuilderState<'ml, 'a> {
    pub attrs: Vec<mlir_ir::Attribute<'ml>>,
    pub args: HashMap<crate::attr::GpuItem, Vec<mlir_ir::Value<'ml, 'a>>>,
    pub inside_gpu_scope: bool,
}

impl<'ml, 'a> GpuBuilderState<'ml, 'a> {
    pub fn new() -> Self {
        Self { attrs: vec![], args: HashMap::new(), inside_gpu_scope: false }
    }
}

impl<'ml, 'a> Drop for GpuBuilderState<'ml, 'a> {
    fn drop(&mut self) {
        /*assert!(
            self.attrs.is_empty(),
            "attrs should be empty when dropping GpuBuilderState: {:?}",
            self.attrs
        );*/
    }
}

#[cfg(feature = "inplace_bound_check")]
#[derive(Debug)]
struct InplaceBoundCheckData<'ml, 'a> {
    pub cond: mlir_ir::Value<'ml, 'a>,
    pub idx: Vec<mlir_ir::Value<'ml, 'a>>,
    // If not find inbound_gep that has matched idx,
    // we add select_ptr for all ld/store when loc matches
    pub checked_by_loc: bool,
}

#[cfg(feature = "inplace_bound_check")]
impl<'ml, 'a> InplaceBoundCheckData<'ml, 'a> {
    pub fn new(cond: mlir_ir::Value<'ml, 'a>, idx: mlir_ir::Value<'ml, 'a>) -> Self {
        Self { cond, idx: vec![idx], checked_by_loc: false }
    }
}

#[derive(Debug)]
pub(crate) struct GpuBuilder<'tcx, 'ml, 'a> {
    pub cx: &'a GPUCodegenContext<'tcx, 'ml, 'a>,
    pub name: String,
    pub cur_block: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::BasicBlock,
    pub cur_span: rustc_span::Span,
    pub span_to_type: HashMap<rustc_span::Span, mlir_type::Type<'ml>>,
    pub op_to_extra_values: HashMap<String, Vec<mlir_ir::Value<'ml, 'a>>>,
    pub extra_state: GpuBuilderState<'ml, 'a>,
    san_dummy: Option<mlir_ir::Value<'ml, 'a>>,
    dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
    #[cfg(feature = "inplace_bound_check")]
    valid_mem_access: Option<InplaceBoundCheckData<'ml, 'a>>,
    const_values: HashMap<String, mlir_ir::Value<'ml, 'a>>,
}

impl<'tcx, 'ml, 'a> Drop for GpuBuilder<'tcx, 'ml, 'a> {
    fn drop(&mut self) {
        assert!(
            self.extra_state.attrs.is_empty(),
            "attrs should be empty when dropping GpuBuilder: {:?}",
            self.extra_state.attrs
        );

        #[cfg(feature = "inplace_bound_check")]
        assert!(
            self.valid_mem_access.iter().all(|v| v.checked_by_loc),
            "valid_mem_access should be None {:?} at {}",
            self.valid_mem_access,
            crate::mlir::value_loc(self.valid_mem_access.as_ref().unwrap().idx[0]).unwrap()
        );
        *self.cx.builder.write().unwrap() = None;
    }
}
impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub fn cur_loc(&self) -> Location<'ml> {
        self.cx.to_mlir_loc(self.cur_span)
    }

    /*pub fn cur_block(&self) -> &'a mlir_ir::Block<'ml> {
        unsafe { self.cur_block.to_ref() }
    }*/

    pub fn const_value(
        &mut self,
        val: impl std::fmt::Display,
        ty: mlir_ir::Type<'ml>,
    ) -> mlir_ir::Value<'ml, 'a> {
        let attr_str = format!("{val} : {ty}");
        if let Some(val) = self.const_values.get(&attr_str) {
            return *val;
        }
        let const_val = self.mlir_const_val_from_type(val, ty, self.cur_block);
        self.const_values.insert(attr_str, const_val);
        const_val
    }

    fn get_const_str(&self, arg: mlir_ir::Value<'ml, 'a>) -> Result<String, ()> {
        let name = arg.to_get_global_name();
        if name.is_err() {
            return Err(());
        }
        let global_name = name.unwrap().value().to_string();
        let bytes = self.get_const_bytes_by_name(global_name.as_str());
        if let Ok(bytes) = std::str::from_utf8(bytes) { Ok(bytes.to_string()) } else { Err(()) }
    }

    fn get_const_attribute(
        &self,
        arg: mlir_ir::Value<'ml, 'a>,
    ) -> Result<mlir_ir::Attribute<'ml>, ()> {
        self.get_const_str(arg).map_or(Err(()), |bytes| {
            melior::ir::Attribute::parse(self.mlir_ctx, bytes.as_str()).ok_or(())
        })
    }

    fn static_size_of(&self, ty: mlir_ir::Type<'_>) -> usize {
        crate::mlir::static_size_of(ty)
    }

    fn call_op(
        &mut self,
        fn_ptr_value: melior::ir::Value<'ml, 'a>,
        instance: Option<Instance<'tcx>>,
        args: &[melior::ir::Value<'ml, 'a>],
        ftype: Option<FunctionType<'ml>>,
        span: Span,
    ) -> Result<Option<melior::ir::OperationRef<'ml, 'a>>, melior::Error> {
        let mut return_type = vec![];
        let mut input_types = vec![];
        let fn_sym_ptr = fn_ptr_value.to_func_sym();
        let builtin_sym = fn_ptr_value.get_op_attr::<StringAttribute>(BUILTIN_SYM);
        if let Some(ftype) = ftype {
            for i in 0..ftype.result_count() {
                return_type.push(ftype.result(i).unwrap());
            }
            for i in 0..ftype.input_count() {
                input_types.push(ftype.input(i).unwrap());
            }
        } else {
            panic!("")
        }
        let loc = self.to_mlir_loc(span);
        if let Ok(builtin_sym) = builtin_sym {
            if let Ok(gpu_item) = GpuItem::try_from(builtin_sym.value()) {
                return self.call_gpu_builtin_operation(
                    gpu_item,
                    fn_ptr_value,
                    instance,
                    args,
                    &return_type,
                    span,
                );
            } else {
                panic!("{}", builtin_sym.value());
            }
        }
        let args = &args
            .iter()
            .zip(input_types.iter())
            .map(|(v, t)| self.use_value_as_ty(*v, *t))
            .collect::<Vec<_>>();
        if let Ok(fn_sym_ptr) = fn_sym_ptr {
            let op_ref = self.append_op(melior::dialect::func::call(
                self.mlir_ctx,
                fn_sym_ptr,
                args,
                &return_type,
                loc,
            ));
            Ok(Some(op_ref))
        } else {
            let op_ref = self.append_op(melior::dialect::func::call_indirect(
                fn_ptr_value,
                args,
                &return_type,
                loc,
            ));
            Ok(Some(op_ref))
        }
    }

    fn call_gpu_builtin_operation(
        &mut self,
        gpu_item: GpuItem,
        _fn_ptr_value: melior::ir::Value<'ml, 'a>,
        instance: Option<Instance<'tcx>>,
        args: &[melior::ir::Value<'ml, 'a>],
        return_types: &[melior::ir::Type<'ml>],
        span: Span,
    ) -> Result<Option<melior::ir::OperationRef<'ml, 'a>>, melior::Error> {
        let loc = self.to_mlir_loc(span);
        let get_generic_type = || {
            let mut generic_types = vec![];
            if let Some(instance) = instance {
                for arg in instance.args.iter() {
                    if let rustc_type_ir::GenericArgKind::Type(ty) = arg.kind() {
                        generic_types.push(ty);
                    }
                }
            }
            generic_types
        };

        let get_generic_const = || {
            let mut generic_types = vec![];
            if let Some(instance) = instance {
                for arg in instance.args.iter() {
                    if let rustc_type_ir::GenericArgKind::Const(c) = arg.kind() {
                        generic_types.push(c);
                    }
                }
            }
            generic_types
        };
        let get_closures = || {
            let mut closure_ptrs = vec![];
            for ty in get_generic_type() {
                if let Some(c) = self.ty_to_closure(&ty) {
                    closure_ptrs.push(c);
                }
            }
            closure_ptrs
        };

        match gpu_item {
            GpuItem::AllReduce => {
                assert!(self.extra_state.attrs.len() == 1);
                let op_attr = self.extra_state.attrs.pop().unwrap();
                let op = self.append_op(all_reduce(self.mlir_ctx, args[0], op_attr, true, loc));
                Ok(Some(op))
            }
            GpuItem::ThreadId => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                assert!(return_types.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                Ok(Some(self.append_op(DimFn::ThreadId.build(self.mlir_ctx, dimention, loc))))
            }
            GpuItem::GlobalThreadId => {
                let dimention = self.extra_state.attrs.pop().unwrap();
                Ok(Some(self.append_op(DimFn::GlobalThreadId.build(self.mlir_ctx, dimention, loc))))
            }
            GpuItem::BlockId => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                Ok(Some(self.append_op(DimFn::BlockId.build(self.mlir_ctx, dimention, loc))))
            }
            GpuItem::LaneId => {
                assert!(self.extra_state.attrs.is_empty());
                Ok(Some(self.append_op(NonDimFn::LaneId.build(self.mlir_ctx, loc))))
            }
            GpuItem::SubgroupId => {
                assert!(self.extra_state.attrs.is_empty());
                Ok(Some(self.append_op(NonDimFn::SubgroupId.build(self.mlir_ctx, loc))))
            }
            GpuItem::SubgroupSize => {
                assert!(self.extra_state.attrs.is_empty());
                Ok(Some(self.append_op(NonDimFn::SubgroupSize.build(self.mlir_ctx, loc))))
            }
            GpuItem::SubgroupReduce => {
                let Ok(op_str) = self.get_const_str(args[1]) else {
                    let err = format!("{:?} must take a constant str as op", gpu_item);
                    self.emit_error(err.clone(), span);
                };
                let op_attr = Attribute::parse(self.mlir_ctx, &op_str)
                    .unwrap_or_else(|| panic!("failed to parse op attribute {}", op_str));
                let generic_consts = get_generic_const();
                let c_size = generic_consts[0].try_to_target_usize(self.tcx).unwrap() as _;
                let c_stride = generic_consts[1].try_to_target_usize(self.tcx).unwrap() as _;
                if args[0].r#type() != self.type_i32() {
                    self.emit_error(
                        format!(
                            "gpu.subgroup_reduce: args[0] must be i32, but got {}",
                            args[0].r#type()
                        ),
                        span,
                    );
                }
                if c_size != 32 && c_stride != 0 {
                    self.emit_error(
                        format!("Due to limited support in MLIR gpu::subgroup_reduce now supports cluster_size = warp_size = 32, and stride_size = 1, but c_size = {}, c_stride = {}. consider using nvcc_redux_sync or reduce_with_shuffle", c_size, c_stride),
                        span,
                    );
                }
                if (c_size > 32 || c_stride > 32) || c_size == 0 || c_stride == 0 {
                    self.emit_error(
                        format!(
                            "gpu.subgroup_reduce: c_size = {}, c_stride = {} must be in range [1, 32]",
                            c_size, c_stride
                        ),
                        span,
                    );
                }
                assert!(c_stride <= 32);
                let op =
                    subgroup_reduce(self.mlir_ctx, args[0], op_attr, true, c_size, c_stride, loc);
                Ok(Some(self.append_op(op)))
            }
            GpuItem::Shuffle => {
                let Ok(op_str) = self.get_const_str(args[3]) else {
                    let err = format!("{:?} must take a constant str as op", gpu_item);
                    self.emit_error(err.clone(), span);
                };
                let op_attr = Attribute::parse(self.mlir_ctx, &op_str)
                    .unwrap_or_else(|| panic!("failed to parse op attribute {}", op_str));
                let val: Value<'_, '_> = args[0];
                let offset = args[1];
                let width = args[2];
                assert!(offset.r#type() == self.type_i32());
                assert!(width.r#type() == self.type_i32());
                let op = crate::mlir::gpu::shuffle(self.mlir_ctx, val, offset, width, op_attr, loc);
                Ok(Some(self.append_op(op)))
            }
            GpuItem::NvvmReduxSync => {
                let Ok(op_str) = self.get_const_str(args[2]) else {
                    let err = format!("{:?} must take a constant str as op", gpu_item);
                    self.emit_error(err.clone(), span);
                };
                let op_attr = Attribute::parse(self.mlir_ctx, &op_str)
                    .unwrap_or_else(|| panic!("failed to parse op attribute {}", op_str));
                let mask = args[1];
                assert!(mask.r#type() == self.type_i32());
                let abs = false;
                let val = args[0];
                let is_float = val.r#type().is_float();
                let op = crate::mlir::gpu::nvvm_redux_sync(
                    self.mlir_ctx,
                    val,
                    mask,
                    op_attr,
                    abs,
                    is_float,
                    loc,
                );
                Ok(Some(self.append_op(op)))
            }
            GpuItem::BlockDim => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                Ok(Some(self.append_op(DimFn::BlockDim.build(self.mlir_ctx, dimention, loc))))
            }
            GpuItem::GridDim => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                Ok(Some(self.append_op(DimFn::GridDim.build(self.mlir_ctx, dimention, loc))))
            }
            GpuItem::PrintArgs => {
                // printf function should starts with a format passed by add_mlir_string_attr
                // args can be passed to printf as a list of values.
                // printf ends with an empty printf
                assert!(
                    args.len() == 3,
                    "gpu.print_args must take (val, str, str_len) arguments but got {:?}",
                    args
                );
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.extra_state.args.entry(gpu_item.clone())
                {
                    e.insert(args[0..2].to_vec());
                } else {
                    self.extra_state.args.get_mut(&gpu_item).unwrap().extend(&args[0..2]);
                }
                Ok(None)
            }
            GpuItem::PrintFormat => {
                let Ok(format) = self.get_const_str(args[0]) else {
                    let err = format!("{:?} must take a constant str as format", gpu_item);
                    self.emit_error(err.clone(), span);
                };
                let mut args: Vec<Value<'ml, 'a>> = vec![];
                let mut holders = vec![];
                if let Some(extra) = self.extra_state.args.remove(&GpuItem::PrintArgs) {
                    let mut extra_iter = extra.iter();
                    while let Some(arg) = extra_iter.next() {
                        args.push(*arg);
                        holders.push(
                            extra_iter.next().and_then(|v| self.get_const_str(*v).ok()).unwrap(),
                        );
                    }
                };
                let format = print::println_fmt_to_prinf_fmt(format.as_str(), &holders)
                    .unwrap_or_else(|e| {
                        self.emit_error(e.clone(), span);
                    });

                let fmt_attr = StringAttribute::new(self.mlir_ctx, &format);
                let op_ref = self.append_op(
                    melior::dialect::ods::gpu::printf(self.mlir_ctx, &args, fmt_attr, loc).into(),
                );
                Ok(Some(op_ref))
            }
            GpuItem::AddStringAttr => {
                // args must be a const string.
                let arg = args[0];
                let err_msg =
                    || format!("{:?} must take valid string as MLIR attritutes", gpu_item);
                if let Ok(attr) = self.get_const_attribute(arg) {
                    self.extra_state.attrs.push(attr);
                } else {
                    self.emit_error(err_msg(), span);
                }
                Ok(None)
            }
            GpuItem::Scope => {
                use rustc_codegen_ssa_gpu::traits::MiscCodegenMethods;
                let closure = get_closures()[0];
                self.extra_state.inside_gpu_scope = true;
                let fn_ptr = self.get_fn_addr(closure);
                let fn_type: FunctionType<'ml> = fn_ptr.r#type().try_into().unwrap();
                debug!("gpu.scope args: {:?} {} {}", args, fn_type, fn_type.input_count());
                let extra_arg_len = fn_type.input_count() - 1;
                assert!(extra_arg_len <= args.len());
                let start = args.len() - extra_arg_len;
                let mut closure_args = vec![];
                closure_args.extend(&args[start..start + extra_arg_len]);
                closure_args.push(self.alloca_san_dummy()); // &ThreadScope
                self.call_op(fn_ptr, None, &closure_args, Some(fn_type), span)
            }
            GpuItem::Launch => {
                panic!("gpu.launch is not yet implemented");
            }
            GpuItem::NewChunk => {
                // Not a builtin.
                unreachable!();
            }
            GpuItem::UniqueChunk => {
                debug!("gpu.iter_next args: {:?}", args);
                let arg = args[0];
                let ptr = self.load(
                    self.type_memref(self.type_i8(), &[1], None, None),
                    arg,
                    rustc_abi::Align::EIGHT,
                );
                let one = self.const_value(1, self.type_index());
                let size_ptr = self.inbounds_gep(self.type_index(), arg, &[one]);
                let size = self.load(self.type_index(), size_ptr, rustc_abi::Align::EIGHT);
                let two = self.const_value(2, self.type_index());
                let window_ptr = self.inbounds_gep(self.type_i64(), arg, &[two]);
                let window = self.load(self.type_i64(), window_ptr, rustc_abi::Align::EIGHT);
                let three = self.const_value(3, self.type_index());
                let index_ptr = self.inbounds_gep(self.type_index(), arg, &[three]);
                let index = self.load(self.type_index(), index_ptr, rustc_abi::Align::EIGHT);
                let offset = self.mul(window, index);
                self.call_gpu_builtin_operation(
                    GpuItem::SubsliceMut,
                    _fn_ptr_value,
                    instance,
                    &[ptr, offset, index, window],
                    return_types,
                    span,
                )
            }
            GpuItem::SyncThreads => {
                self.append_op(melior::dialect::ods::gpu::barrier(self.mlir_ctx, loc).into());
                Ok(None)
            }
            GpuItem::Subslice | GpuItem::SubsliceMut => {
                trace!("gpu.subslice(_mut) args: {:?}", args);
                // args[0]: original:      memref<size xi8>
                // args[1]: original_size: i64
                // args[2]: offset:        index
                // args[3]: window:        i64

                // Note that there is no actual difference between the subslice and a
                // subslice_mut at the MLIR level

                let original = args[0];
                let original_size = args[1];
                let offset = args[2];
                let window = args[3];

                // 1. Bound check: offset + window <= original_size
                //    a.k.a. offset + window - 1 < original size
                //    The subslice must fit within the range of the original buffer
                let index_int_upper = self.add(window, offset);
                let one = self.const_value(1, self.type_index());
                let index_int_upper = self.sub(index_int_upper, one);
                self.emit_bound_check(index_int_upper, original_size, self.san_dummy.unwrap());
                self.emit_bound_check(offset, original_size, self.san_dummy.unwrap());

                // 2. Build the subslice: Done by transforming memref<1xi8> into the
                //    strided form using subview. Shortcut to use inbounds_gep
                //    IMPORTANT: ALL OFFSETS IN memref<size xi8> MUST BE BYTE-BASED,
                //               NOT ELEMENT-BASED
                let slice_ty = get_generic_type()[0];
                let slice_ty_layout = self.cx.layout_of(slice_ty);
                let slice_type = self.mlir_type(slice_ty_layout, false);

                // offset needs to time size of slice_type
                let indices = vec![offset; 1];

                let res = self.inbounds_gep(slice_type, original, &indices);

                // 3. Build the 'pair' of memref and i64
                //    IMPORTANT: WINDOW HERE IS STILL ELEMENT-BASED. BYTE-BASDE IS ONLY
                //               USED FOR MEMREF!
                self.op_to_extra_values.insert(self.cur_loc().to_string(), vec![res, window]);

                Ok(None)
            }
            GpuItem::NewSharedMem => {
                // Do not init the content of the shared memory.
                Ok(None)
            }
            GpuItem::AtomicRMW => {
                trace!("gpu.atomic_rmw args: {:?}", args);
                // args[0]: ptr:      memref<1xi8>
                // args[1]: val:      value, can be any thing... (f32, i32, ...)
                // args[2]: kind:     string attribute

                let ptr = args[0];
                let val = args[1];
                let kind = self.get_const_attribute(args[2]).unwrap_or_else(|_| {
                    self.emit_error(
                        "gpu.atomic_rmw must have a string attribute to specify the kind"
                            .to_string(),
                        span,
                    )
                });

                let offset = self.const_value(0, self.type_index());
                let indices_vec = vec![offset];
                let indices = &indices_vec;
                // Translate ptr into the correct form
                let ptr_memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
                let ptr_t = if self.mlir_element_type(ptr.r#type()) != val.r#type() {
                    let target_memref_ty =
                        self.type_memref(val.r#type(), &[1], None, ptr_memref_ty.memory_space());
                    self.mlir_memref_view(ptr, target_memref_ty, None, None)
                } else {
                    ptr
                };

                let mut atomic_rmw_op: melior::ir::Operation<'ml> =
                    melior::dialect::ods::memref::atomic_rmw(
                        self.mlir_ctx,
                        val.r#type(),
                        val,
                        ptr_t,
                        indices,
                        kind,
                        self.cur_loc(),
                    )
                    .into();
                atomic_rmw_op.set_attribute("kind", kind);

                Ok(Some(self.append_op(atomic_rmw_op)))
            }
            GpuItem::GetLocalMut2D => {
                // Not a builtin.
                unreachable!();
            }
            GpuItem::GetLocal2D => {
                // Not a builtin.
                unreachable!();
            }
            GpuItem::BuildSFI => {
                trace!("gpu.build_sfi args: {:?}", args);
                // args[0]: cond:    bool
                // args[1]: idx:     index

                let cond = args[0];
                let ptr = args[1];

                let ret = self.assert_ptr(cond, ptr);
                self.op_to_extra_values.insert(self.cur_loc().to_string(), vec![ret]);
                Ok(None)
            }
            GpuItem::DynamicShared => {
                // return (base + remain_size, valid_size)
                let ret_type = self.type_memref(
                    self.type_i8(),
                    &[crate::mlir::memref::dynamic_size()],
                    None,
                    Some(crate::mlir::memref::MemorySpace::DynamicShared.to_attr(self.mlir_ctx)),
                );
                let final_ret_type = self.type_memref(
                    self.type_i8(),
                    &[crate::mlir::memref::dynamic_size()],
                    None,
                    Some(crate::mlir::memref::MemorySpace::Shared.to_attr(self.mlir_ctx)),
                );
                let ret = self.append_op_res(
                    melior::dialect::ods::gpu::dynamic_shared_memory(
                        self.mlir_ctx,
                        ret_type,
                        self.cur_loc(),
                    )
                    .into(),
                );
                let op_ref = self.append_op(
                    melior::dialect::ods::memref::memory_space_cast(
                        self.mlir_ctx,
                        final_ret_type,
                        ret,
                        self.cur_loc(),
                    )
                    .into(),
                );
                Ok(Some(op_ref))
            }
            GpuItem::DeviceIntrinsic(name) => {
                // Not a builtin.
                let op = intrinsic::device_intrinsic(self, &name, args, self.cur_loc());
                Ok(Some(op))
            }
            GpuItem::Core(lang_item) => {
                // Not a builtin.
                let zero = self.const_value(0, self.type_i1());
                if lang_item.name().to_string().starts_with("panic") {
                    self.assert(zero, &lang_item.name().to_string());
                    Ok(None)
                } else {
                    todo!("unhandled core lang item: {:?}", lang_item);
                }
            }
            GpuItem::CoreFn(fn_name) => {
                if crate::attr::is_panic_function(fn_name.as_str()) {
                    let zero = self.const_value(0, self.type_i1());
                    self.assert(zero, &format!("{} abort at {}", fn_name, self.cur_loc()));
                    return Ok(None);
                }
                todo!();
            }
            GpuItem::DiagnoseOnly(name) => {
                unreachable!();
            }
        }
    }

    // Instead of generating things with a true type (e.g., memref<?xf32>), this function
    // should generate memref<size xi8> (e.g., 4xi8) with the offset given. This is
    // critical since otherwise we can't really change the type anymore.
    // IMPORTANT: THE OFFSETS IN THIS memref<size xi8> MUST BE SIZE-BASED!
    fn inbounds_gep_op(
        &mut self,
        ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
        ptr: melior::ir::Value<'ml, 'a>,
        indices: &[melior::ir::Value<'ml, 'a>],
    ) -> mlir_ir::Operation<'ml> {
        let ptr = self.use_value(ptr);
        if indices.len() != 1 {
            panic!("only supports single index");
        }
        let src_ty = ptr.r#type();
        let src_memref_ty = MemRefType::try_from(src_ty).unwrap();
        let base_ty = src_memref_ty.element();
        let mut indices = indices
            .iter()
            .map(|v| {
                if v.r#type().is_integer() || v.r#type().is_index() {
                    let v = self.use_value(*v);

                    let element_size =
                        self.const_value(self.static_size_of(ty) as u64, self.type_i64());
                    let byte_index = self.mul(v, element_size);

                    self.intcast(byte_index, self.type_index(), false).into()
                } else {
                    panic!("Must be int or index type");
                }
            })
            .collect::<Vec<_>>();
        // target_memref_ty here has to be i8, size is ty's size
        let target_memref_ty = MemRefType::try_from(self.type_memref(
            self.type_i8(),
            &[self.static_size_of(ty) as i64],
            None,
            src_memref_ty.memory_space(),
        ))
        .unwrap();
        // base_ty is always i8
        // size on the first stride represents the element size
        let mut sizes = vec![(self.static_size_of(ty) as i64).into()];
        let mut strides = vec![1.into()];
        let base_ty = target_memref_ty.element();
        if target_memref_ty.rank() > 1 {
            for i in 1..target_memref_ty.rank() {
                indices.push(0.into());
                sizes.push((target_memref_ty.dim_size(i).unwrap() as i64).into());
                strides.push(1.into());
                debug!(
                    "inbounds_gep_op: only supports 1D memref, but got rank {}, strides: {:?}",
                    target_memref_ty.rank(),
                    strides,
                );
            }
        }
        crate::mlir::memref::subview(
            self.mlir_ctx,
            base_ty,
            ptr,
            &indices,
            &sizes,
            &strides,
            self.cur_loc(),
        )
    }

    fn inbounds_gep_ret(
        &mut self,
        ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
        ptr: melior::ir::Value<'ml, 'a>,
        indices: &[melior::ir::Value<'ml, 'a>],
        check: bool,
    ) -> melior::ir::Value<'ml, 'a> {
        let op = self.inbounds_gep_op(ty, ptr, indices);
        let ptr = self.append_op_res(op);
        #[cfg(feature = "inplace_bound_check")]
        let ptr = if check && !self.disable_bound_check {
            self.select_ptr(ptr, Some(indices[0]))
        } else {
            ptr
        };
        ptr
    }
    /*fn size_of(&self, ty: mlir_ir::Type<'_>) -> melior::ir::Value<'ml, 'a> {
            if ty.is_integer() {
                self.const_value(self.cx.mlir_integer_width(ty) / 8, self.type_index())
            } else if ty.is_index() || ty.is_mem_ref(){
                let op = melior::dialect::ods::index::sizeof(self.mlir_ctx, self.type_index(), self.cur_loc());
                self.append_op_res(op.into())
            } else if ty.is_float() {
                self.const_value(self.mlir_float_width(ty) / 8, self.type_index())
            } else if ty.is_tuple() {
                let tuple = mlir_type::TupleType::try_from(ty).unwrap();
                let mut total_size = None;
                for i in 0..tuple.type_count() {
                    let size = self.size_of(tuple.r#type(i).unwrap());
                    if i > 0 {
                        let op = melior::dialect::arith::addi(total_size.unwrap(), size, self.cur_loc());
                        total_size = Some(self.append_op_res(op))
                    } else {
                        total_size = Some(size);
                    }
                }
                if total_size.is_none() {
                    self.const_value(0, self.type_index())
                } else {
                    total_size.unwrap()
                }
            } else if ty.is_ranked_tensor() {
                let ranked_ty = mlir_type::RankedTensorType::try_from(ty).unwrap();
                let ty = self.mlir_element_type(ty);
                let base_size = self.size_of(ty);
                todo!();
            } else {
                panic!("Unsupported type: {:?}", ty);
            }
    }*/

    fn get_type_by_span(&self, span: &rustc_span::Span) -> Option<mlir_type::Type<'ml>> {
        if self.span_to_type.contains_key(span) {
            Some(self.span_to_type[span])
        } else if self.cx.span_to_types.read().unwrap().contains_key(span) {
            Some(self.cx.span_to_types.read().unwrap()[span])
        } else {
            None
        }
    }

    pub fn use_value(
        &mut self,
        val: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Value,
    ) -> <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Value {
        if let Ok(op) = val.is_from_op(Some("arith.constant")) {
            let attr = op.attribute("value").unwrap();
            let attr_str = format!("{}", attr);
            self.const_values.get(&attr_str).cloned().unwrap_or_else(|| {
                let op = self.append_op((op).clone());
                let val = op.result(0).unwrap().into();
                self.const_values.insert(attr_str, val);
                val
            })
        } else if let Ok(op) = val.is_from_op(Some("memref.get_global")) {
            let op = self.append_op((op).clone());
            op.result(0).unwrap().into()
        } else if let Ok(op) = val.is_from_op(Some("poison.const_offset")) {
            let ptr: Value<'ml, 'a> = op.operand(0).unwrap();
            let offset: IntegerAttribute = op.attribute("offset").unwrap().try_into().unwrap();
            let ptr = self.use_value(ptr);
            self.append_op_res(crate::mlir::memref::subview(
                self.mlir_ctx,
                self.type_i8(),
                ptr,
                &[offset.value().into()],
                &[1.into()],
                &[1.into()],
                self.unknown_loc(),
            ))
        } else {
            val
        }
    }

    fn use_value_as_integer(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
        signed: bool,
    ) -> melior::ir::Value<'ml, 'a> {
        let val = self.use_value(val);
        let ty = val.r#type();
        if ty.is_index() { self.intcast(val, self.type_i64(), signed) } else { val }
    }

    // If the value is memref, use view to convert it to a memref offset = 0.
    fn use_value_as_ty(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
        dst_ty: melior::ir::Type<'ml>,
    ) -> melior::ir::Value<'ml, 'a> {
        let val = self.use_value(val);
        let ty = val.r#type();
        if ty == dst_ty {
            val
        } else if ty.is_mem_ref() && dst_ty.is_mem_ref() {
            // Should be i8 to i8 here
            self.mlir_memref_view(val, dst_ty, None, None)
        } else if ty.is_index() || ty.is_integer() || dst_ty.is_index() || dst_ty.is_integer() {
            self.intcast(val, dst_ty, false)
        } else {
            panic!("Cannot use value {:?} as type {:?}", val, dst_ty);
        }
    }

    pub fn extract_strided_metadata(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
    ) -> StridedMetaDataResults<'ml, 'a> {
        let ty = val.r#type();
        let op = extract_strided_metadata(self.cx.mlir_ctx, val, self.cur_loc());
        let op = self.append_op(op.into());
        let results = extract_strided_metadata_results(ty.try_into().unwrap(), op);
        results
    }

    pub fn mlir_memref_view(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
        dst_ty: melior::ir::Type<'ml>,
        byte_offset: Option<Value<'ml, 'a>>,
        dy_size: Option<Value<'ml, 'a>>,
    ) -> Value<'ml, 'a> {
        let mut val = val;
        let ty = val.r#type();
        assert!(ty.is_mem_ref());
        assert!(dst_ty.is_mem_ref());

        if dst_ty == ty {
            return val;
        }
        let dst_memref_ty: MemRefType<'ml> = dst_ty.try_into().expect("expected memref type");
        let memref_ty: MemRefType<'ml> = ty.try_into().expect("expected memref type");
        if memref_ty.memory_space() != dst_memref_ty.memory_space() {
            let memref_ty = self.memref_set_memory_space(memref_ty, dst_memref_ty.memory_space());
            val = self.append_op_res(
                melior::dialect::ods::memref::memory_space_cast(
                    self.mlir_ctx,
                    memref_ty.into(),
                    val,
                    self.cur_loc(),
                )
                .into(),
            );
            if val.r#type() == dst_ty {
                return val;
            }
            //self.memref_set_memory_space(dst_memref_ty, memref_ty.memory_space())
        }
        let layout = crate::mlir::attr::StridedLayoutAttribute::try_from(memref_ty.layout());
        let base_memref = val;
        let (base_memref, base_byte_offset) = match layout {
            Ok(layout) if layout.get_offset() != 0 => {
                // view cannot work on offset !=0;
                debug!("mlir_memref_view with offset != 0 casting {} to {}", val, dst_ty);
                let results = self.extract_strided_metadata(val);
                let element_ty = self.mlir_element_type(ty);

                if element_ty != self.type_i8() {
                    debug!("mlir_memref_view: element_ty is not type_i8");
                    let backtrace = std::backtrace::Backtrace::force_capture();
                    println!("{}", backtrace);
                }

                // The results.base_memref is memref<i8>,
                // but we need 1d memref<sizexi8> to use view
                // ensures this is 1d memref.
                assert!(results.sizes.len() == 1);
                let op = crate::mlir::memref::reinterpret_cast(
                    self.cx.mlir_ctx,
                    element_ty,
                    results.base_memref,
                    &[0.into()],
                    &results.sizes,
                    &results.strides,
                    self.cur_loc(),
                );
                debug!("base memref: {:?}", val);
                (self.append_op_res(op), results.byte_offset)
            }
            _ => (val, self.const_value(0, self.type_index())),
        };

        // If byte_offset is None, we use the base_byte_offset + byte_offset
        let byte_offset = if let Some(byte_offset) = byte_offset {
            self.add(byte_offset, base_byte_offset)
        } else {
            base_byte_offset
        };
        let dy_sizes = if let Some(dy_size) = dy_size {
            let dy_size = self.intcast(dy_size, self.type_index(), false);
            vec![dy_size]
        } else {
            vec![] // No dynamic sizes
        };
        let op = melior::dialect::memref::view(
            self.cx.mlir_ctx,
            base_memref,
            byte_offset,
            &dy_sizes,
            dst_memref_ty,
            self.cur_loc(),
        );
        self.append_op_res(op)
    }

    #[inline]
    fn append_fast_op_res(&self, mut op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        op.set_attribute(
            "fastmath",
            Attribute::parse(self.mlir_ctx, "#arith.fastmath<fast>").unwrap(),
        );
        self.append_op_res(op)
    }

    fn append_op_res(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        trace!("append_op_res: {:?}", op);
        let op_ref = self.cur_block.append_operation(op);
        op_ref.result(0).unwrap().into()
    }

    fn is_unreachable(&self) -> bool {
        false
    }

    fn append_op(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
        trace!("append_op: {:?}", op);
        if self.is_unreachable() {
            panic!("Cannot append operation to unreachable block");
        }
        self.cur_block.append_operation(op)
    }

    #[inline]
    fn append_fast_op(&self, mut op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
        op.set_attribute(
            "fastmath",
            Attribute::parse(self.mlir_ctx, "#arith.fastmath<fast>").unwrap(),
        );
        self.append_op(op)
    }

    fn ptrtollvmptr(&mut self, ptr: mlir_ir::Value<'ml, 'a>) -> mlir_ir::Value<'ml, 'a> {
        let memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
        let addr = self.ptrtoint(ptr, self.type_i64());
        let addr_space = if let Some(addr_space) = memref_ty.memory_space() {
            use crate::mlir::memref::MemorySpace;
            if addr_space == MemorySpace::Shared.to_attr(self.mlir_ctx) {
                3
            } else if addr_space == MemorySpace::Global.to_attr(self.mlir_ctx) {
                1
            } else {
                0
            }
        } else {
            0
        };
        let dest_ty = self.type_llvm_ptr(addr_space);
        let op = melior::dialect::ods::llvm::inttoptr(self.mlir_ctx, dest_ty, addr, self.cur_loc())
            .into();
        self.append_op_res(op)
    }

    #[allow(dead_code)]
    pub fn inside_gpu_mod(&self) -> bool {
        if let Some(op) = self.cur_block.parent_operation() { op.is_gpu_func() } else { false }
    }

    pub fn inside_kernel_func(&self) -> bool {
        if let Some(op) = self.cur_block.parent_operation() { op.is_kernel_func() } else { false }
    }

    fn mlir_load(
        &mut self,
        ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
        ptr: mlir_ir::Value<'ml, 'a>,
        indices: &[mlir_ir::Value<'ml, 'a>],
        align: rustc_abi::Align,
    ) -> mlir_ir::Value<'ml, 'a> {
        if ptr.r#type().is_llvm_pointer_type() {
            let op = melior::dialect::llvm::load(
                self.mlir_ctx,
                ptr,
                ty,
                self.cur_loc(),
                melior::dialect::llvm::LoadStoreOptions::new()
                    .align(Some(self.align_to_attr(align))),
            );
            return self.append_op_res(op);
        }
        // ptr is almost always memref<sizexi8>. Must be casted into ty
        // let ptr = self.mlir_cast_memref(ptr, MemRefType::new(ty, &[1], None, None).into());
        self.append_op_res(melior::dialect::memref::load(ptr, indices, self.cur_loc()))
    }

    fn get_params(&mut self) -> Vec<mlir_ir::Value<'ml, 'a>> {
        let mut ret = vec![];
        for i in 0..self.cur_block.argument_count() {
            let val = self.cur_block.argument(i).unwrap().into();
            ret.push(val);
        }
        ret
    }

    #[cfg(feature = "arith_immediate_bound_check")]
    fn emit_llvm_volatile_load(&mut self, ptr: mlir_ir::Value<'ml, 'a>) {
        // Only used by the bound check stuff
        let raw_ptr_op = melior::dialect::ods::memref::extract_aligned_pointer_as_index(
            self.mlir_ctx,
            self.type_index(),
            ptr,
            self.cur_loc(),
        )
        .into();
        let raw_ptr = self.append_op_res(raw_ptr_op);

        // And Ptr
        let llvm_raw_ptr_int = self.intcast(raw_ptr, self.type_i64(), false);

        // Ptr to LLVM
        let llvm_ptr = self.ptrtollvmptr(raw_ptr);

        // LLVM store with volatile
        let llvm_store_op = melior::dialect::llvm::load(
            self.mlir_ctx,
            llvm_ptr,
            self.type_i8(),
            self.cur_loc(),
            melior::dialect::llvm::LoadStoreOptions::new()
                .volatile(true)
                .align(Some(self.align_to_attr(rustc_abi::Align::EIGHT))),
        );

        self.append_op(llvm_store_op);
    }

    fn int_val_pair_cast(
        &mut self,
        lhs: mlir_ir::Value<'ml, 'a>,
        rhs: mlir_ir::Value<'ml, 'a>,
    ) -> (mlir_ir::Value<'ml, 'a>, mlir_ir::Value<'ml, 'a>) {
        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        let ty = if int_width(rhs_ty) < int_width(lhs_ty) { rhs_ty } else { lhs_ty };
        let lhs = self.use_value(lhs);
        let lhs = self.intcast(lhs, ty, false);
        let rhs = self.use_value(rhs);
        let rhs = self.intcast(rhs, ty, false);
        (lhs, rhs)
    }

    // Overflowing add that always return the wrapped value.
    // return wrapped value, overflow flag(bool)
    fn mlir_overflowing_add(
        &mut self,
        lhs: mlir_ir::Value<'ml, 'a>,
        rhs: mlir_ir::Value<'ml, 'a>,
        signed: Option<bool>,
    ) -> (mlir_ir::Value<'ml, 'a>, mlir_ir::Value<'ml, 'a>) {
        let (lhs, rhs) = self.int_val_pair_cast(lhs, rhs);
        if let Some(res) = crate::mlir::const_add(lhs, rhs, signed) {
            return (self.const_value(res, lhs.r#type()), self.const_value(0, self.type_i1()));
        }
        let op = melior::dialect::arith::addi(lhs, rhs, self.cur_loc());
        let ret = self.append_op_res(op);
        let mut overflow = self.const_value(0, self.type_i1());
        if self.tcx.sess.opts.cg.overflow_checks != Some(true) {
            return (ret, overflow);
        }
        let Some(signed) = signed else {
            // signless add does not care overflow.
            return (ret, overflow);
        };
        let cmp_op = if signed { IntPredicate::IntSGT } else { IntPredicate::IntUGT };
        let lhs_greater = self.icmp(cmp_op, lhs, ret);
        // overflow = (z < x) == (y > 0)
        overflow = if signed {
            let zero: Value<'_, '_> = self.const_value(0, rhs.r#type());
            let rhs_pos = self.icmp(IntPredicate::IntSGE, rhs, zero);
            self.icmp(IntPredicate::IntEQ, lhs_greater, rhs_pos)
        } else {
            lhs_greater
        };
        (ret, overflow)
    }

    fn mlir_overflowing_sub(
        &mut self,
        lhs: mlir_ir::Value<'ml, 'a>,
        rhs: mlir_ir::Value<'ml, 'a>,
        signed: Option<bool>,
    ) -> (mlir_ir::Value<'ml, 'a>, mlir_ir::Value<'ml, 'a>) {
        let (lhs, rhs) = self.int_val_pair_cast(lhs, rhs);
        if let Some(res) = crate::mlir::const_sub(lhs, rhs, signed) {
            return (self.const_value(res, lhs.r#type()), self.const_value(0, self.type_i1()));
        }
        let op = melior::dialect::arith::subi(lhs, rhs, self.cur_loc());
        let ret = self.append_op_res(op);
        let mut overflow = self.const_value(0, self.type_i1());
        if self.tcx.sess.opts.cg.overflow_checks != Some(true) {
            return (ret, overflow);
        }
        let Some(signed) = signed else {
            // signless sub does not care overflow.
            return (ret, overflow);
        };
        // if signed: overflow = (y < 0) == (z < x)
        // if unsigned: overflow = (x < y)
        overflow = if signed {
            let zero = self.const_value(0, rhs.r#type());
            let neg_rhs = self.icmp(IntPredicate::IntSLT, rhs, zero);
            let pos_rhs = self.icmp(IntPredicate::IntSLT, rhs, zero);
            let res_smaller_lhs = self.icmp(IntPredicate::IntSLT, ret, lhs);
            let res_greater_lhs = self.icmp(IntPredicate::IntSLT, ret, lhs);
            let c1 = self.and(res_smaller_lhs, neg_rhs);
            let c2 = self.and(res_greater_lhs, pos_rhs);
            self.or(c1, c2)
        } else {
            self.icmp(IntPredicate::IntULT, lhs, rhs)
        };
        (ret, overflow)
    }

    fn mlir_overflowing_mul(
        &mut self,
        lhs: mlir_ir::Value<'ml, 'a>,
        rhs: mlir_ir::Value<'ml, 'a>,
        signed: Option<bool>,
    ) -> (mlir_ir::Value<'ml, 'a>, mlir_ir::Value<'ml, 'a>) {
        let (lhs, rhs) = self.int_val_pair_cast(lhs, rhs);
        if let Some(res) = crate::mlir::const_mul(lhs, rhs, signed) {
            return (self.const_value(res, lhs.r#type()), self.const_value(0, self.type_i1()));
        }

        let op = melior::dialect::arith::muli(lhs, rhs, self.cur_loc());
        let ret = self.append_op_res(op);
        let mut overflow = self.const_value(0, self.type_i1());
        if self.tcx.sess.opts.cg.overflow_checks != Some(true) {
            return (ret, overflow);
        }
        let Some(signed) = signed else {
            // signless mul does not care overflow.
            return (ret, overflow);
        };
        let zero = self.const_value(0, rhs.r#type());
        // if signed, overflow = (((x ^ y ^ z) & (x ^ z)) < 0);
        // if unsigned, overflow = (y != 0 && x > z);
        overflow = if signed {
            let x_xor_y = self.xor(lhs, rhs);
            let x_xor_y_xor_z = self.xor(x_xor_y, ret);
            let x_xor_z = self.xor(lhs, ret);
            let check = self.and(x_xor_y_xor_z, x_xor_z);
            self.icmp(IntPredicate::IntSLT, check, zero)
        } else {
            let lhs_greater = self.icmp(IntPredicate::IntUGT, lhs, ret);
            let rhs_not_zero = self.icmp(IntPredicate::IntNE, rhs, zero);
            let lhs_not_zero = self.icmp(IntPredicate::IntNE, lhs, zero);
            let tmp = self.and(rhs_not_zero, lhs_greater);
            self.and(lhs_not_zero, tmp)
        };
        (ret, overflow)
    }

    fn load_with_check(
        &mut self,
        ty: mlir_ir::Type<'ml>,
        ptr: mlir_ir::Value<'ml, 'a>,
        align: rustc_abi::Align,
        check: bool,
    ) -> mlir_ir::Value<'ml, 'a> {
        // If the type is memref, we need to load the address (See store).
        let load_ty = if ty.is_mem_ref() { self.type_index() } else { ty };
        // ptr is almost always memref<sizexi8>. Must be casted into memref<ty>
        let src_ptr_ty = ptr.r#type();
        let src_memref_ty: MemRefType<'ml> = src_ptr_ty.try_into().unwrap();
        let ptr = if load_ty != src_memref_ty.element() {
            self.mlir_memref_view(
                ptr,
                self.type_memref(load_ty, &[1], None, src_memref_ty.memory_space()),
                None,
                None,
            )
        } else {
            ptr
        };

        #[cfg(feature = "inplace_bound_check")]
        let ptr = if check && !self.disable_bound_check { self.select_ptr(ptr, None) } else { ptr };
        let zero = self.const_value(0, self.type_index());
        let mut loaded = self.mlir_load(load_ty, ptr, &[zero], align);
        // If the type is memref, we need to cast the address to the correct type.
        if ty.is_mem_ref() {
            loaded = self.inttoptr(loaded, ty);
        }
        loaded
    }

    fn store_with_check(
        &mut self,
        val: mlir_ir::Value<'ml, 'a>,
        ptr: mlir_ir::Value<'ml, 'a>,
        align: rustc_abi::Align,
        check: bool,
    ) -> mlir_ir::Value<'ml, 'a> {
        if self.is_unreachable() {
            return val;
        }
        let val_ty = val.r#type();
        let ptr_ty = ptr.r#type();
        let mut val = self.use_value(val);

        // If the type is memref, we get the address and store it as 8-byte data
        // This is because Rust always treat Ref/Pointer as 8bytes while MemRef size > 24 bytes
        let store_ty = if val_ty.is_mem_ref() {
            val = self.ptrtoint(val, self.type_index());
            self.type_index()
        } else {
            val_ty
        };
        let const_idx = self.const_value(0, self.type_index());
        let dst_memref_ty = MemRefType::try_from(ptr_ty).unwrap();
        let target_memref_ty = self.type_memref(store_ty, &[1], None, dst_memref_ty.memory_space());
        let ptr = if self.mlir_element_type(ptr_ty) != store_ty {
            self.mlir_memref_view(ptr, target_memref_ty, None, None)
        } else {
            ptr
        };

        #[cfg(feature = "inplace_bound_check")]
        let ptr = if check && !self.disable_bound_check { self.select_ptr(ptr, None) } else { ptr };
        self.append_op(mlir_memref::store(val, ptr, &[const_idx], self.cur_loc()));
        val
    }

    fn null_ptr(&mut self, ty: mlir_ir::Type<'ml>) -> mlir_ir::Value<'ml, 'a> {
        let zero = self.const_value(0, self.type_i64());
        self.inttoptr(zero, ty)
    }

    #[cfg(feature = "inplace_bound_check")]
    /// Why this works?
    /// When generating basic block, ssa always use reverse postorder traversal,
    /// so we should see the assert_before_index before any inbound-gep/load/store using the index.
    /// If the inbound-gep uses the same index we need to check, that is exactly the place we want to
    /// do the select ptr.
    /// In some cases, do to stack allocation, we different expr of index in the inbound-gep, but they are
    /// actually the same value. So we need to check the location as well.
    /// In that case, we rely on load/store to do the select_ptr.
    /// For all load/store ops are from same location with the target bound check, we do the select_ptr.
    fn select_ptr(
        &mut self,
        ptr: mlir_ir::Value<'ml, 'a>,
        idx: Option<mlir_ir::Value<'ml, 'a>>,
    ) -> mlir_ir::Value<'ml, 'a> {
        if let Some(mut valid_mem_access) = self.valid_mem_access.take() {
            let same_loc = valid_mem_access.idx.iter().any(|i| {
                if let Some((file, line, col)) = crate::mlir::value_loc_decoded(*i) {
                    if let Some((file2, line2, col2)) = crate::mlir::loc_decoded(self.cur_loc()) {
                        file == file2 && line == line2 && col >= col2
                    } else {
                        false
                    }
                } else {
                    false
                }
            });
            let same_idx = idx.is_some()
                && valid_mem_access.idx.iter().any(|i| crate::mlir::same_value(*i, idx.unwrap()));
            let cond = valid_mem_access.cond;
            if same_loc {
                debug!(
                    "check by loc {:?} idx = {:?} {:?}",
                    self.valid_mem_access,
                    idx,
                    crate::mlir::loc_decoded(self.cur_loc())
                );
                valid_mem_access.checked_by_loc = true;
            }
            if !same_idx {
                self.valid_mem_access = Some(valid_mem_access);
            }

            if same_loc || same_idx {
                self.assert_ptr(cond, ptr)
            } else {
                debug!("skipped {:?} idx = {:?} {}", self.valid_mem_access, idx, self.cur_loc());
                ptr
            }
        } else {
            ptr
        }
    }

    fn assert_ptr(
        &mut self,
        cond: mlir_ir::Value<'ml, 'a>,
        ptr: mlir_ir::Value<'ml, 'a>,
    ) -> mlir_ir::Value<'ml, 'a> {
        let ty = ptr.r#type();
        let nullptr = self.null_ptr(ty);
        let cond = self.use_value_as_ty(cond, self.type_i1());
        let debug = self.tcx.sess.opts.debug_assertions;
        if debug {
            self.assert(cond, "assert_ptr");
            ptr
        } else if self.disable_bound_check {
            ptr
        } else {
            #[cfg(feature = "inplace_bound_check")]
            {
                self.select(cond, ptr, nullptr)
            }
            #[cfg(not(feature = "inplace_bound_check"))]
            {
                self.assert(cond, "assert_ptr");
                ptr
            }
        }
    }

    fn assert(&mut self, cond: mlir_ir::Value<'ml, 'a>, msg: &str) {
        if !self.tcx.sess.opts.debug_assertions {
            use rustc_codegen_ssa_gpu::traits::IntrinsicCallBuilderMethods;
            self.assume(cond);
            return;
        }
        // trap-based bound check is handled by assert op.
        let cond = self.use_value_as_ty(cond, self.type_i1());
        let msg = format!("{} {}", msg, self.cur_loc());
        let op = melior::dialect::cf::assert(self.mlir_ctx, cond, &msg, self.cur_loc());
        self.append_op(op);
    }

    #[cfg(not(any(
        feature = "arith_immediate_bound_check",
        feature = "inplace_bound_check",
        feature = "trap_bound_check"
    )))]
    fn _assert_before_index(
        &mut self,
        cond: mlir_ir::Value<'ml, 'a>,
        idx: mlir_ir::Value<'ml, 'a>,
    ) {
    }

    #[cfg(feature = "trap_bound_check")]
    fn _assert_before_index(
        &mut self,
        cond: mlir_ir::Value<'ml, 'a>,
        idx: mlir_ir::Value<'ml, 'a>,
    ) {
        // trap-based bound check is handled by assert op.
        self.assert(cond, "assert_before_index");
    }

    #[cfg(feature = "inplace_bound_check")]
    fn _assert_before_index(
        &mut self,
        cond: mlir_ir::Value<'ml, 'a>,
        idx: mlir_ir::Value<'ml, 'a>,
    ) {
        if let Some(mut valid_mem_access) = self.valid_mem_access.take() {
            valid_mem_access.idx.push(idx);
            let old_cond = valid_mem_access.cond;
            valid_mem_access.cond = self.and(old_cond, cond);
            self.valid_mem_access = Some(valid_mem_access);
        } else {
            self.valid_mem_access = Some(InplaceBoundCheckData::new(cond, idx));
        }
    }

    #[cfg(feature = "arith_immediate_bound_check")]
    fn _assert_before_index(
        &mut self,
        cond: mlir_ir::Value<'ml, 'a>,
        idx: mlir_ir::Value<'ml, 'a>,
    ) {
        let ty = self.san_dummy.unwrap().r#type();
        let nullptr = self.null_ptr(ty);
        let ptr = self.select(cond, self.san_dummy.unwrap(), nullptr);
        self.emit_llvm_volatile_load(ptr);
    }

    fn assert_before_index(&mut self, cond: mlir_ir::Value<'ml, 'a>, idx: mlir_ir::Value<'ml, 'a>) {
        let cond = self.use_value_as_ty(cond, self.type_i1());
        self._assert_before_index(cond, idx);
    }
}

impl<'tcx, 'ml, 'a> BackendTypes for GpuBuilder<'tcx, 'ml, 'a> {
    type Value = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Value;

    type Metadata = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Metadata;

    type Function = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Function;

    type BasicBlock = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::BasicBlock;

    type Type = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Type;
    type Funclet = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::Funclet;
    type DIScope = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::DIScope;
    type DILocation = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::DILocation;
    type DIVariable = <GPUCodegenContext<'tcx, 'ml, 'a> as BackendTypes>::DIVariable;
}

impl<'tcx, 'ml, 'a> StaticBuilderMethods for GpuBuilder<'tcx, 'ml, 'a> {
    fn get_static(&mut self, def_id: rustc_hir::def_id::DefId) -> Self::Value {
        todo!()
    }
}

pub fn append_block<'tcx, 'ml, 'val, 'a>(
    cx: &GPUCodegenContext<'tcx, 'ml, 'val>,
    llfn: mlir_ir::operation::OperationRef<'ml, 'a>,
    name: &str,
) -> mlir_ir::block::BlockRef<'ml, '_> {
    let name = rustc_data_structures::small_c_str::SmallCStr::new(name);
    let region: RegionRef<'ml, 'a> = unsafe { llfn.to_ref() }.region(0).unwrap();
    let types = llfn.get_op_operands_types();
    let block: mlir_ir::BlockRef<'ml, '_> = region.append_block(melior::ir::Block::new(
        &types.iter().map(|t| (*t, Location::unknown(cx.mlir_ctx))).collect::<Vec<_>>(),
    ));
    block
}

impl<'tcx, 'ml, 'a> Deref for GpuBuilder<'tcx, 'ml, 'a> {
    fn deref(&self) -> &Self::Target {
        *self.cx.builder.write().unwrap() = Some(crate::context::BuilderInfo {
            name: self.name.to_string(),
            cur_block: self.cur_block,
            cur_span: self.cur_span,
        });
        self.cx
    }

    type Target = GPUCodegenContext<'tcx, 'ml, 'a>;
}

impl<'tcx: 'a, 'ml: 'a, 'a: 'val, 'val: 'a> BuilderMethods<'a, 'tcx>
    for GpuBuilder<'tcx, 'ml, 'val>
{
    type CodegenCx = GPUCodegenContext<'tcx, 'ml, 'val>;

    fn build(cx: &'a Self::CodegenCx, llbb: Self::BasicBlock) -> Self {
        let sym = StringAttribute::try_from(
            llbb.parent_operation().unwrap().attribute("sym_name").unwrap(),
        )
        .unwrap();
        *cx.builder.write().unwrap() = None;
        let mut builder = Self {
            cx,
            name: sym.value().to_string(),
            cur_block: llbb,
            cur_span: rustc_span::DUMMY_SP,
            extra_state: GpuBuilderState::new(),
            dummy: PhantomData,
            span_to_type: HashMap::new(),
            op_to_extra_values: HashMap::new(),
            san_dummy: None,
            #[cfg(feature = "inplace_bound_check")]
            valid_mem_access: None,
            const_values: HashMap::new(),
        };
        builder.add_dim_assumptions();
        builder
    }

    fn build_with_san_dummy(
        cx: &'a Self::CodegenCx,
        llbb: Self::BasicBlock,
        san_dummy: Self::Value,
    ) -> Self {
        let mut builder = Self::build(cx, llbb);
        builder.san_dummy = Some(san_dummy);
        builder
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        self.cur_block
    }

    fn set_span(&mut self, span: rustc_span::Span) {
        self.cur_span = span;
        debug!("set span {:?}", span);
    }

    fn append_block(
        cx: &'a Self::CodegenCx,
        llfn: mlir_ir::operation::OperationRef<'ml, 'a>,
        name: &str,
    ) -> Self::BasicBlock {
        debug!("append_block: {:?}", name);
        let name = rustc_data_structures::small_c_str::SmallCStr::new(name);
        let region: RegionRef<'ml, 'a> = unsafe { llfn.to_ref() }.region(0).unwrap();
        let types = llfn.get_op_operands_types();
        let block: mlir_ir::BlockRef<'ml, 'a> = region.append_block(melior::ir::Block::new(
            &types.iter().map(|t| (*t, Location::unknown(cx.mlir_ctx))).collect::<Vec<_>>(),
        ));
        block
        //llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, name.as_ptr())
    }

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock {
        debug!("append_sibling_block: {:?}", name);
        let sibling_block = melior::ir::Block::new(&[]);
        self.cur_block.parent_region().as_ref().unwrap().append_block(sibling_block)
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        self.cur_block = llbb;
    }

    fn ret_void(&mut self) {
        if self.is_unreachable() {
            return;
        }
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&[], self.cur_loc())
        } else {
            self.cx.cpu_return(&[], self.cur_loc())
        };
        self.append_op(op);
    }

    fn ret(&mut self, v: Self::Value) {
        if self.is_unreachable() {
            return;
        }
        let mut rets = vec![];
        let func_type = self.cur_block.parent_operation().unwrap().get_func_type().unwrap();
        if v.r#type().is_tuple() {
            let mut opt_rets = vec![];
            if let Err(err) = crate::mlir::poison::decode_ret_value(
                self.mlir_ctx,
                v,
                &mut opt_rets,
                self.cur_loc(),
            ) {
                self.emit_error(
                    format!("Failed to decode return value: {}", self.name),
                    self.cur_span,
                );
            }
            rets.extend(
                opt_rets
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.use_value_as_ty(v.unwrap(), func_type.result(i).unwrap())),
            );
        } else {
            // Force convert types into the destination type. Technically there shouldn't
            // be any type changes except for the memref which needs to be 'viewed' as
            // unranked
            rets.push(self.use_value_as_ty(v, func_type.result(0).unwrap()));
        }
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&rets, self.cur_loc())
        } else {
            self.cx.cpu_return(&rets, self.cur_loc())
        };
        self.append_op(op);
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        if self.is_unreachable() {
            return;
        }
        let op = melior::dialect::cf::br(
            &dest,
            &self.get_params()[0..dest.argument_count()],
            self.cur_loc(),
        );
        self.append_op(op);
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        if self.is_unreachable() {
            return;
        }
        let cond = self.use_value_as_ty(cond, self.type_i1());
        let op = melior::dialect::cf::cond_br(
            self.mlir_ctx,
            cond,
            &then_llbb,
            &else_llbb,
            &self.get_params()[0..then_llbb.argument_count()],
            &self.get_params()[0..else_llbb.argument_count()],
            self.cur_loc(),
        );
        self.append_op(op);
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    ) {
        use rustc_codegen_ssa_gpu::traits::AbiBuilderMethods;
        if self.is_unreachable() {
            return;
        }

        let v = self.intcast(v, self.type_i64(), false);
        // Build the default case
        let mut else_llbb_args = vec![];
        for i in 0..else_llbb.argument_count() {
            else_llbb_args.push(self.get_param(i));
        }

        // Now handle cases and their lables
        // Note that MLIR requires lables to be seperated from cases. It also
        // requires cases to be tuples with their arguments. We have to split
        // the cases and re-zip it into what we want.
        let mut case_values = vec![];
        let mut case_destinations = vec![];

        let mut dest_list: Vec<_> = vec![];
        let mut cases_args: Vec<_> = vec![];

        // Spliting the tuple since tuples can't be referenced individually
        for (on_val, dest) in cases {
            let mut case_args = vec![];

            dest_list.push(dest);

            case_values.push(on_val as i64);

            for i in 0..dest.argument_count() {
                case_args.push(self.get_param(i));
            }
            cases_args.push(case_args.clone());
        }

        for i in 0..dest_list.len() {
            // Build the cases
            case_destinations.push((dest_list[i].deref(), cases_args[i].as_slice()));
        }

        let op = melior::dialect::cf::switch(
            self.mlir_ctx,
            &case_values,
            v,
            v.r#type(),
            (&else_llbb, &else_llbb_args),
            &case_destinations,
            self.cur_loc(),
        )
        .expect("valid operation");
        self.append_op(op);
    }

    fn invoke(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs>,
        fn_abi: Option<&rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value {
        todo!()
    }

    fn unreachable(&mut self) {
        let zero = self.const_value(0, self.type_i1());
        self.assert(zero, "unreachable");
        let block = self.append_sibling_block("unreachable");
        self.br(block);
        let cur = self.cur_block;
        self.switch_to_block(block);
        self.br(self.cur_block);
        self.switch_to_block(cur);
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.mlir_overflowing_add(lhs, rhs, None).0
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::addf(lhs, rhs, self.cur_loc());
        self.append_fast_op_res(op)
    }

    // Do we even handle this?
    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    // Do we even handle this?
    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.mlir_overflowing_sub(lhs, rhs, None).0
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::subf(lhs, rhs, self.cur_loc());
        self.append_fast_op_res(op)
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        self.mlir_overflowing_mul(lhs, rhs, None).0
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::mulf(lhs, rhs, self.cur_loc());
        self.append_fast_op_res(op)
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let (lhs, rhs) = self.int_val_pair_cast(lhs, rhs);
        let op = melior::dialect::arith::divui(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // TODO: check division is exact, if not, panic
        let rem = self.urem(lhs, rhs);
        let zero = self.const_value(0, rem.r#type());
        let is_exact = self.icmp(IntPredicate::IntEQ, rem, zero);
        self.assert(is_exact, "exactudiv remainder is not zero");
        self.udiv(lhs, rhs)
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::divsi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::divf(lhs, rhs, self.cur_loc());
        self.append_fast_op_res(op)
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let lhs = self.use_value_as_integer(lhs, false);
        let rhs = self.use_value_as_integer(rhs, false);
        let op = melior::dialect::arith::remui(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let lhs = self.use_value_as_integer(lhs, true);
        let rhs = self.use_value_as_integer(rhs, true);
        let op = melior::dialect::arith::remsi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::remf(lhs, rhs, self.cur_loc());
        self.append_fast_op_res(op)
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let (rhs, lhs) = self.int_val_pair_cast(rhs, lhs);
        let op = melior::dialect::arith::shli(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::shrui(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let rhs_casted = self.intcast(rhs, lhs.r#type(), false);
        let op = melior::dialect::arith::shrsi(lhs, rhs_casted, self.cur_loc());
        self.append_op_res(op)
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let (lhs, rhs) = self.int_val_pair_cast(lhs, rhs);
        let op = melior::dialect::arith::andi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::ori(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::xori(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        let zero = self.const_value(0, v.r#type());
        self.sub(zero, v)
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::negf(v, self.cur_loc());
        self.append_fast_op_res(op)
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        // So the not here is actually bitwise not per rust documentation
        // There is no bitwise not in MLIR and therefore it's going to be an
        // xor to 0xFFFFFFFF which is -1
        let ty = v.r#type();
        let mask =
            self.const_value((1i128 << int_width(ty).expect("expected integer type")) - 1, ty);
        let op = melior::dialect::arith::xori(v, mask, self.cur_loc());
        self.append_op_res(op)
    }

    fn checked_binop(
        &mut self,
        oop: rustc_codegen_ssa_gpu::traits::OverflowOp,
        ty: rustc_middle::ty::Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        // The default add/sub/mul in GPU is wrapped add and so
        // we do not need to wrap it specially.
        let signed = match ty.kind() {
            rustc_middle::ty::Int(ity) => true,
            rustc_middle::ty::Uint(uty) => false,
            _ => panic!("non integer discriminant"),
        };
        match oop {
            OverflowOp::Add => self.mlir_overflowing_add(lhs, rhs, Some(signed)),
            OverflowOp::Sub => self.mlir_overflowing_sub(lhs, rhs, Some(signed)),
            OverflowOp::Mul => self.mlir_overflowing_mul(lhs, rhs, Some(signed)),
        }
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        trace!("from_immediate: {:?}", val);
        if val.r#type() == self.cx().type_i1() { self.zext(val, self.cx().type_i8()) } else { val }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: rustc_abi::Scalar) -> Self::Value {
        trace!("to_immediate_scalar: {:?} {:?}", val, scalar);
        val
    }

    fn alloca_san_dummy(&mut self) -> Self::Value {
        let val = self.alloca(rustc_abi::Size::from_bytes(8), rustc_abi::Align::EIGHT);
        self.san_dummy = Some(val);
        val
    }

    fn alloca_shared(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
        let ret_final_type = self.type_shared_memref(self.type_i8(), &[size.bytes() as i64], None);
        let name = self.define_static_shared_mem(size, align, self.cur_loc());
        let ptr = self.append_op_res(mlir_memref::get_global(
            self.mlir_ctx,
            &name,
            ret_final_type,
            self.cur_loc(),
        ));
        ptr
    }

    fn alloca(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
        let mut count = 1i64;
        let ty = if let Some(ty) = self.get_type_by_span(&self.cur_span) {
            ty
        } else {
            count = size.bytes() as i64;
            self.type_i8()
        };
        // The alloca function is used to allocate memory on the stack and thus must be default memory space.
        let mem_space = self.local_mem_space();
        let mem_ref_ty = self.type_memref(ty, &[count], None, Some(mem_space)).try_into().unwrap();
        let op = melior::dialect::memref::alloca(
            self.mlir_ctx,
            mem_ref_ty,
            &[],
            &[],
            Some(self.align_to_attr(align)),
            self.cur_loc(),
        );
        self.append_op_res(op)
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        self.load_with_check(ty, ptr, align, true)
    }

    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value {
        self.load_with_check(ty, ptr, rustc_abi::Align::ONE, false)
    }

    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: rustc_abi::Size,
    ) -> Self::Value {
        todo!()
    }

    fn load_operand(
        &mut self,
        place: rustc_codegen_ssa_gpu::mir::place::PlaceRef<'tcx, Self::Value>,
    ) -> rustc_codegen_ssa_gpu::mir::operand::OperandRef<'tcx, Self::Value> {
        assert!(!self.is_unreachable());
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        let llval = self.use_value(place.val.llval);

        let val = if place.val.llextra.is_some() {
            OperandValue::Ref(place.val)
        } else if self.cx.is_backend_immediate(place.layout) {
            let llval = self.load(self.mlir_type(place.layout, true), llval, place.val.align);
            OperandValue::Immediate(llval)
        } else if let BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a.primitive().size(self).align_to(b.primitive().align(self).abi);

            let mut load = |i, scalar: rustc_abi::Scalar, align| {
                let base_ty = self.scalar_pair_element_backend_type(place.layout, i, false);
                let ptr = llval;
                let offset = if i == 0 {
                    None
                } else {
                    Some(self.const_value(b_offset.bytes(), self.type_index()))
                };
                let ptr_memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
                let ptr = if i > 0 || base_ty != ptr_memref_ty.element() {
                    // TODO: Likely OK but I'm not sure... This thing feeds into the load
                    // eventually which, if ptr is already the base_ty, will bypass the
                    // mlir_memref_view in load
                    self.mlir_memref_view(
                        ptr,
                        self.type_memref(base_ty, &[1], None, ptr_memref_ty.memory_space()),
                        offset,
                        None,
                    )
                } else {
                    ptr
                };
                self.load(self.scalar_pair_element_backend_type(place.layout, i, false), ptr, align)
            };

            OperandValue::Pair(
                load(0, a, place.val.align),
                load(1, b, place.val.align.restrict_for_offset(b_offset)),
            )
        } else {
            OperandValue::Ref(place.val)
        };
        OperandRef { val, layout: place.layout }
    }

    fn write_operand_repeatedly(
        &mut self,
        elem: rustc_codegen_ssa_gpu::mir::operand::OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: rustc_codegen_ssa_gpu::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        for i in 0..count {
            let idx = self.const_value(i, self.type_index());
            let dest_elem = dest.project_index(self, idx);
            elem.val.store(self, dest_elem);
        }
    }

    fn range_metadata(&mut self, load: Self::Value, range: rustc_abi::WrappingRange) {
        todo!()
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        todo!()
    }

    #[cfg(all(feature = "inplace_bound_check", feature = "arith_immediate_bound_check"))]
    compile_error!(
        "Features `inplace_bound_check` and `arith_immediate_bound_check` are mutually exclusive"
    );

    #[cfg(feature = "trap_bound_check")]
    fn emit_bound_check(&mut self, idx: Self::Value, len: Self::Value, ptr: Self::Value) -> bool {
        // trap-based bound check is handled by assert op.
        if self.disable_bound_check {
            return true;
        }
        false
    }

    #[cfg(not(feature = "trap_bound_check"))]
    fn emit_bound_check(&mut self, idx: Self::Value, len: Self::Value, ptr: Self::Value) -> bool {
        if self.disable_bound_check {
            return true;
        }
        // Constant index can be evaluated at compile time
        let debug = self.tcx.sess.opts.debug_assertions;
        if debug {
            return false; // use default trap-based bound check
        }
        if let Some(v1) = crate::mlir::mlir_val_to_const_int(idx) {
            if let Some(v2) = crate::mlir::mlir_val_to_const_int(len) {
                if v1 < v2 {
                    return true;
                }
            }
        };
        let cmp = self.icmp(rustc_codegen_ssa_gpu::common::IntPredicate::IntULT, idx, len);
        self.assert_before_index(cmp, idx);
        true
    }

    fn store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
    ) -> Self::Value {
        self.store_with_check(val, ptr, align, true)
    }

    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) -> Self::Value {
        if self.is_unreachable() {
            return val;
        }
        self.store(val, ptr, align)
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: rustc_abi::Size,
    ) {
        todo!()
    }

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        let ptr = self.use_value(ptr);
        let idx = self.use_value(indices[0]);
        let addr = self.ptrtoint(ptr, self.type_i64());
        let size = self.static_size_of(ty);
        let size = self.const_value(size as i64, self.type_i64());
        let offset = self.mul(idx, size);
        let addr_with_offset = self.add(addr, offset);
        let ptr_ty = ptr.r#type();
        let mem_ref_ty = MemRefType::try_from(ptr_ty).unwrap();
        let ptr = self.inttoptr(
            addr_with_offset,
            self.type_memref(ty, &[1], None, mem_ref_ty.memory_space()),
        );
        ptr
    }

    // indices are unsigned
    fn inbounds_nuw_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        self.inbounds_gep_ret(ty, ptr, indices, true)
    }

    // indices are signed
    fn inbounds_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        let index = indices[0];
        if let Some(const_index) = crate::mlir::mlir_val_to_const_int(index) {
            if (const_index as i64) < 0 {
                return self.gep(ty, ptr, indices);
            }
        }
        self.inbounds_gep_ret(ty, ptr, indices, true)
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let src_ty = val.r#type();
        let val = if src_ty.is_index() { self.intcast(val, self.type_i128(), false) } else { val };
        let src_ty = val.r#type();
        if dest_ty.is_integer() && src_ty.is_integer() {
            self.append_op_res(melior::dialect::arith::trunci(val, dest_ty, self.cur_loc()))
        } else if dest_ty.is_float() && src_ty.is_float() {
            self.append_op_res(
                melior::dialect::ods::arith::truncf(self.mlir_ctx, dest_ty, val, self.cur_loc())
                    .into(),
            )
        } else {
            self.emit_error(format!("Unsupported trunc: {} to {}", src_ty, dest_ty), self.cur_span);
        }
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let op = melior::dialect::arith::extsi(val, dest_ty, self.cur_loc());
        self.append_op_res(op)
    }

    fn fptoui_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptosi_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.use_value_as_integer(val, false);
        self.append_op_res(melior::dialect::arith::uitofp(val, dest_ty, self.cur_loc()))
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.use_value_as_integer(val, true);
        self.append_op_res(melior::dialect::arith::sitofp(val, dest_ty, self.cur_loc()))
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let op = melior::dialect::arith::extf(val, dest_ty, self.cur_loc());
        self.append_op_res(op)
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        if val == self.const_null(val.r#type()) {
            return self.const_value(0, dest_ty);
        }
        let results = self.extract_strided_metadata(val);
        let base_ptr = self.append_op_res(
            melior::dialect::ods::memref::extract_aligned_pointer_as_index(
                self.mlir_ctx,
                self.type_index(),
                results.base_memref,
                self.cur_loc(),
            )
            .into(),
        );
        let base = self.intcast(base_ptr, dest_ty, false);
        let offset = self.intcast(results.byte_offset, dest_ty, false);
        self.add(base, offset)
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let align = rustc_abi::Align::from_bytes(8).unwrap();
        let base_memref = self.alloca(rustc_abi::Size::from_bytes(64), align);
        let zero = self.const_value(0, self.type_index());
        let one = self.const_value(1, self.type_index());
        let two = self.const_value(2, self.type_index());
        let ptr = self.inbounds_gep_ret(self.type_i64(), base_memref, &[zero], false);
        let val = self.intcast(val, self.type_i64(), false);
        self.store_with_check(val, ptr, align, false);
        let ptr = self.inbounds_gep_ret(self.type_i64(), base_memref, &[one], false);
        self.store_with_check(val, ptr, align, false);
        let ptr = self.inbounds_gep_ret(self.type_i64(), base_memref, &[two], false);
        let zero_u64 = self.const_value(0, self.type_i64());
        self.store_with_check(zero_u64, ptr, align, false);
        if self.fn_shared_memory_size.read().unwrap()[&self.name] > 0 {
            panic!(
                "inttoptr: {:?} {} -> {} where memoryspace is unknown",
                self.cur_span,
                val.r#type(),
                dest_ty
            );
        }
        let base_memref_ty = MemRefType::try_from(base_memref.r#type()).unwrap();
        let casted_base_memref = self.mlir_memref_view(
            base_memref,
            // dest_ty is memref type but it is safe to do so here.
            unsafe {
                self._type_memref(
                    dest_ty,
                    &[1],
                    Some(base_memref_ty.layout()),
                    base_memref_ty.memory_space(),
                )
                .into()
            },
            None,
            None,
        );
        self.mlir_load(dest_ty, casted_base_memref, &[zero], align)
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.use_value(val);
        if val.r#type() == dest_ty {
            val
        } else {
            self.append_op_res(melior::dialect::arith::bitcast(val, dest_ty, self.cur_loc()))
        }
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        assert!(!self.is_unreachable());
        let src_ty = val.r#type();
        if src_ty == dest_ty {
            return val;
        }
        if src_ty.is_mem_ref() {
            assert!(!is_signed);
            return self.ptrtoint(val, dest_ty);
        }
        if !src_ty.is_integer() && !src_ty.is_index() {
            panic!();
        }
        if let Some(const_val) = crate::mlir::mlir_val_to_const_int(val) {
            let bit_width = int_width(dest_ty).unwrap();
            let val = const_val & ((1 << bit_width) - 1);
            match bit_width {
                1 => return self.const_value(val != 0, dest_ty),
                8 => return self.const_value(val as u8 as i8, dest_ty),
                16 => return self.const_value(val as u16 as i16, dest_ty),
                32 => return self.const_value(val as u32 as i32, dest_ty),
                64 => return self.const_value(val as u64 as i64, dest_ty),
                128 => return self.const_value(val as u128 as i128, dest_ty),
                _ => panic!("Unsupported intcast to width {}", bit_width),
            }
        }
        if dest_ty.is_mem_ref() {
            return self.inttoptr(val, dest_ty);
        };
        let op = if src_ty.is_index() || dest_ty.is_index() {
            // If either is index, we need to use index_cast
            melior::dialect::arith::index_cast(val, dest_ty, self.cur_loc())
        } else if src_ty.is_integer() || dest_ty.is_integer() {
            // Integer. Are we doing an extension or trucation?
            let src_int_ty = melior::ir::r#type::IntegerType::try_from(src_ty).unwrap();
            let dst_int_ty = melior::ir::r#type::IntegerType::try_from(dest_ty)
                .unwrap_or_else(|_| panic!("expected integer type {} {}", val, dest_ty));
            if src_int_ty.width() > dst_int_ty.width() {
                melior::dialect::arith::trunci(val, dest_ty, self.cur_loc())
            } else if !is_signed {
                melior::dialect::arith::extui(val, dest_ty, self.cur_loc())
            } else {
                melior::dialect::arith::extsi(val, dest_ty, self.cur_loc())
            }
        } else {
            panic!("Unsupported intcast: {} to {}", src_ty, dest_ty);
        };

        self.append_op_res(op)
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        val
    }

    fn icmp(
        &mut self,
        op: rustc_codegen_ssa_gpu::common::IntPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        if self.is_unreachable() {
            return lhs;
        }
        let predicate = match op {
            rustc_codegen_ssa_gpu::common::IntPredicate::IntEQ => {
                melior::dialect::arith::CmpiPredicate::Eq
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntNE => {
                melior::dialect::arith::CmpiPredicate::Ne
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntUGT => {
                melior::dialect::arith::CmpiPredicate::Ugt
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntUGE => {
                melior::dialect::arith::CmpiPredicate::Uge
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntULT => {
                melior::dialect::arith::CmpiPredicate::Ult
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntULE => {
                melior::dialect::arith::CmpiPredicate::Ule
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntSGT => {
                melior::dialect::arith::CmpiPredicate::Sgt
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntSGE => {
                melior::dialect::arith::CmpiPredicate::Sge
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntSLT => {
                melior::dialect::arith::CmpiPredicate::Slt
            }
            rustc_codegen_ssa_gpu::common::IntPredicate::IntSLE => {
                melior::dialect::arith::CmpiPredicate::Sle
            }
        };

        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        let mut normalized_value = |ty: Self::Type, val| {
            let val = self.use_value(val);
            if ty.is_mem_ref() {
                self.ptrtoint(val, self.type_index())
            } else if ty.is_index() || ty.is_integer() {
                val
            } else {
                self.emit_error(format!("Unsupported type for icmp: {}", ty), self.cur_span)
            }
        };
        let lhs = normalized_value(lhs_ty, lhs);
        let mut rhs = normalized_value(rhs_ty, rhs);

        let lhs_ty = lhs.r#type();
        let rhs_ty = rhs.r#type();

        if rhs_ty != lhs_ty {
            rhs = self.intcast(rhs, lhs_ty, false);
        }

        self.append_op_res(melior::dialect::arith::cmpi(
            self.mlir_ctx,
            predicate,
            lhs,
            rhs,
            self.cur_loc(),
        ))
    }

    fn fcmp(
        &mut self,
        op: rustc_codegen_ssa_gpu::common::RealPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        if self.is_unreachable() {
            return lhs;
        }
        let predicate = match op {
            rustc_codegen_ssa_gpu::common::RealPredicate::RealPredicateFalse => {
                melior::dialect::arith::CmpfPredicate::False
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealOEQ => {
                melior::dialect::arith::CmpfPredicate::Oeq
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealOGT => {
                melior::dialect::arith::CmpfPredicate::Ogt
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealOGE => {
                melior::dialect::arith::CmpfPredicate::Oge
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealOLT => {
                melior::dialect::arith::CmpfPredicate::Olt
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealOLE => {
                melior::dialect::arith::CmpfPredicate::Ole
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealONE => {
                melior::dialect::arith::CmpfPredicate::One
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealORD => {
                melior::dialect::arith::CmpfPredicate::Ord
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealUNO => {
                melior::dialect::arith::CmpfPredicate::Uno
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealUEQ => {
                melior::dialect::arith::CmpfPredicate::Ueq
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealUGT => {
                melior::dialect::arith::CmpfPredicate::Ugt
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealUGE => {
                melior::dialect::arith::CmpfPredicate::Uge
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealULT => {
                melior::dialect::arith::CmpfPredicate::Ult
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealULE => {
                melior::dialect::arith::CmpfPredicate::Ule
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealUNE => {
                melior::dialect::arith::CmpfPredicate::Une
            }
            rustc_codegen_ssa_gpu::common::RealPredicate::RealPredicateTrue => {
                melior::dialect::arith::CmpfPredicate::True
            }
        };
        let op = melior::dialect::arith::cmpf(
            self.mlir_ctx,
            predicate,
            self.use_value(lhs),
            self.use_value(rhs),
            self.cur_loc(),
        );
        self.append_fast_op_res(op)
    }

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: rustc_codegen_ssa_gpu::MemFlags,
        tt: Option<rustc_ast::expand::typetree::FncTree>,
    ) {
        self.vector_memcpy(dst, dst_align, src, src_align, size, flags);
    }

    fn memmove(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) {
        todo!()
    }

    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: rustc_abi::Align,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) {
        let is_volatile = flags.contains(rustc_codegen_ssa_gpu::MemFlags::VOLATILE);
        let is_volatile = IntegerAttribute::new(self.type_i1(), if is_volatile { 1 } else { 0 });
        let dst = self.ptrtollvmptr(ptr);
        let fill_byte = self.use_value(fill_byte);
        let size = self.use_value(size);
        self.append_op(
            melior::dialect::ods::llvm::intr_memset(
                self.mlir_ctx,
                dst,
                fill_byte,
                size,
                is_volatile,
                self.cur_loc(),
            )
            .into(),
        );
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        if self.is_unreachable() {
            return cond;
        }
        let cond = self.use_value_as_ty(cond, self.type_i1());
        let then_val = self.use_value(then_val);
        let else_val = self.use_value(else_val);
        self.append_op_res(melior::dialect::arith::select(cond, then_val, else_val, self.cur_loc()))
    }

    fn va_arg(&mut self, list: Self::Value, ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value {
        todo!()
    }

    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value {
        todo!()
    }

    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value {
        if self.is_unreachable() {
            return agg_val;
        }
        if agg_val.r#type().is_ranked_tensor() {
            let op = melior::dialect::ods::tensor::extract(
                self.mlir_ctx,
                self.element_type(agg_val.r#type()),
                agg_val,
                &[self.const_value(0, self.type_index())],
                self.cur_loc(),
            );
            self.append_op_res(op.into())
        } else if agg_val.r#type().is_vector() {
            let op = melior::dialect::ods::vector::extract(
                self.mlir_ctx,
                agg_val,
                &[],
                DenseI64ArrayAttribute::new(self.mlir_ctx, &[idx as i64]).into(),
                self.cur_loc(),
            );
            self.append_op_res(op.into())
        } else if agg_val.r#type().is_tuple() {
            todo!()
        } else {
            let Ok(op_val) = mlir_ir::operation::OperationResult::<'ml, 'a>::try_from(agg_val)
            else {
                panic!("agg_val is not an operation result {}", agg_val);
            };
            if idx == 0 {
                return agg_val;
            }
            *self.op_to_extra_values[&op_val.owner().location().to_string()]
                .get(idx as usize)
                .unwrap_or_else(|| {
                    panic!("Index out of bounds in extract_value {} for {}", idx, agg_val)
                })
        }
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        self.append_op_res(crate::mlir::poison::insert_value(
            self.mlir_ctx,
            agg_val,
            elt,
            idx,
            self.cur_loc(),
        ))
    }

    fn set_personality_fn(&mut self, personality: Self::Function) {
        todo!()
    }

    fn cleanup_landing_pad(&mut self, pers_fn: Self::Function) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn filter_landing_pad(&mut self, pers_fn: Self::Function) -> () {
        todo!()
    }

    fn resume(&mut self, exn0: Self::Value, exn1: Self::Value) {
        todo!()
    }

    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet {
        todo!()
    }

    fn cleanup_ret(&mut self, funclet: &Self::Funclet, unwind: Option<Self::BasicBlock>) {
        todo!()
    }

    fn catch_pad(&mut self, parent: Self::Value, args: &[Self::Value]) -> Self::Funclet {
        todo!()
    }

    fn catch_switch(
        &mut self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        handlers: &[Self::BasicBlock],
    ) -> Self::Value {
        todo!()
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn atomic_rmw(
        &mut self,
        op: rustc_codegen_ssa_gpu::common::AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        ret_ptr: bool,
    ) -> Self::Value {
        self.emit_error(
            "GPU has different intrinsics. Please use gpu::sync::atomic_xxx".into(),
            self.cur_span,
        );
    }

    fn atomic_fence(
        &mut self,
        order: rustc_middle::ty::AtomicOrdering,
        scope: rustc_codegen_ssa_gpu::common::SynchronizationScope,
    ) {
        self.emit_error(
            "GPU has different intrinsics. Use gpu::sync::sync_threads to sync in a block.".into(),
            self.cur_span,
        );
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        todo!()
    }

    // There are lifetime dialects for LLVM IR. GCC decides to ignore these. We can
    // skip them for now and may (or may not) add them back later.
    fn lifetime_start(&mut self, ptr: Self::Value, size: rustc_abi::Size) {
        //todo!()
        // Don't care!
    }

    fn lifetime_end(&mut self, ptr: Self::Value, size: rustc_abi::Size) {
        //todo!()
        // Don't care!
    }

    fn call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs>,
        fn_abi: Option<&rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value {
        if self.is_unreachable() {
            return llfn;
        }
        let args = args.iter().map(|arg| self.use_value(*arg)).collect::<Vec<_>>();
        let args = &args;
        let fn_sym_ptr = llfn.to_func_sym().unwrap();
        let sym = fn_sym_ptr.value();
        debug!("fn_sym_ptr = {}", fn_sym_ptr);
        let ftype = Some(self.fn_db.read().unwrap()[sym].op.get_func_type().unwrap());
        let span = self.cur_span;
        let op = self.call_op(llfn, instance, args, ftype, span).unwrap();
        let loc = self.cur_loc().to_string();

        if let Some(op) = op {
            if op.result_count() > 0 {
                let mut ret_vec = vec![];
                for i in 0..op.result_count() {
                    let ret: mlir_ir::Value<'ml, 'val> = op.result(i).unwrap().into();
                    ret_vec.push(ret);
                }
                self.op_to_extra_values.insert(loc.clone(), ret_vec);
            }
        }

        *self.op_to_extra_values.get(&loc).unwrap_or(&vec![llfn]).first().unwrap()
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let op = melior::dialect::arith::extui(val, dest_ty, self.cur_loc());
        self.append_op_res(op)
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value) {
        todo!()
    }

    fn tail_call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs>,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, rustc_middle::ty::Ty<'tcx>>,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) {
        unimplemented!();
    }
}
