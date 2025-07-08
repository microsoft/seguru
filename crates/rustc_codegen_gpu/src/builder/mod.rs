mod abi;
mod coverage;
mod debug;
mod intrinsic;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;

use melior::dialect::memref as mlir_memref;
use melior::helpers::BuiltinBlockExt;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{self as mlir_type, FunctionType, MemRefType};
use melior::ir::{
    self as mlir_ir, BlockLike, Location, RegionLike, RegionRef, ShapedTypeLike, TypeLike, Value,
    ValueLike,
};
use rustc_abi::BackendRepr;
use rustc_codegen_ssa_gpu::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa_gpu::traits::{
    AsmBuilderMethods, BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, OverflowOp, StaticBuilderMethods,
};
use rustc_span::Span;
use tracing::{debug, trace, warn};

use crate::attr::GpuItem;
use crate::context::GPUCodegenContext;
use crate::mlir::memref::{extract_strided_metadata, extract_strided_metadata_results};
use crate::mlir::{BUILTIN_SYM, BlockRefWithTime, MLIROpHelpers, ValueToOpRef};

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
}

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub fn cur_loc(&self) -> Location<'ml> {
        self.cx.to_mlir_loc(self.cur_span)
    }

    pub fn cur_block(&self) -> &'a mlir_ir::Block<'ml> {
        unsafe { self.cur_block.to_ref() }
    }

    pub fn const_value(
        &self,
        val: impl std::fmt::Display,
        ty: mlir_ir::Type<'ml>,
    ) -> mlir_ir::Value<'ml, 'a> {
        self.mlir_const_val_from_type(val, ty, self.cur_block())
    }

    fn static_size_of(&self, ty: mlir_ir::Type<'_>) -> usize {
        if ty.is_integer() {
            self.cx.mlir_integer_width(ty) / 8
        } else if ty.is_index() || ty.is_mem_ref() {
            size_of::<usize>()
        } else if ty.is_float() {
            self.mlir_float_width(ty) / 8
        } else if ty.is_tuple() {
            let tuple = mlir_type::TupleType::try_from(ty).unwrap();
            let mut total_size = 0;
            for i in 0..tuple.type_count() {
                total_size += self.static_size_of(tuple.r#type(i).unwrap());
            }
            total_size
        } else if ty.is_ranked_tensor() {
            let ranked_ty = mlir_type::RankedTensorType::try_from(ty).unwrap();
            let ty = self.mlir_element_type(ty);
            let base_size = self.static_size_of(ty);
            todo!();
        } else {
            panic!("Unsupported type: {:?}", ty);
        }
    }

    fn call_op(
        &mut self,
        fn_ptr_value: melior::ir::Value<'ml, 'a>,
        instance: Option<rustc_middle::ty::Instance<'tcx>>,
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
            trace!("call_op fn_sym_ptr: {:?}", builtin_sym.value());
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
                panic!();
            }
        }
        let mut next_itype = input_types.iter();
        let args = &args
            .iter()
            .map(|v| self.use_value_as_ty(*v, *(next_itype.next().unwrap())))
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

    fn build_sfi(
        &mut self,
        ptr: melior::ir::Value<'ml, 'a>,
        ptr_size: melior::ir::Value<'ml, 'a>,
        offset: melior::ir::Value<'ml, 'a>,
    ) {
        let one = self.mlir_const_val_from_type(1, self.type_i64(), self.cur_block());
        let index_int = self.sub(offset, one);
        let oob = self.sub(index_int, ptr_size);
        let shift = self.emit_constant(63, self.type_i64());
        let oob_flag = self.ashr(oob, shift);

        self.emit_llvm_volatile_and_load(oob_flag, self.san_dummy.unwrap());
    }

    fn call_gpu_builtin_operation(
        &mut self,
        gpu_item: GpuItem,
        _fn_ptr_value: melior::ir::Value<'ml, 'a>,
        instance: Option<rustc_middle::ty::Instance<'tcx>>,
        args: &[melior::ir::Value<'ml, 'a>],
        return_types: &[melior::ir::Type<'ml>],
        span: Span,
    ) -> Result<Option<melior::ir::OperationRef<'ml, 'a>>, melior::Error> {
        let loc = self.to_mlir_loc(span);
        let get_generic_type = || {
            let mut generic_types = vec![];
            if let Some(instance) = instance {
                for arg in instance.args.iter() {
                    if let rustc_type_ir::GenericArgKind::Type(ty) = arg.unpack() {
                        generic_types.push(ty);
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
            GpuItem::ThreadId => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                assert!(return_types.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                let op_ref =
                    self.append_op(crate::mlir::gpu::thread_id(self.mlir_ctx, dimention, loc));
                Ok(Some(op_ref))
            }
            GpuItem::GlobalThreadId => {
                let dimention = self.extra_state.attrs.pop().unwrap();
                let op_ref =
                    self.append_op(crate::mlir::gpu::global_id(self.mlir_ctx, dimention, loc));
                Ok(Some(op_ref))
            }
            GpuItem::BlockDim => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                let op_ref =
                    self.append_op(crate::mlir::gpu::block_dim(self.mlir_ctx, dimention, loc));
                Ok(Some(op_ref))
            }
            GpuItem::GridDim => {
                // attrs must be parsed from #gpu<dim(x)>
                assert!(self.extra_state.attrs.len() == 1);
                let dimention = self.extra_state.attrs.pop().unwrap();
                let op_ref =
                    self.append_op(crate::mlir::gpu::grid_dim(self.mlir_ctx, dimention, loc));
                Ok(Some(op_ref))
            }
            GpuItem::PrintArgs => {
                // printf function should starts with a format passed by add_mlir_string_attr
                // args can be passed to printf as a list of values.
                // printf ends with an empty printf
                assert!(self.extra_state.attrs.len() == 1);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.extra_state.args.entry(gpu_item)
                {
                    e.insert(args.to_vec());
                } else {
                    self.extra_state.args.get_mut(&gpu_item).unwrap().extend(args);
                }
                Ok(None)
            }
            GpuItem::PrintFormat => {
                let Ok(format) = self.extra_state.attrs.pop().unwrap().try_into() else {
                    let err =
                        format!("{:?} must take a single StringAttribute as format", gpu_item);
                    self.emit_error(err.clone(), span);
                };
                let op_ref = self.append_op(
                    melior::dialect::ods::gpu::printf(self.mlir_ctx, args, format, loc).into(),
                );
                Ok(Some(op_ref))
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
                warn!("gpu.scope args: {:?} {} {}", args, fn_type, fn_type.input_count());
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
                warn!("gpu.iter_next args: {:?}", args);
                let arg = args[0];
                let ptr = self.load(
                    self.type_memref(self.type_i8(), &[1], None),
                    arg,
                    rustc_abi::Align::EIGHT,
                );
                let size_ptr = self.inbounds_gep(
                    self.type_index(),
                    arg,
                    &[self.const_value(1, self.type_index())],
                );
                let size = self.load(self.type_index(), size_ptr, rustc_abi::Align::EIGHT);
                let window_ptr = self.inbounds_gep(
                    self.type_i64(),
                    arg,
                    &[self.const_value(2, self.type_index())],
                );
                let window = self.load(self.type_i64(), window_ptr, rustc_abi::Align::EIGHT);

                let index_ptr = self.inbounds_gep(
                    self.type_index(),
                    arg,
                    &[self.const_value(3, self.type_index())],
                );
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
                // args[0]: original:      memref<1xi8>
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
                self.build_sfi(original, original_size, index_int_upper);

                // 2. Build the subslice: Done by transforming memref<1xi8> into the
                //    strided form using subview. Shortcut to use inbounds_gep
                let indices = vec![offset; 1];
                let subview_op = self.inbounds_gep_op(self.type_i8(), original, &indices);
                let op_ref = self.append_op(subview_op);

                // 3. Build the 'pair' of memref and i64
                let res = op_ref.result(0).unwrap().into();
                self.op_to_extra_values.insert(op_ref.location().to_string(), vec![res, window]);

                Ok(None)
            }
            GpuItem::NewSharedMem => {
                // Do not init the content of the shared memory.
                Ok(None)
            }
            GpuItem::AtomicAdd => {
                trace!("gpu.atomic_add args: {:?}", args);
                // args[0]: ptr:      memref<1xi8>
                // args[1]: val:      value, can be any thing... (f32, i32, ...)

                let ptr = args[0];
                let val = args[1];
                let offset = self.emit_constant(0, self.type_index());

                let indices_vec = vec![offset];
                let indices = &indices_vec;
                let kind = if val.r#type().is_integer() || val.r#type().is_index() {
                    mlir_ir::attribute::IntegerAttribute::new(self.type_i64(), 1).into()
                } else {
                    mlir_ir::attribute::IntegerAttribute::new(self.type_i64(), 0).into()
                };

                // Translate ptr into the correct form
                let ptr_memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
                let ptr_t = if self.mlir_element_type(ptr.r#type()) != val.r#type() {
                    let target_memref_ty =
                        MemRefType::new(val.r#type(), &[1], None, ptr_memref_ty.memory_space());
                    self.mlir_memref_view(ptr, target_memref_ty.into(), None)
                } else {
                    ptr
                };

                let atomic_rmw_op = melior::dialect::ods::memref::atomic_rmw(
                    self.mlir_ctx,
                    val.r#type(),
                    val,
                    ptr_t,
                    indices,
                    kind,
                    self.cur_loc(),
                );

                self.append_op(atomic_rmw_op.into());

                Ok(None)
            }
        }
    }

    fn inbounds_gep_op(
        &mut self,
        ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
        ptr: melior::ir::Value<'ml, 'a>,
        indices: &[melior::ir::Value<'ml, 'a>],
    ) -> mlir_ir::Operation<'ml> {
        if indices.len() != 1 {
            panic!("only supports single index");
        }
        let src_ty = ptr.r#type();
        let src_memref_ty = MemRefType::try_from(src_ty).unwrap();
        let base_ty = src_memref_ty.element();
        let indices = indices
            .iter()
            .map(|v| {
                if v.r#type().is_integer() || v.r#type().is_index() {
                    let v = self.use_value(*v);
                    self.intcast(v, self.type_index(), false).into()
                } else {
                    panic!("Must be int or index type");
                }
            })
            .collect::<Vec<_>>();
        let ptr = if ty != base_ty {
            let size = self.static_size_of(ty);
            let base_size = self.static_size_of(base_ty);
            if size % base_size != 0 {
                warn!(
                    "inbounds_gep_op: size {} is not a multiple of base size {} for type {:?}",
                    size, base_size, ty
                );
            }
            assert!(size % base_size == 0);
            let target_memref_ty = self.type_memref(ty, &[1], src_memref_ty.memory_space());
            self.mlir_memref_view(ptr, target_memref_ty, None)
        } else {
            ptr
        };
        crate::mlir::memref::subview(
            self.mlir_ctx,
            ty,
            ptr,
            &indices,
            &[1.into()],
            &[1.into()],
            self.cur_loc(),
        )
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
            let op = self.append_op((op).clone());
            op.result(0).unwrap().into()
        } else if let Ok(op) = val.is_from_op(Some("memref.get_global")) {
            let op = self.append_op((op).clone());
            op.result(0).unwrap().into()
        } else {
            val
        }
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
            self.mlir_memref_view(val, dst_ty, None)
        } else if ty.is_index() || dst_ty.is_index() {
            assert!(ty.is_integer() || dst_ty.is_integer());
            self.intcast(val, dst_ty, false)
        } else {
            panic!("Cannot use value {:?} as type {:?}", val, dst_ty);
        }
    }

    pub fn mlir_memref_view(
        &mut self,
        val: melior::ir::Value<'ml, 'a>,
        dst_ty: melior::ir::Type<'ml>,
        byte_offset: Option<Value<'ml, 'a>>,
    ) -> Value<'ml, 'a> {
        let ty = val.r#type();
        assert!(ty.is_mem_ref());
        assert!(dst_ty.is_mem_ref());
        if dst_ty == ty {
            return val;
        }
        let dst_memref_ty: MemRefType<'ml> = dst_ty.try_into().expect("expected memref type");
        let memref_ty: MemRefType<'ml> = ty.try_into().expect("expected memref type");
        assert!(memref_ty.memory_space() == dst_memref_ty.memory_space());
        let layout = crate::mlir::attr::StridedLayoutAttribute::try_from(memref_ty.layout());
        let base_memref = val;
        let (base_memref, base_byte_offset) = match layout {
            Ok(layout) if layout.get_offset() != 0 => {
                // view cannot work on offset !=0;
                warn!("mlir_memref_view with offset != 0 casting {} to {}", val, dst_ty);
                let op = extract_strided_metadata(self.cx.mlir_ctx, val, self.cur_loc());
                let op = self.append_op(op.into());
                let results = extract_strided_metadata_results(memref_ty, op);
                let element_ty = self.mlir_element_type(ty);
                assert!(element_ty == self.type_i8());

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
        let op = melior::dialect::memref::view(
            self.cx.mlir_ctx,
            base_memref,
            byte_offset,
            &[], // No dynamic sizes
            dst_ty.try_into().unwrap(),
            self.cur_loc(),
        );
        self.append_op_res(op)
    }

    fn append_op_res(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        trace!("append_op_res: {:?}", op);
        self.cur_block().append_op_result(op).unwrap()
    }

    fn is_unreachable(&self) -> bool {
        self.cur_block().terminator().is_some()
    }

    fn append_op(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
        trace!("append_op: {:?}", op);
        if self.is_unreachable() {
            panic!("Cannot append operation to unreachable block");
        }
        self.cur_block().append_operation(op)
    }

    fn inttollvmptr(
        &mut self,
        val: mlir_ir::Value<'ml, 'a>,
        dest_ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
    ) -> mlir_ir::Value<'ml, 'a> {
        assert!(!self.is_unreachable());

        let int64_val = if val.r#type() == self.type_i64() {
            val
        } else {
            self.intcast(val, self.type_i64(), false)
        };
        let op =
            melior::dialect::ods::llvm::inttoptr(self.mlir_ctx, dest_ty, int64_val, self.cur_loc())
                .into();
        self.append_op_res(op)
    }

    #[allow(dead_code)]
    pub fn inside_gpu_mod(&self) -> bool {
        if let Some(op) = self.cur_block().parent_operation() { op.is_gpu_func() } else { false }
    }

    pub fn inside_kernel_func(&self) -> bool {
        if let Some(op) = self.cur_block().parent_operation() { op.is_kernel_func() } else { false }
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
        for i in 0..self.cur_block().argument_count() {
            let val = self.cur_block().argument(i).unwrap().into();
            ret.push(val);
        }
        ret
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

impl<'tcx: 'a, 'ml: 'a, 'a> AsmBuilderMethods<'tcx> for GpuBuilder<'tcx, 'ml, 'a> {
    fn codegen_inline_asm(
        &mut self,
        template: &[rustc_ast::InlineAsmTemplatePiece],
        operands: &[rustc_codegen_ssa_gpu::traits::InlineAsmOperandRef<'tcx, Self>],
        options: rustc_ast::InlineAsmOptions,
        line_spans: &[rustc_span::Span],
        instance: rustc_middle::ty::Instance<'_>,
        dest: Option<Self::BasicBlock>,
        catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        todo!()
    }
}

impl<'tcx, 'ml, 'a> Deref for GpuBuilder<'tcx, 'ml, 'a> {
    fn deref(&self) -> &Self::Target {
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
        Self {
            cx,
            name: sym.value().to_string(),
            cur_block: llbb,
            cur_span: rustc_span::DUMMY_SP,
            extra_state: GpuBuilderState::new(),
            dummy: PhantomData,
            span_to_type: HashMap::new(),
            op_to_extra_values: HashMap::new(),
            san_dummy: None,
        }
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
        self.cur_block().parent_region().as_ref().unwrap().append_block(sibling_block)
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
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&[self.use_value(v)], self.cur_loc())
        } else {
            self.cx.cpu_return(&[self.use_value(v)], self.cur_loc())
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
        instance: Option<rustc_middle::ty::Instance<'tcx>>,
    ) -> Self::Value {
        todo!()
    }

    fn unreachable(&mut self) {
        if self.is_unreachable() {
            return;
        }
        let block = self.append_sibling_block("unreachable");
        self.br(block);
        let cur = self.cur_block;
        self.switch_to_block(block);
        let op = melior::dialect::cf::assert(
            self.mlir_ctx,
            self.const_value(0, self.type_i1()),
            "unreachable",
            self.cur_loc(),
        );
        self.append_op(op);
        self.br(self.cur_block);
        self.switch_to_block(cur);
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let lhs = self.use_value(lhs);
        let rhs = self.use_value(rhs);
        let rhs = self.intcast(rhs, lhs.r#type(), false);
        let op = melior::dialect::arith::addi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::addf(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
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
        // TODO: Currently casting rhs to lhs. A better way is to see who's longer...
        let rhs_casted = self.intcast(rhs, lhs.r#type(), false);
        let op = melior::dialect::arith::subi(lhs, rhs_casted, self.cur_loc());
        self.append_op_res(op)
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::subf(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // TODO: Currently casting rhs to lhs. A better way is to see who's longer...
        let lhs = self.use_value(lhs);
        let rhs = self.use_value(rhs);
        let rhs_casted = self.intcast(rhs, lhs.r#type(), false);
        let op = melior::dialect::arith::muli(lhs, rhs_casted, self.cur_loc());
        self.append_op_res(op)
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::mulf(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::divui(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
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
        self.append_op_res(op)
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::remui(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::remsi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::remf(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
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
        let rhs_casted = self.intcast(rhs, lhs.r#type(), false);
        let op = melior::dialect::arith::andi(lhs, rhs_casted, self.cur_loc());
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
        let op =
            melior::dialect::arith::subi(self.const_value(0, self.type_index()), v, self.cur_loc());
        self.append_op_res(op)
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        let op = melior::dialect::arith::negf(v, self.cur_loc());
        self.append_op_res(op)
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        // So the not here is actually bitwise not per rust documentation
        // There is no bitwise not in MLIR and therefore it's going to be an
        // xor to 0xFFFFFFFF which is -1

        let op = melior::dialect::arith::xori(
            v,
            self.const_value(-1, self.type_index()),
            self.cur_loc(),
        );
        self.append_op_res(op)
    }

    fn checked_binop(
        &mut self,
        oop: rustc_codegen_ssa_gpu::traits::OverflowOp,
        ty: rustc_middle::ty::Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        warn!("Incomplete checked_binop: {:?} {:?} {:?}", oop, lhs, rhs);
        // TODO: Build op with set_overflow_flags
        let ret = match oop {
            OverflowOp::Add => self.add(lhs, rhs),
            OverflowOp::Sub => self.sub(lhs, rhs),
            OverflowOp::Mul => self.mul(lhs, rhs),
        };
        (ret, self.const_value(0, self.type_i1()))
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
        let ret_type =
            self.type_shared_memref(self.type_i8(), &[crate::mlir::memref::dynamic_size()]);
        let ret_final_type = self.type_shared_memref(self.type_i8(), &[size.bytes() as i64]);
        let used_size = self.fn_shared_memory_size.read().unwrap()[&self.name];
        if used_size > 0 {
            self.emit_error(
                "Only support a single shared var allocation in current version".into(),
                self.cur_span,
            );
        }
        *self.fn_shared_memory_size.write().unwrap().get_mut(&self.name).unwrap() +=
            size.bytes() as usize;
        let ptr = self.append_op_res(
            melior::dialect::ods::gpu::dynamic_shared_memory(
                self.mlir_ctx,
                ret_type,
                self.cur_loc(),
            )
            .into(),
        );
        let ptr = self.mlir_memref_view(ptr, ret_final_type, None);
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
        let mem_ref_ty = self.type_memref(ty, &[count], None).try_into().unwrap();
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

    fn dynamic_alloca(&mut self, size: Self::Value, align: rustc_abi::Align) -> Self::Value {
        // add dynamic_size in memref::alloca
        todo!();
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        // If the type is memref, we need to load the address (See store).
        let load_ty = if ty.is_mem_ref() { self.type_index() } else { ty };
        // ptr is almost always memref<sizexi8>. Must be casted into memref<ty>
        let src_ptr_ty = ptr.r#type();
        let src_memref_ty: MemRefType<'ml> = src_ptr_ty.try_into().unwrap();
        let ptr = if load_ty != src_memref_ty.element() {
            self.mlir_memref_view(
                ptr,
                self.type_memref(load_ty, &[1], src_memref_ty.memory_space()),
                None,
            )
        } else {
            ptr
        };
        let mut loaded =
            self.mlir_load(load_ty, ptr, &[self.const_value(0, self.type_index())], align);

        // If the type is memref, we need to cast the address to the correct type.
        if ty.is_mem_ref() {
            loaded = self.inttoptr(loaded, ty);
        }
        loaded
    }

    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value {
        todo!()
    }

    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
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

        let val = if place.val.llextra.is_some() {
            OperandValue::Ref(place.val)
        } else if self.cx.is_backend_immediate(place.layout) {
            let llval =
                self.load(self.mlir_type(place.layout, true), place.val.llval, place.val.align);
            OperandValue::Immediate(llval)
        } else if let BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a.primitive().size(self).align_to(b.primitive().align(self).abi);

            let mut load = |i, scalar: rustc_abi::Scalar, align| {
                let base_ty = self.scalar_pair_element_backend_type(place.layout, i, false);
                let ptr = place.val.llval;
                let offset = if i == 0 {
                    None
                } else {
                    Some(self.const_value(b_offset.bytes(), self.type_index()))
                };
                let ptr_memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
                let ptr = if i > 0 || base_ty != ptr_memref_ty.element() {
                    self.mlir_memref_view(
                        ptr,
                        self.type_memref(base_ty, &[1], ptr_memref_ty.memory_space()),
                        offset,
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
        todo!()
    }

    fn range_metadata(&mut self, load: Self::Value, range: rustc_abi::WrappingRange) {
        todo!()
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        todo!()
    }

    fn emit_constant(&mut self, val: u64, ty: Self::Type) -> Self::Value {
        self.mlir_const_val_from_type(val, ty, self.cur_block())
    }

    fn emit_gpu_scalar_to_backend(
        &self,
        cv: rustc_const_eval::interpret::Scalar,
        layout: rustc_abi::Scalar,
        ty: Self::Type,
    ) -> Self::Value {
        match cv {
            rustc_const_eval::interpret::Scalar::Int(int) => {
                assert_eq!(int.size(), layout.primitive().size(self));
                let data = int.to_uint(int.size());

                if let rustc_abi::Primitive::Pointer(_) = layout.primitive() {
                    if data == 0 { self.const_null(ty) } else { self.const_undef(ty) }
                } else {
                    self.mlir_const_val_from_type(
                        cv.assert_scalar_int().to_int(int.size()),
                        ty,
                        self.cur_block(),
                    )
                }
            }
            rustc_const_eval::interpret::Scalar::Ptr(ptr, s) => {
                let (prov, offset) = ptr.into_parts();
                let alloc_id = prov.alloc_id();
                trace!("scalar_to_backend ptr: {:?}", self.tcx.global_alloc(alloc_id));
                self.const_data_memref_from_alloc_id(alloc_id)
            }
        }
    }

    fn emit_llvm_volatile_and_load(&mut self, mask: Self::Value, ptr: Self::Value) {
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
        let llvm_raw_ptr_and = self.and(llvm_raw_ptr_int, mask);

        // Ptr to LLVM
        let llvm_ptr = self.inttollvmptr(llvm_raw_ptr_and, self.type_llvm_ptr());

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

    fn store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
    ) -> Self::Value {
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
        let target_memref_ty = MemRefType::new(store_ty, &[1], None, dst_memref_ty.memory_space());
        let ptr = if self.mlir_element_type(ptr_ty) != store_ty {
            dbg!("Implicit cast {} {}", ptr_ty, store_ty);
            self.mlir_memref_view(ptr, target_memref_ty.into(), None)
        } else {
            ptr
        };
        self.append_op(mlir_memref::store(val, ptr, &[const_idx], self.cur_loc()));
        val
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
        order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
        size: rustc_abi::Size,
    ) {
        todo!()
    }

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        todo!()
    }

    fn inbounds_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        if self.is_unreachable() {
            return ptr;
        }
        let op = self.inbounds_gep_op(ty, ptr, indices);
        self.append_op_res(op)
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
        todo!()
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
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
        let ret = self.append_op_res(
            melior::dialect::ods::memref::extract_aligned_pointer_as_index(
                self.mlir_ctx,
                self.type_index(),
                val,
                self.cur_loc(),
            )
            .into(),
        );
        self.intcast(ret, dest_ty, false)
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let align = rustc_abi::Align::from_bytes(8).unwrap();
        let base_memref = self.alloca(rustc_abi::Size::from_bytes(64), align);
        let zero = self.const_value(0, self.type_index());
        let one = self.const_value(1, self.type_index());
        let two = self.const_value(2, self.type_index());
        let ptr = self.inbounds_gep(self.type_i64(), base_memref, &[zero]);
        let val = self.intcast(val, self.type_i64(), false);
        self.store(val, ptr, align);
        let ptr = self.inbounds_gep(self.type_i64(), base_memref, &[one]);
        self.store(val, ptr, align);
        let ptr = self.inbounds_gep(self.type_i64(), base_memref, &[two]);
        self.store(self.const_value(0, self.type_i64()), ptr, align);
        if self.fn_shared_memory_size.read().unwrap()[&self.name] > 0 {
            panic!(
                "inttoptr: {:?} {} -> {} where memoryspace is unknown",
                self.cur_span,
                val.r#type(),
                dest_ty
            );
        }
        let casted_base_memref = self.mlir_memref_view(
            base_memref,
            unsafe { self._type_memref(dest_ty, &[1], None) },
            None,
        );
        self.mlir_load(dest_ty, casted_base_memref, &[zero], align)
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
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
            return self.const_value(const_val, dest_ty);
        }
        let op = if src_ty.is_index() || dest_ty.is_index() {
            // If either is index, we need to use index_cast
            melior::dialect::arith::index_cast(val, dest_ty, self.cur_loc())
        } else if !is_signed {
            melior::dialect::arith::extui(val, dest_ty, self.cur_loc())
        } else {
            melior::dialect::arith::extsi(val, dest_ty, self.cur_loc())
        };
        self.append_op_res(op)
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        if val.r#type().is_llvm_pointer_type() {
            self.append_op_res(
                melior::dialect::ods::llvm::ptrtoint(self.mlir_ctx, dest_ty, val, self.cur_loc())
                    .into(),
            )
        } else {
            self.mlir_memref_view(val, dest_ty, None)
        }
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
        let rhs = normalized_value(rhs_ty, rhs);

        let op = if lhs.r#type().is_index() {
            melior::dialect::arith::cmpi(
                self.mlir_ctx,
                predicate,
                lhs,
                self.intcast(rhs, self.type_index(), false),
                self.cur_loc(),
            )
        } else {
            melior::dialect::arith::cmpi(self.mlir_ctx, predicate, lhs, rhs, self.cur_loc())
        };
        self.append_op_res(op)
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
        self.append_op_res(op)
    }

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: rustc_codegen_ssa_gpu::MemFlags,
    ) {
        if self.is_unreachable() {
            return;
        }
        let dst_ty = MemRefType::try_from(dst.r#type()).unwrap();
        let src_ty = MemRefType::try_from(src.r#type()).unwrap();
        let dst_ty = if dst_ty.memory_space() != src_ty.memory_space() {
            let mut sizes = vec![];
            for i in 0..dst_ty.rank() {
                sizes.push(dst_ty.dim_size(i).unwrap() as i64);
            }
            self.type_memref(dst_ty.element(), &sizes, src_ty.memory_space())
        } else {
            dst_ty.into()
        };
        let src = self.mlir_memref_view(src, dst_ty, None);
        let op = melior::dialect::ods::memref::copy(self.mlir_ctx, dst, src, self.cur_loc()).into();
        self.append_op(op);
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
        todo!()
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
        } else if agg_val.r#type().is_tuple() {
            dbg!(agg_val);
            todo!()
        } else {
            let Ok(op_val) = mlir_ir::operation::OperationResult::<'ml, 'a>::try_from(agg_val)
            else {
                panic!("agg_val is not an operation result");
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
        /*if agg_val == self.const_poison(agg_val.r#type()) {
            return elt;
        } else {
            return &[agg_val, elt];
        }
        warn!("insert_value {} {} {}", agg_val, elt, idx);*/
        todo!()
    }

    fn set_personality_fn(&mut self, personality: Self::Value) {
        todo!()
    }

    fn cleanup_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn filter_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
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
        order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
        failure_order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn atomic_rmw(
        &mut self,
        op: rustc_codegen_ssa_gpu::common::AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
    ) -> Self::Value {
        todo!()
    }

    fn atomic_fence(
        &mut self,
        order: rustc_codegen_ssa_gpu::common::AtomicOrdering,
        scope: rustc_codegen_ssa_gpu::common::SynchronizationScope,
    ) {
        todo!()
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
        instance: Option<rustc_middle::ty::Instance<'tcx>>,
    ) -> Self::Value {
        if self.is_unreachable() {
            return llfn;
        }
        let args = args.iter().map(|arg| self.use_value(*arg)).collect::<Vec<_>>();
        let args = &args;
        let ftype = fn_abi.map(|abi| {
            self.fn_abi_to_fn_type(abi, false).unwrap_or_else(|e| self.emit_error(e, self.cur_span))
        });
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
}
