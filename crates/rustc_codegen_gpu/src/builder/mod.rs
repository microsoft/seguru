mod abi;
mod coverage;
mod debug;
mod intrinsic;

use crate::mlir::ValueToOpRef;
use log_derive::logfn;
use melior::dialect::memref as mlir_memref;
use melior::dialect::ods::memref as mlir_ods_memref;
use melior::helpers::BuiltinBlockExt;
use melior::ir::attribute::{DenseI64ArrayAttribute, TypeAttribute};
use melior::ir::r#type::{self as mlir_type, MemRefType};
use rustc_abi::BackendRepr;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use std::collections::HashMap;
use std::{marker::PhantomData, ops::Deref};

use crate::rustc_codegen_ssa::traits::MiscCodegenMethods;
use melior::ir::{
    self as mlir_ir, BlockLike, Location, RegionLike, RegionRef, TypeLike, ValueLike,
};
use rustc_codegen_ssa::traits::{
    AsmBuilderMethods, BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, StaticBuilderMethods,
};

use crate::mlir::BlockRefWithTime;
use crate::{context::GPUCodegenContext, mlir::MLIROpHelpers};
mod llvm;
enum FnStackState {
    Alloca,
    Store(usize),
}

pub(crate) struct GpuBuilderState<'ml, 'a> {
    pub attrs: Vec<mlir_ir::Attribute<'ml>>,
    pub args: HashMap<crate::attr::GpuItem, Vec<mlir_ir::Value<'ml, 'a>>>,
    pub inside_gpu_scope: bool,
}

impl<'ml, 'a> GpuBuilderState<'ml, 'a> {
    pub fn new() -> Self {
        Self {
            attrs: vec![],
            args: HashMap::new(),
            inside_gpu_scope: false,
        }
    }
}

pub(crate) struct GpuBuilder<'tcx, 'ml, 'a> {
    pub cx: &'a GPUCodegenContext<'tcx, 'ml, 'a>,
    pub cur_block: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::BasicBlock,
    pub cur_span: rustc_span::Span,
    pub span_to_type: HashMap<rustc_span::Span, mlir_type::Type<'ml>>,
    pub op_to_extra_values: HashMap<String, Vec<mlir_ir::Value<'ml, 'a>>>,
    stack_state: FnStackState,
    pub extra_state: GpuBuilderState<'ml, 'a>,
    dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
}

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub fn cur_loc(&self) -> Location<'ml> {
        self.cx.to_mlir_loc(self.cur_span)
    }

    pub fn cur_block(&self) -> &'a mlir_ir::Block<'ml> {
        unsafe { self.cur_block.to_ref() }
    }

    fn const_value(
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

    fn append_op_res(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        self.cur_block().append_op_result(op).unwrap()
    }

    fn append_op(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
        log::trace!("append_op: {:?}", op);
        self.cur_block().append_operation(op)
    }

    pub fn skip_op(&mut self, update: bool) -> bool {
        match self.stack_state {
            FnStackState::Alloca if self.cur_block().argument_count() > 0 => {
                if update {
                    self.stack_state = FnStackState::Store(0);
                }
                true
            }
            FnStackState::Store(idx) if idx < self.cur_block().argument_count() => {
                if update {
                    self.stack_state = FnStackState::Store(idx + 1);
                }
                true
            }
            _ => false,
        }
    }

    #[allow(dead_code)]
    pub fn inside_gpu_mod(&self) -> bool {
        if let Some(op) = self.cur_block().parent_operation() {
            op.is_gpu_func()
        } else {
            false
        }
    }

    pub fn inside_kernel_func(&self) -> bool {
        if let Some(op) = self.cur_block().parent_operation() {
            op.is_kernel_func()
        } else {
            false
        }
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
        self.append_op_res(melior::dialect::memref::load(ptr, indices, self.cur_loc()))
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
        operands: &[rustc_codegen_ssa::traits::InlineAsmOperandRef<'tcx, Self>],
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
        Self {
            cx,
            cur_block: llbb,
            stack_state: FnStackState::Alloca,
            cur_span: rustc_span::DUMMY_SP,
            extra_state: GpuBuilderState::new(),
            dummy: PhantomData,
            span_to_type: HashMap::new(),
            op_to_extra_values: HashMap::new(),
        }
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        todo!()
    }

    fn set_span(&mut self, span: rustc_span::Span) {
        self.cur_span = span;
        log::debug!("set span {:?}", span);
    }

    fn append_block(
        cx: &'a Self::CodegenCx,
        llfn: mlir_ir::operation::OperationRef<'ml, 'a>,
        name: &str,
    ) -> Self::BasicBlock {
        let name = rustc_data_structures::small_c_str::SmallCStr::new(name);
        let region: RegionRef<'ml, 'a> = unsafe { llfn.to_ref() }.region(0).unwrap();
        let types = llfn.get_op_operands_types();
        let block: mlir_ir::BlockRef<'ml, 'a> = region.append_block(melior::ir::Block::new(
            &types
                .iter()
                .map(|t| (*t, Location::unknown(cx.mlir_ctx)))
                .collect::<Vec<_>>(),
        ));
        block
        //llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, name.as_ptr())
    }

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock {
        self.cur_block()
            .parent_region()
            .as_ref()
            .unwrap()
            .append_block(melior::ir::Block::new(&[]))
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        self.br(llbb);
    }

    fn ret_void(&mut self) {
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&[], self.cur_loc())
        } else {
            self.cx.cpu_return(&[], self.cur_loc())
        };
        self.append_op(op);
    }

    fn ret(&mut self, v: Self::Value) {
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&[self.use_value(v)], self.cur_loc())
        } else {
            self.cx.cpu_return(&[self.use_value(v)], self.cur_loc())
        };
        self.append_op(op);
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        let op = melior::dialect::cf::br(&dest, &[], self.cur_loc());
        self.append_op(op);
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        let op = melior::dialect::cf::cond_br(
            self.mlir_ctx,
            cond,
            &then_llbb,
            &else_llbb,
            &[],
            &[],
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
        todo!()
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
        let op = melior::dialect::cf::assert(
            self.mlir_ctx,
            self.mlir_const_val_from_type(0, self.type_i1(), self.cur_block()),
            "unreachable",
            self.cur_loc(),
        );
        self.append_op(op);
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        todo!()
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        todo!()
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        todo!()
    }

    fn checked_binop(
        &mut self,
        oop: rustc_codegen_ssa::traits::OverflowOp,
        ty: rustc_middle::ty::Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        log::trace!("from_immediate: {:?}", val);
        if val.r#type() == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: rustc_abi::Scalar) -> Self::Value {
        log::trace!("to_immediate_scalar: {:?} {:?}", val, scalar);
        val
    }

    fn alloca(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
        let mut count = 1i64;
        let ty = if let Some(ty) = self.get_type_by_span(&self.cur_span) {
            ty   
        } else {
            count = size.bytes() as i64;
            self.type_i8()
        };
        let mem_ref_ty =  mlir_type::MemRefType::new(ty, &[count], None, None);;
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

    #[logfn(TRACE)]
    fn dynamic_alloca(&mut self, size: Self::Value, align: rustc_abi::Align) -> Self::Value {
        // add dynamic_size in memref::alloca
        todo!();
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        self.mlir_load(
            ty,
            ptr,
            &[self.mlir_const_val_from_type(0, self.type_index(), self.cur_block())],
            align,
        )
    }

    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value {
        todo!()
    }

    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        size: rustc_abi::Size,
    ) -> Self::Value {
        todo!()
    }

    fn load_operand(
        &mut self,
        place: rustc_codegen_ssa::mir::place::PlaceRef<'tcx, Self::Value>,
    ) -> rustc_codegen_ssa::mir::operand::OperandRef<'tcx, Self::Value> {
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        let val = if place.val.llextra.is_some() {
            OperandValue::Ref(place.val)
        } else if self.cx.is_backend_immediate(place.layout) {
            let llval = self.load(
                self.mlir_type(place.layout, true),
                place.val.llval,
                place.val.align,
            );
            OperandValue::Immediate(llval)
        } else if let BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a
                .primitive()
                .size(self)
                .align_to(b.primitive().align(self).abi);

            let mut load = |i, scalar: rustc_abi::Scalar, align| {
                self.mlir_load(
                    self.scalar_pair_element_backend_type(place.layout, i, false),
                    place.val.llval,
                    &[self.const_usize(b_offset.bytes())],
                    align,
                )
            };

            OperandValue::Pair(
                load(0, a, place.val.align),
                load(1, b, place.val.align.restrict_for_offset(b_offset)),
            )
        } else {
            OperandValue::Ref(place.val)
        };
        OperandRef {
            val,
            layout: place.layout,
        }
    }

    fn write_operand_repeatedly(
        &mut self,
        elem: rustc_codegen_ssa::mir::operand::OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: rustc_codegen_ssa::mir::place::PlaceRef<'tcx, Self::Value>,
    ) {
        todo!()
    }

    fn range_metadata(&mut self, load: Self::Value, range: rustc_abi::WrappingRange) {
        todo!()
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        todo!()
    }

    fn store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
    ) -> Self::Value {
        /*if ptr.r#type().is_llvm_pointer_type() {
            let op = melior::dialect::llvm::store(
                self.mlir_ctx,
                self.val_to_llvm_value(val),
                ptr,
                self.cur_loc(),
                melior::dialect::llvm::LoadStoreOptions::new()
                    .align(Some(self.align_to_attr(align))),
            );
            self.append_op(op);
            return val;
        }*/
        let val = self.use_value(val);
        let const_idx = self.mlir_const_val_from_type(0, self.type_index(), self.cur_block());
        let memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
        let target_memref_ty = MemRefType::new(val.r#type(), &[1], Some(memref_ty.layout()), memref_ty.memory_space());
        let prt = if self.mlir_element_type(ptr.r#type()) != val.r#type() {
            dbg!("Implicit cast {} {}", ptr.r#type(), val.r#type());
            let op = melior::dialect::memref::view(self.cx.mlir_ctx, ptr, const_idx, &[], target_memref_ty, self.cur_loc());
            self.append_op_res(op.into())
        } else {
            ptr
        };
        let op = mlir_memref::store(val, ptr, &[const_idx], self.cur_loc());
        self.append_op(op);
        // TODO(check): memref::store op does not return a value.
        val
    }

    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
        flags: rustc_codegen_ssa::MemFlags,
    ) -> Self::Value {
        self.store(val, ptr, align)
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
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
        if indices.len() != 1 {
            panic!("only supports single index");
        }
        let strides = [];
        let sizes = [];

        let static_offsets = DenseI64ArrayAttribute::new(self.mlir_ctx, &[0]).into();
        let static_sizes =
            DenseI64ArrayAttribute::new(self.mlir_ctx, &[self.static_size_of(ty) as i64]).into();
        let static_strides = DenseI64ArrayAttribute::new(self.mlir_ctx, &[1]).into();

        let result_ty = MemRefType::new(ty, &[indices.len() as i64], None, None);
        let mut op = mlir_ods_memref::reinterpret_cast(
            self.mlir_ctx,
            result_ty.into(),
            ptr,
            indices,
            &sizes,
            &strides,
            static_offsets,
            static_sizes,
            static_strides,
            self.cur_loc(),
        );
        op.set_static_sizes(static_sizes);
        op.set_static_offsets(static_offsets);
        op.set_static_strides(static_strides);
        let mut op: mlir_ir::Operation<'ml> = op.into();
        op.set_attribute(
            "operand_segment_sizes",
            mlir_ir::attribute::DenseI32ArrayAttribute::new(self.cx.mlir_ctx, &[1, 1, 0, 0]).into(),
        );
        self.append_op_res(op.into())
        //let op = self.append_op(mlir_memref::cast(ptr, result_ty, self.cur_loc()));
        //op.result(0).unwrap().into()
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let op =
            melior::dialect::ods::llvm::trunc(self.mlir_ctx, dest_ty, val, self.cur_loc()).into();
        self.append_op_res(op)
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
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
        todo!()
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.append_op_res(
            melior::dialect::ods::llvm::ptrtoint(self.mlir_ctx, dest_ty, val, self.cur_loc())
                .into(),
        )
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let int64_val = self.intcast(val, self.type_i64(), false);
        let op =
            melior::dialect::ods::llvm::inttoptr(self.mlir_ctx, dest_ty, int64_val, self.cur_loc())
                .into();
        self.append_op_res(op)
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        let src_ty = val.r#type();
        assert!(dest_ty.is_integer());
        let op = if src_ty.is_index() {
            melior::dialect::arith::index_cast(val, dest_ty, self.cur_loc())
        } else {
            melior::dialect::arith::index_cast(val, dest_ty, self.cur_loc())
        };
        self.append_op_res(op)
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let op = if val.r#type().is_llvm_pointer_type() {
            melior::dialect::ods::llvm::ptrtoint(self.mlir_ctx, dest_ty, val, self.cur_loc()).into()
        } else {
            melior::dialect::ods::memref::cast(self.mlir_ctx, dest_ty, val, self.cur_loc()).into()
        };

        self.append_op_res(op)
    }

    fn icmp(
        &mut self,
        op: rustc_codegen_ssa::common::IntPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        let predicate = match op {
            rustc_codegen_ssa::common::IntPredicate::IntEQ => {
                melior::dialect::arith::CmpiPredicate::Eq
            }
            rustc_codegen_ssa::common::IntPredicate::IntNE => {
                melior::dialect::arith::CmpiPredicate::Ne
            }
            rustc_codegen_ssa::common::IntPredicate::IntUGT => {
                melior::dialect::arith::CmpiPredicate::Ugt
            }
            rustc_codegen_ssa::common::IntPredicate::IntUGE => {
                melior::dialect::arith::CmpiPredicate::Uge
            }
            rustc_codegen_ssa::common::IntPredicate::IntULT => {
                melior::dialect::arith::CmpiPredicate::Ult
            }
            rustc_codegen_ssa::common::IntPredicate::IntULE => {
                melior::dialect::arith::CmpiPredicate::Ule
            }
            rustc_codegen_ssa::common::IntPredicate::IntSGT => {
                melior::dialect::arith::CmpiPredicate::Sgt
            }
            rustc_codegen_ssa::common::IntPredicate::IntSGE => {
                melior::dialect::arith::CmpiPredicate::Sge
            }
            rustc_codegen_ssa::common::IntPredicate::IntSLT => {
                melior::dialect::arith::CmpiPredicate::Slt
            }
            rustc_codegen_ssa::common::IntPredicate::IntSLE => {
                melior::dialect::arith::CmpiPredicate::Sle
            }
        };
        let op = melior::dialect::arith::cmpi(self.mlir_ctx, predicate, lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn fcmp(
        &mut self,
        op: rustc_codegen_ssa::common::RealPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        todo!()
    }

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: rustc_codegen_ssa::MemFlags,
    ) {
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
        flags: rustc_codegen_ssa::MemFlags,
    ) {
        todo!()
    }

    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: rustc_abi::Align,
        flags: rustc_codegen_ssa::MemFlags,
    ) {
        todo!()
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        self.append_op_res(melior::dialect::arith::select(
            cond,
            then_val,
            else_val,
            self.cur_loc(),
        ))
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
        if agg_val.r#type().is_ranked_tensor() {
            let op = melior::dialect::ods::tensor::extract(
                self.mlir_ctx,
                self.element_type(agg_val.r#type()),
                agg_val,
                &[self.mlir_const_val_from_type(0, self.type_index(), self.cur_block())],
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
                .unwrap()
        }
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
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
        order: rustc_codegen_ssa::common::AtomicOrdering,
        failure_order: rustc_codegen_ssa::common::AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn atomic_rmw(
        &mut self,
        op: rustc_codegen_ssa::common::AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
    ) -> Self::Value {
        todo!()
    }

    fn atomic_fence(
        &mut self,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        scope: rustc_codegen_ssa::common::SynchronizationScope,
    ) {
        todo!()
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        todo!()
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: rustc_abi::Size) {
        todo!()
    }

    fn lifetime_end(&mut self, ptr: Self::Value, size: rustc_abi::Size) {
        todo!()
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
        dbg!(instance);
        let mut args = args.into_iter().map(|arg| self.use_value(*arg)).collect::<Vec<_>>();
        let mut closure_ptrs = vec![];
        if let Some(instance) = instance {
            let mut closure_count = 0;
            for arg in instance.args.iter() {
                if let rustc_type_ir::GenericArgKind::Type(ty) = arg.unpack() {
                    if let rustc_middle::ty::Closure(closure_def_id, closure_substs) = *ty.kind() {
                        // ✅ You’ve found the closure passed to this call.
                        // Closure type is represented as a type(ClosureArgs) that implements fn trait. Thus.
                        // the call only see the ClosureArgs as inputs and need to explicitly resolve the closure.
                        closure_count += 1;
                        let closure_inst = rustc_middle::ty::Instance::resolve_closure(
                            self.tcx,
                            closure_def_id,
                            closure_substs,
                            rustc_middle::ty::ClosureKind::FnOnce,
                        );
                        log::debug!("Closure def_id: {:?}", closure_def_id);
                        closure_ptrs.push(self.to_mir_func_const(closure_inst, Some(self.cur_block)));
                    }
                }
            }
        }
        let args = if closure_ptrs.is_empty() {
            &args
        } else if closure_ptrs.len() == 1 {
            closure_ptrs.append(&mut args);
            &closure_ptrs
        } else {
            // TODO: handle multiple closures by check the number of ClosureArgs and group args into different closures.
            unimplemented!()
        };
        let ftype = fn_abi.map(|abi| self.fn_abi_to_fn_type(abi));
        let span = self.cur_span;
        let op = self
            .cx
            .call_op(llfn, args, ftype, &mut self.extra_state, span)
            .unwrap();
        if let Some(op) = op {
            let op = self.append_op(op);
            if op.result_count() > 0 {
                let ret: mlir_ir::Value<'ml, 'val> = op.result(0).unwrap().into();
                if op.result_count() > 1 {
                    let mut ret_vec = vec![];
                    for i in 0..op.result_count() {
                        let ret: mlir_ir::Value<'ml, 'val> = op.result(i).unwrap().into();
                        ret_vec.push(ret);
                    }
                    self.op_to_extra_values
                        .insert(op.location().to_string(), ret_vec);
                }
                //self.span_to_type.insert(self.cur_span, ret.r#type());
                ret
            } else {
                llfn
            }
        } else {
            // This is a virtual call used by the compiler.
            llfn
        }
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value) {
        todo!()
    }
}
