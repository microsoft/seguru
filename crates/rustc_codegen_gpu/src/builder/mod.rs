mod abi;
mod coverage;
mod debug;
mod intrinsic;

use crate::mlir::ValueToOpRef;
use log_derive::logfn;
use melior::dialect::memref as mlir_memref;
use melior::dialect::ods::memref as mlir_ods_memref;
use melior::helpers::BuiltinBlockExt;
use melior::ir::attribute::DenseI64ArrayAttribute;
use melior::ir::r#type::{self as mlir_type, MemRefType};
use rustc_abi::BackendRepr;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use std::collections::HashMap;
use std::fmt::Display;
use std::{marker::PhantomData, ops::Deref};

use melior::ir::{
    self as mlir_ir, BlockLike, Location, OperationRef, RegionLike, RegionRef, TypeLike, ValueLike,
};
use rustc_codegen_ssa::traits::{
    AsmBuilderMethods, BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, StaticBuilderMethods,
};

use crate::mlir::BlockRefWithTime;
use crate::{context::GPUCodegenContext, mlir::MLIROpHelpers};

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
    extra_state: GpuBuilderState<'ml, 'a>,
    dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
}

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub fn cur_loc(&self) -> Location<'ml> {
        self.cx.to_mlir_loc(self.cur_span)
    }

    pub fn cur_block(&self) -> &'a mlir_ir::Block<'ml> {
        unsafe { self.cur_block.to_ref() }
    }

    pub fn use_value(
        &mut self,
        val: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Value,
    ) -> <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Value {
        if let Ok(op) = val.is_from_op(Some("arith.constant")) {
            let op = self.cur_block().append_operation((op).clone());
            op.result(0).unwrap().into()
        } else {
            val
        }
    }

    fn append_op_res(&mut self, op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        self.cur_block().append_op_result(op).unwrap()
    }

    fn append_op(&mut self, op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
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

    // value is typed attribute, for example, 0: i32
    fn add_const_op(&self, value: impl Display, ty: mlir_ir::Type<'ml>) -> OperationRef<'ml, 'a> {
        let attribute = format!("{value} : {ty}");

        self.cur_block().append_operation(
            melior::dialect::ods::arith::constant(
                self.mlir_ctx,
                ty,
                mlir_ir::Attribute::parse(self.mlir_ctx, &attribute).unwrap(),
                self.cur_loc(),
            )
            .into(),
        )
    }

    fn mlir_load(
        &mut self,
        ty: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::Type,
        ptr: mlir_ir::Value<'ml, 'a>,
        indices: &[mlir_ir::Value<'ml, 'a>],
        align: rustc_abi::Align,
    ) -> mlir_ir::Value<'ml, 'a> {
        self.append_op_res(melior::dialect::memref::load(ptr, indices, self.cur_loc()).into())
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
        todo!();
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        todo!()
    }

    fn ret_void(&mut self) {
        let op = if self.inside_kernel_func() {
            self.cx.gpu_return(&[], self.cur_loc())
        } else {
            self.cx.cpu_return(&[], self.cur_loc())
        };
        self.cur_block().append_operation(op);
    }

    fn ret(&mut self, v: Self::Value) {
        let op = if self.inside_kernel_func() {
            log::debug!("gpu_return val: {:?} ty: {}", v, v.r#type());
            self.cx.gpu_return(&[self.use_value(v)], self.cur_loc())
        } else {
            self.cx.cpu_return(&[self.use_value(v)], self.cur_loc())
        };
        self.cur_block().append_operation(op);
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        let op = melior::dialect::cf::br(&*dest, &[], self.cur_loc());
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
            &*then_llbb,
            &*else_llbb,
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
            self.mlir_const_int_from_type(0, self.type_i1(), self.cur_block()),
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
        if val.r#type() == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: rustc_abi::Scalar) -> Self::Value {
        val
    }

    fn alloca(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
        let mut count = 1i64;
        let ty = if self.skip_op(true) {
            if self.span_to_type.contains_key(&self.cur_span) {
                self.span_to_type[&self.cur_span]
            } else {
                match size.bytes() {
                    1 => self.type_i8(),
                    2 => self.type_i16(),
                    4 => self.type_i32(),
                    8 => self.type_i64(),
                    16 => self.type_i128(),
                    _ => {
                        panic!("Unsupported size for alloca: {}", size.bytes());
                    }
                }
            }
        } else if self.span_to_type.contains_key(&self.cur_span) {
            let ty = self.span_to_type[&self.cur_span];
            log::debug!("alloc {:?} {}", self.cur_span, ty);
            ty
        } else {
            match size.bytes() {
                1 => self.type_i8(),
                2 => self.type_i16(),
                4 => self.type_i32(),
                8 => self.type_i64(),
                16 => self.type_i128(),
                val => {
                    if val / 8 * 8 == val {
                        count = (val / 8) as i64;
                        self.type_i8()
                    } else {
                        panic!("Unsupported size for alloca: {}", size.bytes());
                    }
                }
            }
        };

        log::debug!("alloc {} {}", ty, size.bytes());
        let loc = self.cur_loc();
        dbg!(align);
        let mem_ref_ty = mlir_type::MemRefType::new(ty, &[count], None, None);
        let op = melior::dialect::memref::alloca(
            self.mlir_ctx,
            mem_ref_ty,
            &[],
            &[],
            Some(mlir_ir::attribute::IntegerAttribute::new(
                self.cx.type_i64(),
                align.bytes() as i64,
            )),
            loc,
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
            &[self.mlir_const_int_from_type(0, self.type_index(), self.cur_block())],
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
        let val = self.use_value(val);
        if self.skip_op(true) {
            return val;
        }
        let const_idx = self.mlir_const_int_from_type(0, self.type_index(), self.cur_block());
        let op = mlir_memref::store(val, ptr, &[const_idx], self.cur_loc());
        self.cur_block().append_operation(op);
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
        /*if self.skip_op(false) {
            return ptr;
        }*/
        if indices.len() != 1 {
            panic!("only supports single index");
        }
        let strides = [];
        let sizes = [];
        let static_offsets = DenseI64ArrayAttribute::new(self.mlir_ctx, &[]).into();
        let static_size = DenseI64ArrayAttribute::new(self.mlir_ctx, &[]).into();
        let static_strides = DenseI64ArrayAttribute::new(self.mlir_ctx, &[]).into();

        let result_ty = MemRefType::new(ty, &[indices.len() as i64], None, None);
        //let op = self.cur_block().append_operation(
        mlir_ods_memref::reinterpret_cast(
            self.mlir_ctx,
            ty,
            ptr,
            indices,
            &sizes,
            &strides,
            static_offsets,
            static_size,
            static_strides,
            self.cur_loc(),
        );
        // .into(),
        //);
        let op =
            self.cur_block()
                .append_operation(mlir_memref::cast(ptr, result_ty, self.cur_loc()));
        op.result(0).unwrap().into()
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
        todo!()
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        let src_ty = val.r#type();
        let op = if src_ty.is_index() {
            assert!(dest_ty.is_integer());
            melior::dialect::arith::index_cast(val, dest_ty, self.cur_loc())
        } else {
            todo!()
        };
        self.append_op_res(op)
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        dbg!(self.cur_span);
        dbg!(val);
        dbg!(dest_ty);
        let op =
            melior::dialect::memref::cast(val, dest_ty.try_into().unwrap(), self.cur_loc()).into();
        self.append_op_res(op)
    }

    fn icmp(
        &mut self,
        op: rustc_codegen_ssa::common::IntPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        dbg!(op);
        dbg!(lhs);
        dbg!(rhs);
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
        let op =
            melior::dialect::arith::cmpi(self.mlir_ctx, predicate, lhs, rhs, self.cur_loc()).into();
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
        dbg!(cond);
        dbg!(then_val);
        dbg!(else_val);
        self.append_op_res(
            melior::dialect::arith::select(cond, then_val, else_val, self.cur_loc()).into(),
        )
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
                &[self.mlir_const_int_from_type(0, self.type_index(), self.cur_block())],
                self.cur_loc(),
            );
            let op = self.cur_block().append_operation(op.into());
            op.result(0).unwrap().into()
        } else if agg_val.r#type().is_tuple() {
            dbg!(agg_val);
            todo!()
        } else {
            dbg!(agg_val);
            dbg!(idx);
            let Ok(op_val) = mlir_ir::operation::OperationResult::<'ml, 'a>::try_from(agg_val)
            else {
                panic!("agg_val is not an operation result");
            };
            if idx == 0 {
                return agg_val;
            }
            self.op_to_extra_values[&op_val.owner().location().to_string()]
                .get(idx as usize)
                .unwrap()
                .clone()
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
        let ftype = fn_abi.map(|abi| self.fn_abi_to_fn_type(abi));
        let span = self.cur_span;
        let op = self
            .cx
            .call_op(llfn, args, ftype, &mut self.extra_state, span)
            .unwrap();
        if let Some(op) = op {
            let op = self.cur_block().append_operation(op);
            if op.result_count() > 0 {
                let ret: mlir_ir::Value<'ml, 'val> = op.result(0).unwrap().into();
                log::debug!("call op ret: {:?}", ret);
                self.span_to_type.insert(self.cur_span, ret.r#type());
                if op.result_count() > 1 {
                    let mut ret_vec = vec![];
                    for i in 0..op.result_count() {
                        let ret: mlir_ir::Value<'ml, 'val> = op.result(i).unwrap().into();
                        ret_vec.push(ret);
                    }
                    self.op_to_extra_values
                        .insert(op.location().to_string(), ret_vec);
                }
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
