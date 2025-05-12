mod abi;
mod coverage;
mod debug;
mod intrinsic;

use std::collections::HashMap;
use std::{marker::PhantomData, ops::Deref};

use log_derive::logfn;
use melior::dialect::memref as mlir_memref;
use melior::dialect::ods::memref as mlir_ods_memref;
use melior::ir::attribute::DenseI64ArrayAttribute;
use melior::ir::r#type::{self as mlir_type, MemRefType};

use melior::ir::{
    self as mlir_ir, BlockLike, Location, RegionLike, RegionRef, TypeLike, ValueLike,
};
use rustc_codegen_ssa::traits::{
    AsmBuilderMethods, BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    DerivedTypeCodegenMethods, StaticBuilderMethods,
};

use crate::mlir::{self, BlockRefWithTime};
use crate::{context::GPUCodegenContext, mlir::MLIROpHelpers};

enum FnStackState {
    Alloca,
    Store(usize),
}
pub(crate) struct GpuBuilder<'tcx, 'ml, 'a> {
    pub cx: &'a GPUCodegenContext<'tcx, 'ml, 'a>,
    pub cur_block: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::BasicBlock,
    pub cur_span: rustc_span::Span,
    pub span_to_type: HashMap<rustc_span::Span, mlir_type::Type<'ml>>,
    pub stack_state: FnStackState,
    pub extra_attrs: Vec<mlir_ir::Attribute<'ml>>,
    dummy: PhantomData<&'a mlir_ir::operation::Operation<'ml>>,
}

impl<'tcx, 'ml, 'a> GpuBuilder<'tcx, 'ml, 'a> {
    pub fn cur_loc(&self) -> Location<'ml> {
        self.cx.to_mlir_loc(self.cur_span)
    }

    pub fn cur_block(&self) -> &'a mlir_ir::Block<'ml> {
        unsafe { self.cur_block.to_ref() }
    }

    pub fn use_value(&mut self, val: Self::Value) {
        if self.stack_state == FnStackState::Store(0) {
            self.stack_state = FnStackState::Alloca;
        }
        if let Some(op) = self.cur_block().parent_operation() {
            op.use_value(val);
        }
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
            extra_attrs: vec![],
            dummy: PhantomData,
            span_to_type: HashMap::new(),
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
                .map(|t| (t.clone(), Location::unknown(cx.mlir_ctx)))
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
            self.cx.gpu_return(&[v], self.cur_loc())
        } else {
            self.cx.cpu_return(&[v], self.cur_loc())
        };
        self.cur_block().append_operation(op);
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        todo!()
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn alloca(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
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
        } else {
            if self.span_to_type.contains_key(&self.cur_span) {
                let ty = self.span_to_type[&self.cur_span];
                log::debug!("alloc {:?} {}", self.cur_span, ty);
                ty
            } else {
                todo!("{:?}", self.cur_span);
            }
            /*match size.bytes() {
                1 => self.type_i8(),
                2 => self.type_i16(),
                4 => self.type_i32(),
                8 => self.type_i64(),
                16 => self.type_i128(),
                _ => {
                    panic!("Unsupported size for alloca: {}", size.bytes());
                }
            }*/
        };

        log::debug!("alloc {} {}", ty, size.bytes());
        let loc = self.cur_loc();
        dbg!(align);
        let mem_ref_ty = mlir_type::MemRefType::new(ty, &[1], None, None);
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
        let op = self.cur_block().append_operation(op);
        op.result(0).unwrap().into()
    }

    #[logfn(TRACE)]
    fn dynamic_alloca(&mut self, size: Self::Value, align: rustc_abi::Align) -> Self::Value {
        // add dynamic_size in memref::alloca
        todo!();
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        todo!()
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
        todo!()
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
        if self.skip_op(false) {
            return ptr;
        }
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
        let op = self
            .cur_block()
            .append_operation(mlir_memref::cast(ptr, result_ty, self.cur_loc()).into());
        op.result(0).unwrap().into()
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
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
        todo!()
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
        let op = self.cur_block().append_operation(op);
        op.result(0).unwrap().into()
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn icmp(
        &mut self,
        op: rustc_codegen_ssa::common::IntPredicate,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Self::Value {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        let mut extra_attrs = vec![];
        extra_attrs.append(&mut self.extra_attrs);
        let op = self
            .call_op(llfn, args, ftype, &mut extra_attrs, span)
            .unwrap();
        self.extra_attrs = extra_attrs;
        if let Some(op) = op {
            let op = self.cur_block().append_operation(op);
            if op.result_count() > 0 {
                let ret: mlir_ir::Value<'ml, 'val> = op.result(0).unwrap().into();
                log::debug!("call op ret: {:?}", ret);
                self.span_to_type.insert(self.cur_span, ret.r#type());
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
