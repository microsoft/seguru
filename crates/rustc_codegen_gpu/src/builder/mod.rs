mod abi;
mod coverage;
mod debug;
mod intrinsic;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;

use log_derive::logfn;
use melior::dialect::memref as mlir_memref;
use melior::helpers::BuiltinBlockExt;
use melior::ir::r#type::{self as mlir_type, MemRefType};
use melior::ir::{
    self as mlir_ir, BlockLike, Location, RegionLike, RegionRef, TypeLike, Value, ValueLike,
};
use rustc_abi::BackendRepr;
use rustc_codegen_ssa_gpu::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa_gpu::traits::{
    AsmBuilderMethods, BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, StaticBuilderMethods,
};

use crate::context::GPUCodegenContext;
use crate::mlir::{BlockRefWithTime, MLIROpHelpers, ValueToOpRef};

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
    pub cur_block: <GpuBuilder<'tcx, 'ml, 'a> as BackendTypes>::BasicBlock,
    pub cur_span: rustc_span::Span,
    pub span_to_type: HashMap<rustc_span::Span, mlir_type::Type<'ml>>,
    pub op_to_extra_values: HashMap<String, Vec<mlir_ir::Value<'ml, 'a>>>,
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

    pub fn mlir_cast_memref(
        &self,
        val: melior::ir::Value<'ml, 'a>,
        dst_ty: melior::ir::Type<'ml>,
    ) -> Value<'ml, 'a> {
        assert!(val.r#type().is_mem_ref());
        assert!(dst_ty.is_mem_ref());
        let ty = val.r#type();
        let memref_ty: MemRefType<'ml> = ty.try_into().expect("expected memref type");
        let layout = crate::mlir::attr::StridedLayoutAttribute::try_from(memref_ty.layout());
        let base_memref = val;
        log::debug!("mlir_cast_memref layout: {:?}", layout);
        let (base_memref, byte_offset) = if layout.is_ok() && layout.unwrap().get_offset() != 0 {
            let op = crate::mlir::memref::extract_strided_metadata(
                self.cx.mlir_ctx,
                val,
                self.cur_loc(),
            );
            let op = self.append_op(op.into());
            let base_memref: Value<'ml, 'a> = op.result(0).unwrap().into();
            let byte_offset = op.result(1).unwrap().into();
            let element_ty = self.mlir_element_type(base_memref.r#type());
            let op = crate::mlir::memref::reinterpret_cast(
                self.cx.mlir_ctx,
                element_ty,
                base_memref,
                &[0],
                &[],
                &[self.static_size_of(element_ty) as i64],
                &[],
                &[1],
                &[],
                self.cur_loc(),
            );
            log::debug!("base memref: {:?}", val);
            (self.append_op_res(op), byte_offset)
        } else {
            (val, self.const_value(0, self.type_index()))
        };
        let op = melior::dialect::memref::view(
            self.cx.mlir_ctx,
            base_memref,
            byte_offset,
            &[],
            dst_ty.try_into().unwrap(),
            self.cur_loc(),
        );
        self.append_op_res(op)
    }

    fn append_op_res(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::Value<'ml, 'a> {
        self.cur_block().append_op_result(op).unwrap()
    }

    fn is_unreachable(&self) -> bool {
        self.cur_block().terminator().is_some()
    }

    fn append_op(&self, op: mlir_ir::Operation<'ml>) -> mlir_ir::OperationRef<'ml, 'a> {
        log::trace!("append_op: {:?}", op);
        if self.is_unreachable() {
            panic!("Cannot append operation to unreachable block");
        }
        self.cur_block().append_operation(op)
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
        let ptr = self.mlir_cast_memref(ptr, MemRefType::new(ty, &[1], None, None).into());
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
        Self {
            cx,
            cur_block: llbb,
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
        self.cur_block
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
        log::debug!("append_block: {:?}", name);
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
        log::debug!("append_sibling_block: {:?}", name);
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
        let op = melior::dialect::cf::br(&dest, &[], self.cur_loc());
        self.append_op(op);
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        use rustc_codegen_ssa_gpu::traits::AbiBuilderMethods;
        if self.is_unreachable() {
            return;
        }
        let mut then_llbb_args = vec![];
        for i in 0..then_llbb.argument_count() {
            then_llbb_args.push(self.get_param(i));
        }
        let mut else_llbb_args = vec![];
        for i in 0..else_llbb.argument_count() {
            else_llbb_args.push(self.get_param(i));
        }
        let op = melior::dialect::cf::cond_br(
            self.mlir_ctx,
            cond,
            &then_llbb,
            &else_llbb,
            &then_llbb_args,
            &else_llbb_args,
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
        let op = melior::dialect::cf::assert(
            self.mlir_ctx,
            self.const_value(0, self.type_i1()),
            "unreachable",
            self.cur_loc(),
        );
        self.append_op(op);
        self.ret_void();
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
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
        let op = melior::dialect::arith::subi(lhs, rhs, self.cur_loc());
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
        let op = melior::dialect::arith::muli(lhs, rhs, self.cur_loc());
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
        let op = melior::dialect::arith::shrsi(lhs, rhs, self.cur_loc());
        self.append_op_res(op)
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
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
        todo!()
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        log::trace!("from_immediate: {:?}", val);
        if val.r#type() == self.cx().type_i1() { self.zext(val, self.cx().type_i8()) } else { val }
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
        let mem_ref_ty = mlir_type::MemRefType::new(ty, &[count], None, None);
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
        self.mlir_load(ty, ptr, &[self.const_value(0, self.type_index())], align)
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
                log::trace!("scalar_to_backend ptr: {:?}", self.tcx.global_alloc(alloc_id));
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
        let llvm_ptr = self.inttoptr(llvm_raw_ptr_and, self.type_ptr());

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
        let val = self.use_value(val);
        let const_idx = self.const_value(0, self.type_index());
        let memref_ty = MemRefType::try_from(ptr.r#type()).unwrap();
        let target_memref_ty = MemRefType::new(val.r#type(), &[1], None, None);
        let ptr = if self.mlir_element_type(ptr.r#type()) != val.r#type() {
            dbg!("Implicit cast {} {}", ptr.r#type(), val.r#type());
            self.mlir_cast_memref(ptr, target_memref_ty.into())
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
        if indices.len() != 1 {
            panic!("only supports single index");
        }
        let size = self.static_size_of(ty);
        let static_sizes = vec![1_i64; indices.len()]; // Force to be 1 since this is a ptr?
        let static_strides = vec![1_i64; indices.len()];
        let idx = indices[0];
        let mut dynamic = false;
        for index in indices {
            if index.is_from_op(Some("arith.constant")).is_err() {
                dynamic = true;
            }
        }
        let (static_indices, dy_indices) = if !dynamic {
            (
                indices
                    .iter()
                    .map(|v| self.const_to_opt_uint(*v).unwrap() as i64)
                    .collect::<Vec<_>>(),
                vec![],
            )
        } else {
            let indices = indices
                .iter()
                .map(|v| {
                    if v.r#type().is_integer() {
                        let v = self.use_value(*v);
                        self.intcast(v, self.type_index(), false)
                    } else if v.r#type().is_index() {
                        self.use_value(*v)
                    } else {
                        panic!("Must be int or index type");
                    }
                })
                .collect::<Vec<_>>();
            (vec![], indices)
        };
        let base_ty = self.mlir_element_type(ptr.r#type());
        let op = crate::mlir::memref::reinterpret_cast(
            self.mlir_ctx,
            base_ty,
            ptr,
            &static_indices,
            &dy_indices,
            &static_sizes,
            &[],
            &static_strides,
            &[],
            self.cur_loc(),
        );
        self.append_op_res(op)
        //let op = self.append_op(mlir_memref::cast(ptr, result_ty, self.cur_loc()));
        //op.result(0).unwrap().into()
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        assert!(!self.is_unreachable());
        let op =
            melior::dialect::ods::llvm::trunc(self.mlir_ctx, dest_ty, val, self.cur_loc()).into();
        self.append_op_res(op)
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
        assert!(!self.is_unreachable());
        self.append_op_res(
            melior::dialect::ods::llvm::ptrtoint(self.mlir_ctx, dest_ty, val, self.cur_loc())
                .into(),
        )
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
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

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        assert!(!self.is_unreachable());
        let src_ty = val.r#type();
        if src_ty == dest_ty {
            return val;
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
        assert!(!self.is_unreachable());
        let op = if val.r#type().is_llvm_pointer_type() {
            melior::dialect::ods::llvm::ptrtoint(self.mlir_ctx, dest_ty, val, self.cur_loc()).into()
        } else {
            melior::dialect::ods::memref::cast(self.mlir_ctx, dest_ty, val, self.cur_loc()).into()
        };

        self.append_op_res(op)
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
        let mut closure_ptrs = vec![];
        if let Some(instance) = instance {
            let mut closure_count = 0;
            for arg in instance.args.iter() {
                if let rustc_type_ir::GenericArgKind::Type(ty) = arg.unpack() {
                    if let Some(c) = self.ty_to_closure(&ty) {
                        closure_count += 1;
                        closure_ptrs.push(c);
                    }
                }
            }
        }
        let args = &args;
        let ftype = fn_abi.map(|abi| {
            self.fn_abi_to_fn_type(abi).unwrap_or_else(|e| self.emit_error(e, self.cur_span))
        });
        let span = self.cur_span;
        let op = self.cx.call_op(llfn, args, ftype, &mut self.extra_state, span).unwrap();
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
                    self.op_to_extra_values.insert(op.location().to_string(), ret_vec);
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
        let op = melior::dialect::arith::extui(val, dest_ty, self.cur_loc());
        self.append_op_res(op)
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value) {
        todo!()
    }
}
