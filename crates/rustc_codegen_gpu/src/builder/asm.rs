use melior::ir::attribute::{IntegerAttribute, StringAttribute};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa_gpu::mir::operand::OperandValue;
use rustc_codegen_ssa_gpu::traits::{
    AsmBuilderMethods, BaseTypeCodegenMethods, BuilderMethods, InlineAsmOperandRef,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_target::asm::{InlineAsmRegClass, InlineAsmRegOrRegClass, NvptxInlineAsmRegClass};

use crate::builder::GpuBuilder;

fn reg_to_llvm(reg: InlineAsmRegOrRegClass) -> String {
    match reg {
        InlineAsmRegOrRegClass::Reg(reg) => {
            format!("{{{}}}", reg.name())
        }
        InlineAsmRegOrRegClass::RegClass(reg) => match reg {
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => "h",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => "r",
            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => "l",
            _ => unreachable!(),
        }
        .to_string(),
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
        // Collect the types of output operands
        let mut constraints = vec![];
        let mut output_types = vec![];
        let mut op_idx = FxHashMap::default();
        let mut inputs = vec![];
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::Out { reg, late, place } => {
                    let ty = if let Some(ref place) = place {
                        self.mlir_type(place.layout, false)
                    } else {
                        match reg.reg_class() {
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) => {
                                self.type_i16()
                            }
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) => {
                                self.type_i32()
                            }
                            InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) => {
                                self.type_i64()
                            }
                            _ => unreachable!(),
                        }
                    };
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg)));
                }
                InlineAsmOperandRef::InOut { reg, late, in_value, out_place } => {
                    let layout = if let Some(ref out_place) = out_place {
                        &out_place.layout
                    } else {
                        // LLVM required tied operands to have the same type,
                        // so we just use the type of the input.
                        &in_value.layout
                    };
                    let ty = self.mlir_type(*layout, false);
                    output_types.push(ty);
                    op_idx.insert(idx, constraints.len());
                    let prefix = if late { "=" } else { "=&" };
                    constraints.push(format!("{}{}", prefix, reg_to_llvm(reg)));
                }
                _ => {}
            }
        }

        dbg!(&op_idx);

        // Collect input operands
        for (idx, op) in operands.iter().enumerate() {
            match *op {
                InlineAsmOperandRef::In { reg, value } => {
                    let llval = value.immediate();
                    inputs.push(llval);
                    op_idx.insert(idx, constraints.len());
                    constraints.push(reg_to_llvm(reg));
                }
                InlineAsmOperandRef::InOut { reg, late: _, in_value, out_place: _, .. } => {
                    let value = in_value.immediate();
                    inputs.push(value);
                    constraints.push(reg_to_llvm(reg).to_string());
                }
                InlineAsmOperandRef::SymFn { instance } => {
                    todo!();
                    /*inputs.push(self.cx.get_fn_addr(instance));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());*/
                }
                InlineAsmOperandRef::SymStatic { def_id } => {
                    todo!();
                    /*inputs.push(self.cx.get_static(def_id));
                    op_idx.insert(idx, constraints.len());
                    constraints.push("s".to_string());*/
                }
                _ => {}
            }
        }

        dbg!(&op_idx);

        // Build the template string
        let mut template_str = String::new();
        for piece in template {
            match *piece {
                InlineAsmTemplatePiece::String(ref s) => {
                    if s.contains('$') {
                        for c in s.chars() {
                            if c == '$' {
                                template_str.push_str("$$");
                            } else {
                                template_str.push(c);
                            }
                        }
                    } else {
                        template_str.push_str(s)
                    }
                }
                InlineAsmTemplatePiece::Placeholder { operand_idx, span, .. } => {
                    match operands[operand_idx] {
                        InlineAsmOperandRef::In { .. }
                        | InlineAsmOperandRef::Out { .. }
                        | InlineAsmOperandRef::InOut { .. } => {
                            template_str.push_str(&format!("${{{}}}", op_idx[&operand_idx]));
                        }
                        InlineAsmOperandRef::Const { ref string } => {
                            // Const operands get injected directly into the template
                            template_str.push_str(string);
                        }
                        InlineAsmOperandRef::SymFn { .. }
                        | InlineAsmOperandRef::SymStatic { .. } => {
                            // Only emit the raw symbol name
                            template_str.push_str(&format!("${{{}:c}}", op_idx[&operand_idx]));
                        }
                        InlineAsmOperandRef::Label { .. } => {
                            // template_str.push_str(&format!("${{{}:l}}", constraints.len()));
                            // constraints.push("!i".to_owned());
                            // labels.push(label);

                            self.tcx
                                .sess
                                .dcx()
                                .span_fatal(span, "Operands with label refs are unsupported");
                        }
                    }
                }
            }
        }

        /*if !options.contains(InlineAsmOptions::NOMEM) {
            // This is actually ignored by LLVM, but it's probably best to keep
            // it just in case. LLVM instead uses the ReadOnly/ReadNone
            // attributes on the call instruction to optimize.
            constraints.push("~{memory}".to_string());
        }*/
        let volatile = !options.contains(InlineAsmOptions::PURE);
        let alignstack = !options.contains(InlineAsmOptions::NOSTACK);

        let ctx = self.mlir_ctx;
        let mut builder =
            melior::dialect::ods::llvm::InlineAsmOperationBuilder::new(ctx, self.cur_loc())
                .asm_string(StringAttribute::new(ctx, template_str.as_str()))
                .constraints(StringAttribute::new(ctx, constraints.join(",").as_str()));
        if output_types.len() == 1 {
            builder = builder.res(output_types[0]);
        } else if output_types.len() > 1 {
            self.emit_error(
                "Multiple output operands are not supported in inline assembly".to_string(),
                self.cur_span,
            );
        }
        builder = builder.asm_dialect(
            IntegerAttribute::new(self.type_i8(), 0).into(), //ATT
        );
        if alignstack {
            builder = builder.is_align_stack(melior::ir::Attribute::unit(ctx));
        }
        if volatile {
            builder = builder.has_side_effects(melior::ir::Attribute::unit(ctx));
        }
        use melior::ir::{TypeLike, ValueLike};
        for i in &mut inputs {
            if i.r#type().is_mem_ref() {
                let addr = self.ptrtoint(*i, self.type_i64());
                *i = self.inttollvmptr(addr, self.type_llvm_ptr());
            }
        }
        let builder = builder.operands(&inputs);
        let ops = self.append_op(builder.build().into());

        /*if options.contains(InlineAsmOptions::PURE) {
            if options.contains(InlineAsmOptions::NOMEM) {
                llvm::Attribute::ReadNone.apply_callsite(llvm::AttributePlace::Function, result);
            } else if options.contains(InlineAsmOptions::READONLY) {
                llvm::Attribute::ReadOnly.apply_callsite(llvm::AttributePlace::Function, result);
            }
        }*/

        // Write results to outputs
        for (idx, op) in operands.iter().enumerate() {
            if let InlineAsmOperandRef::Out { place: Some(place), .. }
            | InlineAsmOperandRef::InOut { out_place: Some(place), .. } = *op
            {
                let value = ops.result(idx).unwrap().into();
                OperandValue::Immediate(value).store(self, place);
            }
        }
        //panic!("Inline assembly is not supported in GPU codegen yet {:?} {:?} {:?} {:?} {:?} {:?}", template, operands, options, line_spans, instance, dest);
    }
}
