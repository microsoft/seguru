use melior::ir::attribute::{ArrayAttribute, DenseI32ArrayAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationMutLike};
use melior::ir::{
    self as mlir_ir, Attribute, Block, BlockLike, Location, RegionLike, ShapedTypeLike, ValueLike,
};
use mlir_sys::{MlirAffineExpr, mlirAffineDimExprGet, mlirAffineMapGet, mlirAffineSymbolExprGet};

fn affine_dim(ctx: &melior::Context, pos: isize) -> MlirAffineExpr {
    unsafe { mlirAffineDimExprGet(ctx.to_raw(), pos) }
}

#[allow(dead_code)]
fn affine_sym(ctx: &melior::Context, pos: isize) -> MlirAffineExpr {
    unsafe { mlirAffineSymbolExprGet(ctx.to_raw(), pos) }
}

fn affine_identity_map(ctx: &melior::Context, dim_count: isize) -> Attribute {
    let dims = (0..dim_count).map(|i| affine_dim(ctx, i)).collect::<Vec<_>>();
    //let sym = mlirAffineSymbolExprGet(ctx, 0);// only used if not identity map
    let nexprs = dims.len() as isize;
    let exprs = dims.as_ptr() as _;
    unsafe {
        let affine_map = mlirAffineMapGet(
            ctx.to_raw(),
            dim_count,
            0,
            nexprs,
            exprs, // exprs
        );
        Attribute::from_raw(mlir_sys::mlirAffineMapAttrGet(affine_map))
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum IterType {
    Parallel,
    Reduction,
    Window,
}

impl IterType {
    fn as_str(&self) -> &str {
        match self {
            IterType::Parallel => "parallel",
            IterType::Reduction => "reduction",
            IterType::Window => "window",
        }
    }
    fn as_attr<'ml>(&self, ctx: &'ml melior::Context) -> Attribute<'ml> {
        Attribute::parse(ctx, &format!("#linalg.iterator_type<{}>", self.as_str()))
            .expect("valid iterator type")
    }
}

#[allow(dead_code)]
pub(crate) fn linalg_add_op<'ml, 'a>(
    ctx: &'ml melior::Context,
    left: mlir_ir::Value<'ml, 'a>,
    right: mlir_ir::Value<'ml, 'a>,
    output: mlir_ir::Value<'ml, 'a>,
    elem_ty: mlir_ir::Type<'ml>,
    loc: Location<'ml>,
) -> mlir_ir::Operation<'ml> {
    let block = melior::ir::Block::new(&[(elem_ty, loc), (elem_ty, loc), (elem_ty, loc)]);
    let left_arg = block.argument(0).unwrap();
    let right_arg = block.argument(1).unwrap();
    let op = block.append_operation(
        OperationBuilder::new("arith.addf", loc)
            .add_operands(&[left_arg.into(), right_arg.into()])
            .add_results(&[left_arg.r#type()])
            .build()
            .expect("valid operation"),
    );
    block.append_operation(
        OperationBuilder::new("linalg.yield", loc)
            .add_operands(&[op.result(0).unwrap().into()])
            .build()
            .expect("valid operation"),
    );
    linalg_generic_op(ctx, "linalg.add", &[left, right], &[output], elem_ty, block, loc)
}

pub(crate) fn linalg_copy_op<'ml, 'a>(
    ctx: &'ml melior::Context,
    src: mlir_ir::Value<'ml, 'a>,
    output: mlir_ir::Value<'ml, 'a>,
    elem_ty: mlir_ir::Type<'ml>,
    loc: Location<'ml>,
) -> mlir_ir::Operation<'ml> {
    let block = melior::ir::Block::new(&[(elem_ty, loc), (elem_ty, loc)]);
    let src_arg = block.argument(0).unwrap();
    let src_ty = src.r#type();
    let src_memref_ty = mlir_ir::r#type::MemRefType::try_from(src_ty).unwrap();
    assert!(src_memref_ty.rank() == 1);
    assert!(src_memref_ty.element() == elem_ty);
    block.append_operation(
        OperationBuilder::new("linalg.yield", loc)
            .add_operands(&[src_arg.into()])
            .build()
            .expect("valid operation"),
    );
    linalg_generic_op(ctx, "linalg.copy", &[src], &[output], elem_ty, block, loc)
}

pub(crate) fn linalg_generic_op<'ml, 'a>(
    ctx: &'ml melior::Context,
    name: &str,
    ins: &[mlir_ir::Value<'ml, 'a>],
    outs: &[mlir_ir::Value<'ml, 'a>],
    elem_ty: mlir_ir::Type<'ml>,
    block: Block<'ml>,
    loc: Location<'ml>,
) -> mlir_ir::Operation<'ml> {
    let region = melior::ir::Region::new();
    region.append_block(block);
    let mut op = OperationBuilder::new("linalg.generic", loc)
        .add_operands(ins)
        .add_operands(outs)
        .add_attributes(&[(
            melior::ir::Identifier::new(ctx, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(
                ctx,
                &[
                    ins.len() as _, // Must set arg1 ptr
                    outs.len() as _,
                ],
            )
            .into(),
        )])
        .add_regions([region])
        .build()
        .expect("valid operation");
    op.set_attribute("library_call", StringAttribute::new(ctx, name).into());
    let identity_map = affine_identity_map(ctx, 1);
    let identity_maps = vec![identity_map; ins.len() + outs.len()];
    op.set_attribute("indexing_maps", ArrayAttribute::new(ctx, &identity_maps).into());
    let iter_attr = IterType::Parallel.as_attr(ctx);
    op.set_attribute("iterator_types", ArrayAttribute::new(ctx, &[iter_attr]).into());
    op
}
