use crate::mlir::attr::StridedLayoutAttribute;
use melior::ir::TypeLike;
use melior::{
    ir::{
        attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute},
        operation::OperationBuilder,
        r#type::MemRefType,
        Location, Operation, Type, Value, ValueLike,
    },
    Context,
};

pub fn static_reinterpret_cast<'c>(
    mlir_ctx: &'c Context,
    ty: Type<'c>,
    ptr: Value<'c, '_>,
    static_indices: &[i64],
    indices: &[Value<'c, '_>],
    size: usize,
    location: Location<'c>,
) -> Operation<'c> {
    let dim = vec![1];
    let len = static_indices.len() + indices.len();
    let layout = if static_indices.len() == 1 {
        Some(StridedLayoutAttribute::new(mlir_ctx, static_indices[0] as _, &[len as _]).into())
    } else {
        None
    };
    let result_ty = MemRefType::new(ty, &dim, layout, None);

    assert!(static_indices.is_empty() || indices.is_empty());
    let static_sizes = DenseI64ArrayAttribute::new(mlir_ctx, &[size as _]).into();
    let static_strides = DenseI64ArrayAttribute::new(mlir_ctx, &[1]).into();
    //log::trace!("indices = {:?}", indices);
    let mut op = OperationBuilder::new("memref.reinterpret_cast", location)
        .add_operands(&[ptr])
        //.add_operands(indices)
        .add_results(&[result_ty.into()])
        .build()
        .expect("valid operation");
    if !static_indices.is_empty() {
        op.set_attribute(
            "static_offsets",
            DenseI64ArrayAttribute::new(mlir_ctx, static_indices).into(),
        );
    }
    op.set_attribute("static_sizes", static_sizes);
    op.set_attribute("static_strides", static_strides);
    op.set_attribute(
        "operand_segment_sizes",
        DenseI32ArrayAttribute::new(
            mlir_ctx,
            &[1, if !indices.is_empty() { 1 } else { 0 }, 0, 0],
        )
        .into(),
    );
    op
}

pub fn cast<'c>(
    mlir_ctx: &'c Context,
    val: Value<'c, '_>,
    dest_ty: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    melior::dialect::ods::memref::cast(mlir_ctx, dest_ty, val, location).into()
}

pub fn extract_strided_metadata<'c>(
    mlir_ctx: &'c Context,
    val: Value<'c, '_>,
    location: Location<'c>,
) -> melior::dialect::ods::memref::ExtractStridedMetadataOperation<'c> {
    assert!(val.r#type().is_mem_ref());
    melior::dialect::ods::memref::extract_strided_metadata(mlir_ctx, val, location)
}
