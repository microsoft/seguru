use melior::Context;
use melior::dialect::ods::memref as raw_memref;
use melior::ir::attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute};
use melior::ir::r#type::MemRefType;
use melior::ir::{Location, Operation, Type, TypeLike, Value, ValueLike};

use crate::mlir::attr::StridedLayoutAttribute;

/// A negative value indicating that the stride or offset is dynamic.
pub fn dynamic_stride_offset() -> i64 {
    unsafe { mlir_sys::mlirShapedTypeGetDynamicStrideOrOffset() }
}

pub fn dynamic_size() -> i64 {
    unsafe { mlir_sys::mlirShapedTypeGetDynamicSize() }
}

/// Implement reinterpret_cast for MemRef types.
/// If the offset, size, or stride is dynamic, the operation must set static_offsets, static_sizes,
/// and static_strides to dynamic_stride_offset() and size = dynamic_size().
pub fn reinterpret_cast<'c>(
    mlir_ctx: &'c Context,
    ty: Type<'c>,
    ptr: Value<'c, '_>,
    static_indices: &[i64],
    indices: &[Value<'c, '_>],
    static_sizes: &[i64],
    sizes: &[Value<'c, '_>],
    static_strides: &[i64],
    strides: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    assert!(static_indices.is_empty() || indices.is_empty());
    assert!(static_sizes.is_empty() || sizes.is_empty());
    assert!(static_strides.is_empty() || strides.is_empty());
    let len = static_indices.len() + indices.len();
    assert!(len == 1);
    let dim = vec![1];
    // -1 indicates that the index is dynamic
    let static_indices = if static_indices.is_empty() {
        vec![dynamic_stride_offset(); indices.len()]
    } else {
        static_indices.to_vec()
    };
    let static_strides = if static_strides.is_empty() {
        vec![dynamic_stride_offset(); indices.len()]
    } else {
        static_strides.to_vec()
    };

    let static_sizes = if static_sizes.is_empty() {
        vec![dynamic_size(); indices.len()]
    } else {
        static_sizes.to_vec()
    };
    let layout =
        Some(StridedLayoutAttribute::new(mlir_ctx, static_indices[0] as _, &static_strides).into());
    let result_ty = MemRefType::new(ty, &dim, layout, None);

    let static_sizes = DenseI64ArrayAttribute::new(mlir_ctx, &static_sizes).into();
    let static_strides = DenseI64ArrayAttribute::new(mlir_ctx, &static_strides).into();
    let static_offsets = DenseI64ArrayAttribute::new(mlir_ctx, &static_indices).into();

    let mut op: Operation<'c> = raw_memref::reinterpret_cast(
        mlir_ctx,
        result_ty.into(),
        ptr,
        indices,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        location,
    )
    .into();
    op.set_attribute(
        "operand_segment_sizes",
        DenseI32ArrayAttribute::new(
            mlir_ctx,
            &[1, indices.len() as i32, sizes.len() as i32, strides.len() as i32],
        )
        .into(),
    );
    op
}

pub fn extract_strided_metadata<'c>(
    mlir_ctx: &'c Context,
    val: Value<'c, '_>,
    location: Location<'c>,
) -> melior::dialect::ods::memref::ExtractStridedMetadataOperation<'c> {
    assert!(val.r#type().is_mem_ref());
    melior::dialect::ods::memref::extract_strided_metadata(mlir_ctx, val, location)
}
