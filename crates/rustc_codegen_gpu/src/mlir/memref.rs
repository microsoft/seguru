use melior::Context;
use melior::dialect::ods::memref as raw_memref;
use melior::ir::attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute};
use melior::ir::operation::OperationMutLike;
use melior::ir::r#type::{IntegerType, MemRefType};
use melior::ir::{Attribute, Location, Operation, Type, TypeLike, Value, ValueLike};

use crate::mlir::attr::StridedLayoutAttribute;
use crate::mlir::mlir_val_to_const_int;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum MemorySpace {
    Global = 0,
    Shared = 3,
}

impl MemorySpace {
    pub(crate) fn to_attr<'ml>(self, ctx: &'ml Context) -> Attribute<'ml> {
        match self {
            MemorySpace::Shared => Attribute::parse(ctx, "#gpu.address_space<workgroup>").unwrap(),
            _ => IntegerAttribute::new(Type::from(IntegerType::new(ctx, 64)), self as i64).into(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum StaticOrDynamic<'ml, 'a> {
    Static(i64),
    Dynamic(Value<'ml, 'a>),
}

impl<'ml, 'a> From<Value<'ml, 'a>> for StaticOrDynamic<'ml, 'a> {
    fn from(value: Value<'ml, 'a>) -> Self {
        if let Some(static_val) = mlir_val_to_const_int(value) {
            StaticOrDynamic::Static(static_val as _)
        } else {
            StaticOrDynamic::Dynamic(value)
        }
    }
}

impl<'ml, 'a> From<i64> for StaticOrDynamic<'ml, 'a> {
    fn from(value: i64) -> Self {
        StaticOrDynamic::Static(value)
    }
}

impl<'ml, 'a> StaticOrDynamic<'ml, 'a> {
    fn to_dynamic(&self) -> Option<Value<'ml, 'a>> {
        match self {
            StaticOrDynamic::Dynamic(v) => Some(*v),
            StaticOrDynamic::Static(_) => None,
        }
    }

    fn to_static_size(&self) -> i64 {
        match self {
            StaticOrDynamic::Static(v) => *v,
            StaticOrDynamic::Dynamic(_) => dynamic_size(),
        }
    }

    fn to_static_stride_offset(&self) -> i64 {
        match self {
            StaticOrDynamic::Static(v) => *v,
            StaticOrDynamic::Dynamic(_) => dynamic_stride_offset(),
        }
    }

    fn to_dynamic_vec(arr: &[Self]) -> Vec<Value<'ml, 'a>> {
        arr.iter().filter_map(|x| x.to_dynamic()).collect()
    }

    fn to_static_stride_or_offset_vec(arr: &[Self]) -> Vec<i64> {
        arr.iter().map(|x| x.to_static_stride_offset()).collect()
    }

    fn to_static_size_vec(arr: &[Self]) -> Vec<i64> {
        arr.iter().map(|x| x.to_static_size()).collect()
    }

    fn to_static_stride_or_offset_attr(ctx: &'ml Context, arr: &[Self]) -> Attribute<'ml> {
        DenseI64ArrayAttribute::new(ctx, &Self::to_static_stride_or_offset_vec(arr)).into()
    }

    fn to_static_stride_and_offset_attr(
        ctx: &'ml Context,
        offset: &Self,
        strides: &[Self],
    ) -> Attribute<'ml> {
        StridedLayoutAttribute::new(
            ctx,
            offset.to_static_stride_offset() as usize,
            &Self::to_static_stride_or_offset_vec(strides),
        )
        .into()
    }

    fn to_static_size_attr(ctx: &'ml Context, arr: &[Self]) -> Attribute<'ml> {
        DenseI64ArrayAttribute::new(ctx, &Self::to_static_size_vec(arr)).into()
    }
}
/// A negative value indicating that the stride or offset is dynamic.
pub fn dynamic_stride_offset() -> i64 {
    unsafe { mlir_sys::mlirShapedTypeGetDynamicStrideOrOffset() }
}

pub fn dynamic_size() -> i64 {
    unsafe { mlir_sys::mlirShapedTypeGetDynamicSize() }
}

/// Implement reinterpret_cast for MemRef types.
pub fn reinterpret_cast<'c, 'a>(
    mlir_ctx: &'c Context,
    basety: Type<'c>,
    ptr: Value<'c, '_>,
    offsets: &[StaticOrDynamic<'c, 'a>],
    sizes: &[StaticOrDynamic<'c, 'a>],
    strides: &[StaticOrDynamic<'c, 'a>],
    location: Location<'c>,
) -> Operation<'c> {
    subview_or_reinterpret_cast(mlir_ctx, basety, ptr, offsets, sizes, strides, location, true)
}

pub fn subview<'c, 'a>(
    mlir_ctx: &'c Context,
    basety: Type<'c>,
    ptr: Value<'c, '_>,
    offsets: &[StaticOrDynamic<'c, 'a>],
    sizes: &[StaticOrDynamic<'c, 'a>],
    strides: &[StaticOrDynamic<'c, 'a>],
    location: Location<'c>,
) -> Operation<'c> {
    subview_or_reinterpret_cast(mlir_ctx, basety, ptr, offsets, sizes, strides, location, false)
}

#[allow(clippy::too_many_arguments)]
fn subview_or_reinterpret_cast<'c, 'a>(
    mlir_ctx: &'c Context,
    basety: Type<'c>,
    ptr: Value<'c, '_>,
    offsets: &[StaticOrDynamic<'c, 'a>],
    sizes: &[StaticOrDynamic<'c, 'a>],
    strides: &[StaticOrDynamic<'c, 'a>],
    location: Location<'c>,
    use_reinterpret: bool,
) -> Operation<'c> {
    //assert!(offsets.len() == 1);
    //assert!(strides.len() == 1);
    //assert!(sizes.len() == 1);
    let source_ty = ptr.r#type();
    let source_memref_ty: MemRefType<'_> =
        source_ty.try_into().expect("expected memref type for ptr");
    let dst_layout_offset = if let Ok(layout) =
        crate::mlir::attr::StridedLayoutAttribute::try_from(source_memref_ty.layout())
    {
        match offsets[0] {
            StaticOrDynamic::Static(v) => {
                if layout.get_offset() == dynamic_stride_offset() {
                    &layout.get_offset().into()
                } else {
                    &(layout.get_offset() + v).into()
                }
            }
            StaticOrDynamic::Dynamic(_) => &dynamic_stride_offset().into(),
        }
    } else {
        &offsets[0]
    };
    let static_offsets = StaticOrDynamic::to_static_stride_or_offset_attr(mlir_ctx, offsets);
    let dynamic_offsets = StaticOrDynamic::to_dynamic_vec(offsets);
    let static_strides = StaticOrDynamic::to_static_stride_or_offset_attr(mlir_ctx, strides);
    let dynamic_strides = StaticOrDynamic::to_dynamic_vec(strides);
    let static_sizes = StaticOrDynamic::to_static_size_attr(mlir_ctx, sizes);
    let dynamic_sizes = StaticOrDynamic::to_dynamic_vec(sizes);

    // Construct the result type.
    let mut dim = vec![];
    let mut dim_skip = true;
    if sizes.len() > 1 {
        for size in sizes {
            if size.to_static_size() != 1 || !dim_skip {
                dim.push(size.to_static_size());
                dim_skip = false;
            }
        }
    } else {
        dim.push(sizes[0].to_static_size());
    }
    let layout = Some(StaticOrDynamic::to_static_stride_and_offset_attr(
        mlir_ctx,
        dst_layout_offset,
        &strides[0..dim.len()],
    ));

    let result_ty =
        crate::mlir::type_memref(mlir_ctx, basety, &dim, layout, source_memref_ty.memory_space())
            .into();
    let mut op: Operation<'c> = if use_reinterpret {
        raw_memref::reinterpret_cast(
            mlir_ctx,
            result_ty,
            ptr,
            &dynamic_offsets,
            &dynamic_sizes,
            &dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            location,
        )
        .into()
    } else {
        raw_memref::subview(
            mlir_ctx,
            result_ty,
            ptr,
            &dynamic_offsets,
            &dynamic_sizes,
            &dynamic_strides,
            static_offsets,
            static_sizes,
            static_strides,
            location,
        )
        .into()
    };
    op.set_attribute(
        "operand_segment_sizes",
        DenseI32ArrayAttribute::new(
            mlir_ctx,
            &[
                1, // Must set arg1 ptr
                dynamic_offsets.len() as i32,
                dynamic_sizes.len() as i32,
                dynamic_strides.len() as i32,
            ],
        )
        .into(),
    );
    op
}

pub(crate) fn extract_strided_metadata<'c>(
    mlir_ctx: &'c Context,
    val: Value<'c, '_>,
    location: Location<'c>,
) -> melior::dialect::ods::memref::ExtractStridedMetadataOperation<'c> {
    assert!(val.r#type().is_mem_ref());
    melior::dialect::ods::memref::extract_strided_metadata(mlir_ctx, val, location)
}

pub(crate) struct StridedMetaDataResults<'ml: 'a, 'a> {
    pub base_memref: Value<'ml, 'a>,
    pub byte_offset: Value<'ml, 'a>,
    pub sizes: Vec<StaticOrDynamic<'ml, 'a>>,
    pub strides: Vec<StaticOrDynamic<'ml, 'a>>,
}

pub(crate) fn extract_strided_metadata_results<'ml: 'a, 'a>(
    memref_ty: MemRefType<'ml>,
    op: melior::ir::OperationRef<'ml, 'a>,
) -> StridedMetaDataResults<'ml, 'a> {
    let layout = crate::mlir::attr::StridedLayoutAttribute::try_from(memref_ty.layout())
        .expect("expect strided layout attribute");
    let get_metadata = |i: usize| Value::<'ml, 'a>::from(op.result(i).unwrap());
    let base_memref: Value<'ml, 'a> = get_metadata(0); // this is memref<i8> and need to be casted to memref<1xi8>
    let byte_offset: Value<'ml, 'a> = get_metadata(1);
    let mut meta_data_index = 2;
    let mut sizes: Vec<StaticOrDynamic<'ml, '_>> = vec![];
    use melior::ir::ShapedTypeLike;
    let rank = memref_ty.rank();
    for i in 0..rank {
        let s = memref_ty.dim_size(i).unwrap() as i64;
        if s == crate::mlir::memref::dynamic_size() {
            sizes.push(get_metadata(meta_data_index).into());
            meta_data_index += 1;
        } else {
            sizes.push(s.into());
        }
    }
    let mut strides: Vec<StaticOrDynamic<'ml, 'a>> = vec![];
    layout.get_strides().iter().for_each(|s| {
        let val = *s;
        if val == crate::mlir::memref::dynamic_stride_offset() {
            strides.push(get_metadata(meta_data_index).into());
            meta_data_index += 1;
        } else {
            strides.push(val.into());
        }
    });
    StridedMetaDataResults { base_memref, byte_offset, sizes, strides }
}
