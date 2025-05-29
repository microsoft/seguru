use melior::Context;
use melior::ir::{Attribute, AttributeLike};
macro_rules! attribute_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: mlir_sys::MlirAttribute) -> Self {
                Self { attribute: Attribute::from_raw(raw) }
            }
        }

        impl<'c> TryFrom<melior::ir::attribute::Attribute<'c>> for $name<'c> {
            type Error = melior::Error;

            fn try_from(
                attribute: melior::ir::attribute::Attribute<'c>,
            ) -> Result<Self, Self::Error> {
                if attribute.$is_type() {
                    Ok(unsafe { Self::from_raw(attribute.to_raw()) })
                } else {
                    Err(melior::Error::AttributeExpected($string, attribute.to_string()))
                }
            }
        }

        impl<'c> From<$name<'c>> for melior::ir::attribute::Attribute<'c> {
            fn from(attribute: $name<'c>) -> Self {
                attribute.attribute
            }
        }

        impl<'c> melior::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, formatter)
            }
        }
    };
}

pub struct StridedLayoutAttribute<'c> {
    pub attribute: Attribute<'c>,
}

attribute_traits!(StridedLayoutAttribute, is_strided_layout, "strided_layout");

impl<'c> StridedLayoutAttribute<'c> {
    pub fn new(mlir_ctx: &Context, offset: usize, strides: &[i64]) -> Self {
        let attribute = unsafe {
            Attribute::from_raw(mlir_sys::mlirStridedLayoutAttrGet(
                mlir_ctx.to_raw(),
                offset as _,
                strides.len() as _,
                strides.as_ptr() as _,
            ))
        };
        StridedLayoutAttribute { attribute }
    }

    pub fn get_offset(&self) -> i64 {
        unsafe { mlir_sys::mlirStridedLayoutAttrGetOffset(self.attribute.to_raw()) }
    }

    pub fn get_stride(&self, i: usize) -> i64 {
        unsafe { mlir_sys::mlirStridedLayoutAttrGetStride(self.attribute.to_raw(), i as _) }
    }

    pub fn get_stride_num(&self, i: usize) -> isize {
        unsafe { mlir_sys::mlirStridedLayoutAttrGetNumStrides(self.attribute.to_raw()) }
    }

    pub fn get_strides(&self) -> Vec<i64> {
        let num_strides = self.get_stride_num(0) as usize;
        let mut strides = vec![];
        for i in 0..num_strides {
            strides.push(self.get_stride(i));
        }
        strides
    }
}
