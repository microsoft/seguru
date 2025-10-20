#[expect(private_bounds)]
pub trait VecTypeTrait: Default + Copy + core::ops::Add<Output = Self> + VecTypeInternal {
    type InnerType: Copy
        + Default
        + core::ops::IndexMut<usize, Output = Self::Elem>
        + core::iter::IntoIterator<Item = Self::Elem>;
    type Elem: Copy + Default + core::ops::Add<Output = Self::Elem>;

    fn new(data: Self::InnerType) -> Self;

    fn data(&self) -> &Self::InnerType;

    fn iter_mut(&mut self) -> core::slice::IterMut<'_, Self::Elem>;

    fn iter(&self) -> core::slice::Iter<'_, Self::Elem>;
}

/// Internal trait to restrict VecType generic parameter.
trait VecTypeInternal {}

macro_rules! impl_floatn_from {
    ($name: ident, $inner: ident, $align: literal, $base: ty, $N: literal) => {
        #[derive(Clone, Copy, PartialEq, Debug, Default)]
        #[repr(C, align($align))]
        pub struct $inner {
            pub data: [$base; $N],
        }
        impl From<[$base; $N]> for $inner {
            #[inline(always)]
            #[gpu_codegen::device]
            fn from(v: [$base; $N]) -> Self {
                Self { data: v }
            }
        }
        pub type $name = VecType<$inner>;
        impl VecTypeTrait for $inner {
            type Elem = $base;
            type InnerType = [$base; $N];
            #[inline(always)]
            #[gpu_codegen::device]
            fn new(data: Self::InnerType) -> Self {
                Self { data }
            }

            #[inline(always)]
            #[gpu_codegen::device]
            fn data(&self) -> &Self::InnerType {
                &self.data
            }

            #[inline(always)]
            #[gpu_codegen::device]
            fn iter_mut(&mut self) -> core::slice::IterMut<'_, Self::Elem> {
                self.data.iter_mut()
            }

            #[inline(always)]
            #[gpu_codegen::device]
            fn iter(&self) -> core::slice::Iter<'_, Self::Elem> {
                self.data.iter()
            }
        }

        impl VecTypeInternal for $inner {}

        impl core::ops::Add for $inner {
            type Output = Self;
            /// Adds two vectors element-wise.
            #[inline(always)]
            #[gpu_codegen::device]
            fn add(self, rhs: Self) -> Self {
                let mut data: [$base; $N] = [Default::default(); $N];
                for i in 0..$N {
                    data[i] = self.data[i] + rhs.data[i];
                }
                $inner { data }
            }
        }

        impl core::ops::Index<usize> for $inner {
            type Output = $base;
            #[inline(always)]
            #[gpu_codegen::device]
            fn index(&self, index: usize) -> &Self::Output {
                &self.data[index]
            }
        }

        impl core::ops::IndexMut<usize> for $inner {
            #[inline(always)]
            #[gpu_codegen::device]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.data[index]
            }
        }
    };
}

/// A vector type with `N` elements of type `T::Elem`.
/// T must be aligned by the size of the vector.
/// Since we cannot do `repr(align(N * size_of::<T::Elem>))` yet,
/// we define separate types for different sizes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VecType<T: VecTypeTrait> {
    val: T,
}

impl<T: VecTypeTrait> Default for VecType<T> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn default() -> Self {
        Self { val: T::new(Default::default()) }
    }
}

impl<T: VecTypeTrait> VecType<T> {
    #[inline(always)]
    #[gpu_codegen::device]
    pub fn new(val: T::InnerType) -> Self {
        VecType { val: T::new(val) }
    }
}

impl<T: VecTypeTrait> core::ops::Add for VecType<T> {
    type Output = Self;
    /// Adds two vectors element-wise.
    #[inline(always)]
    #[gpu_codegen::device]
    fn add(self, rhs: Self) -> Self {
        let val = self.val + rhs.val;
        VecType { val }
    }
}

impl<T: VecTypeTrait> core::ops::Deref for VecType<T> {
    type Target = T;
    #[inline(always)]
    #[gpu_codegen::device]
    fn deref(&self) -> &T {
        &self.val
    }
}

impl<T: VecTypeTrait> core::ops::DerefMut for VecType<T> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.val
    }
}

impl_floatn_from!(Float2, Float2Inner, 8, f32, 2);
impl_floatn_from!(Float4, Float4Inner, 16, f32, 4);
impl_floatn_from!(Float8, Float8Inner, 32, f32, 8);

impl_floatn_from!(U32_2, U32_2Inner, 8, u32, 2);
impl_floatn_from!(U32_4, U32_4Inner, 16, u32, 4);
impl_floatn_from!(U32_8, U32_8Inner, 32, u32, 8);

pub trait VecFlatten<T2> {
    fn flatten(&self) -> &[T2];
}

/// Useful to optimize code with vector load/store.
/// If length of the slice is not a multiple of N,
/// the remaining elements will be ignored.
///
/// # Safety
/// This is safe since VecType<T> always has a layout \
/// compatible with T::Elem array.
impl<T, T2> VecFlatten<T2> for [VecType<T>]
where
    T: VecTypeTrait<Elem = T2>,
{
    fn flatten(&self) -> &[T2] {
        // SAFETY: the returned slice will be at same size or shorter, so it is safe.
        assert!(size_of::<T>() >= size_of::<T2>(), "T2 is larger than T");
        assert!(align_of::<T>() >= align_of::<T2>(), "T2 has stricter alignment than T");
        unsafe {
            &*core::ptr::slice_from_raw_parts_mut(
                self.as_ptr() as _,
                self.len() * size_of::<T>() / size_of::<T2>(),
            )
        }
    }
}
