use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::params::AsKernelParamsGuard;
use crate::{AsHostKernelParams, CudaMemBox};

macro_rules! impl_chunkable_as_kernel_params {
    ($($t: tt)*) => {
        $($t)*
        _impl_chunkable_as_kernel_params!($($t)*);
    };
}
macro_rules! _impl_chunkable_as_kernel_params {
    ($(#[$meta:meta])*
        $vis:vis struct $name:ident<$l: lifetime, T> {
            $slice:ident : *const $first_ty:ty,
            $extra:ident : $type:ty,
            $(
                $field:ident : PhantomData<$ty:ty>,
            ),* $(,)?
        }) => {
        impl<$l, T> AsKernelParamsGuard for $name<$l, T> {}
        impl<$l, T> AsHostKernelParams for $name<$l, T> {
            fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
                vec![
                    Box::new(self.$slice as *const T as usize),
                    Box::new(self.$slice.len()),
                    Box::new(self.$extra),
                ]
            }
        }

        impl<$l, T> $name<$l, T> {
            #[inline(always)]
            #[gpu_codegen::device]
            pub fn $extra(&self) -> usize {
                self.$extra
            }

            #[inline(always)]
            #[gpu_codegen::device]
            pub fn as_ptr(&self) -> *const [T] {
                self.$slice
            }
        }
    };
    () => {};
}

impl_chunkable_as_kernel_params! {
#[repr(C)]
pub struct GpuChunkableMut<'a, T> {
    slice: *const [T],
    window: usize,
    dummy: PhantomData<&'a mut T>,
}
}

impl_chunkable_as_kernel_params! {
#[repr(C)]
pub struct GpuChunkable<'a, T> {
    slice: *const [T],
    window: usize,
    dummy: PhantomData<&'a T>,
}
}

impl_chunkable_as_kernel_params! {
#[repr(C)]
pub struct GpuChunkableMut2D<'a, T> {
    slice: *const [T],
    size_x: usize,
    dummy: PhantomData<&'a mut T>,
}
}

impl_chunkable_as_kernel_params! {
#[repr(C)]
pub struct GpuChunkable2D<'a, T> {
    slice: *const [T],
    size_x: usize,
    dummy: PhantomData<&'a T>,
}
}

impl<'a, T> GpuChunkable<'a, T> {
    pub fn new(slice: &'a CudaMemBox<[T]>, window: usize) -> Self {
        GpuChunkable { slice: slice.as_ptr(), window, dummy: PhantomData }
    }

    /// # Safety
    /// This is not safe from Host side.
    #[inline(always)]
    #[gpu_codegen::device]
    pub unsafe fn new_from_gpu(slice: &'a [T], window: usize) -> Self {
        GpuChunkable { slice, window, dummy: PhantomData }
    }
}

impl<'a, T> GpuChunkableMut<'a, T> {
    pub fn new(slice: &'a mut CudaMemBox<[T]>, window: usize) -> GpuChunkableMut<'a, T> {
        GpuChunkableMut { slice: slice.as_ptr() as _, window, dummy: PhantomData }
    }

    /// # Safety
    /// This is not safe from Host side.
    #[inline(always)]
    #[gpu_codegen::device]
    pub unsafe fn new_from_gpu(slice: &'a mut [T], window: usize) -> Self {
        GpuChunkableMut { slice, window, dummy: PhantomData }
    }
}

impl<'a, T> GpuChunkableMut2D<'a, T> {
    pub fn new(slice: &'a mut CudaMemBox<[T]>, size_x: usize) -> Self {
        if slice.as_ptr().len() % size_x != 0 || slice.as_ptr().len() == 0 {
            // We're fucked
            panic!("slice is not aligned with the sizes provided");
        }

        GpuChunkableMut2D { slice: slice.as_ptr() as _, size_x, dummy: PhantomData }
    }

    /// # Safety
    /// This is not safe from Host side.
    #[inline(always)]
    #[gpu_codegen::device]
    pub unsafe fn new_from_gpu(slice: &'a mut [T], size_x: usize) -> Self {
        GpuChunkableMut2D { slice, size_x, dummy: PhantomData }
    }
}

impl<'a, T> GpuChunkable2D<'a, T> {
    pub fn new(slice: &'a CudaMemBox<[T]>, size_x: usize) -> Self {
        if slice.as_ptr().len() % size_x != 0 || slice.as_ptr().len() == 0 {
            // We're fucked
            panic!("slice is not aligned with the sizes provided");
        }

        GpuChunkable2D { slice: slice.as_ptr(), size_x, dummy: PhantomData }
    }

    /// # Safety
    /// This is not safe from Host side.
    #[inline(always)]
    #[gpu_codegen::device]
    pub unsafe fn new_from_gpu(slice: &'a [T], size_x: usize) -> Self {
        GpuChunkable2D { slice, size_x, dummy: PhantomData }
    }
}
