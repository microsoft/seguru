use core::mem::MaybeUninit;

use super::unsafe_bindings::*;
#[cfg(feature = "gpu")]
use crate::eprintln;
use crate::sized_or_slice::{SizedOrSlice, SizedOrSliceClone};
use crate::{CUDA_SUCCESS, CudaError, GpuCtxGuard, GpuCtxSpace, TensorView};

// This is similar to Box, but the memory is allocated using cudaHostAlloc
// The memory is page-locked and can be accessed by the GPU directly.
// It is accessible from any GPU context.
// No Copy/Clone
pub struct PinnedHostBox<'a, T: ?Sized> {
    ptr: &'a mut T,
}

#[expect(private_bounds)]
impl<'a, T: ?Sized + SizedOrSlice> PinnedHostBox<'a, T> {
    #[doc(hidden)]
    /// Create a new PinnedHostBox with uninitialized memory
    /// # Safety
    /// The memory is uninitialized and must be initialized before use.
    unsafe fn maybe_uninit<'ctx, NS: GpuCtxSpace>(
        _ctx: &GpuCtxGuard<'ctx, '_, NS>,
        len: usize,
    ) -> Result<Self, CudaError> {
        let align: usize = core::mem::align_of::<T::UnitType>();
        if 0x4000 % align != 0 {
            return Err(CudaError::MemAlignmentTooHigh(0x4000, align));
        }
        let size = core::mem::size_of::<T::UnitType>() * len;
        let mut raw = MaybeUninit::<*mut u8>::uninit();
        let ret = unsafe { cuMemHostAlloc(raw.as_mut_ptr() as _, size, 0) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Err(ret));
        }
        let raw = unsafe { raw.assume_init() };
        let ptr: &mut T = unsafe { &mut *(T::build_const_ptr(raw, len) as *mut T) };
        Ok(PinnedHostBox { ptr })
    }

    /// Create a new PinnedHostBox with data copied from the given tensor
    #[expect(private_bounds)]
    pub fn new_from_tensor<'ctx: 'b, 'b: 'c, 'c, NS: GpuCtxSpace>(
        ctx: &GpuCtxGuard<'ctx, 'b, NS>,
        from_tensor: &TensorView<'c, T>,
    ) -> Result<Self, CudaError>
    where
        T: SizedOrSliceClone,
    {
        // This is safe to use since we initized the memory right after allocation
        let ret = unsafe { Self::maybe_uninit(ctx, from_tensor.len())? };
        from_tensor.copy_to_host(ret.ptr)?;
        Ok(ret)
    }
}

// It is CPU accessible from any GPU context and so we can deref it directly to use it
impl<'a, T: ?Sized> core::ops::Deref for PinnedHostBox<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

// It is CPU accessible from any GPU context and so we can deref it directly to use it
impl<'a, T: ?Sized> core::ops::DerefMut for PinnedHostBox<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr
    }
}

impl<'a, T: ?Sized> Drop for PinnedHostBox<'a, T> {
    fn drop(&mut self) {
        let ret = unsafe { cuMemFreeHost(self.ptr as *const T as _) };
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to free pinned host memory: {}", CudaError::Err(ret));
        }
    }
}
