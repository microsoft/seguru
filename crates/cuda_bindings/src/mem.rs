use core::ffi::c_void;
use core::marker::PhantomData;

use super::unsafe_bindings::*;
use super::{CUDA_SUCCESS, CudaError};
use crate::ctx::{CtxSpaceZero, GpuCtxArenaTrait, GpuCtxGuard, GpuCtxSpace};

#[cfg(feature = "gpu")]
macro_rules! eprintln {
    ($($arg:tt)*) => {{}};
}

/// N is the namespace ID for the GPU context.
#[derive(Debug)]
pub struct CudaMemBox<T: ?Sized, N: GpuCtxSpace + 'static = CtxSpaceZero> {
    pub(crate) ptr: *mut T,
    _marker: PhantomData<N>,
}

pub(crate) trait GpuDataMarker: 'static {}

/// This is to ensure that the data can be safely transferred to the GPU device.
/// This helps to prevent accidental transfer of non-Send types (e.g., *const T)
/// or non-'static types (e.g., &'a T) to the GPU device. But user still need to
/// ensure that the data is valid for GPU usage. For example, a struct with
/// &'static T or &'static mut T is still Send when T is Send, but it may not
/// valid for GPU usage if we do not enable HMM. Thus, we statically ruled out
/// most risky types; If somehow the user accidentally passed &'static T to
/// cudaBox, the GPU should return CUDA_ERROR_ILLEGAL_ADDRESS error when
/// accessing the pointer.
impl<T: ?Sized + Send + 'static> GpuDataMarker for T {}

/// If the data is a CudaMemBox, it is safe to transfer to the GPU device.
/// Thus, we can store a CudaMemBox inside another CudaMemBox.
impl<T: ?Sized + Send + 'static, N: GpuCtxSpace> GpuDataMarker for CudaMemBox<T, N> {}

/// Prevent CudaMemBox from being sent to other threads, as the underlying
/// CUDA memory is tied to a specific GPU context which is not thread-safe.
impl<T: ?Sized, N: GpuCtxSpace + 'static> !Send for CudaMemBox<T, N> {}

impl<T: ?Sized + 'static, N: GpuCtxSpace + 'static> GpuCtxArenaTrait for CudaMemBox<T, N> {
    fn as_any(&mut self) -> &mut (dyn core::any::Any) {
        self
    }
}

pub(crate) unsafe fn gpu_memalloc(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr: *mut c_void = core::ptr::null_mut();
    let ret = unsafe { cuMemAlloc_v2(&mut ptr as *mut _ as _, size) };
    if ret != CUDA_SUCCESS {
        return Err(CudaError::Err(ret));
    }
    Ok(ptr)
}

impl<'ctx, 'a, N: GpuCtxSpace + 'static> GpuCtxGuard<'ctx, 'a, N> {
    unsafe fn __new_gmem<T: GpuDataMarker>(&self) -> Result<&'ctx mut CudaMemBox<T, N>, CudaError> {
        let size = core::mem::size_of::<T>();
        let ptr = unsafe { crate::mem::gpu_memalloc(size)? as _ };
        let m = CudaMemBox::<T, N> { ptr, _marker: PhantomData };
        Ok(self.ctx.alloc_typed(m))
    }

    #[allow(private_bounds)]
    pub fn new_gmem<T: GpuDataMarker>(
        &self,
        init: T,
    ) -> Result<&'ctx mut CudaMemBox<T, N>, CudaError> {
        let ret: &'ctx mut CudaMemBox<T, N> = unsafe { self.__new_gmem() }?;
        ret.copy_from_host(&init, self)?;
        Ok(ret)
    }

    unsafe fn __new_gmem_with_len<T: GpuDataMarker>(
        &self,
        len: usize,
    ) -> Result<&'ctx mut CudaMemBox<[T], N>, CudaError> {
        let size = core::mem::size_of::<T>() * len;
        let ptr = unsafe {
            let raw = gpu_memalloc(size)?;
            core::slice::from_raw_parts_mut(raw as _, len)
        };
        let m = CudaMemBox::<[T], N> { ptr, _marker: PhantomData };
        Ok(self.ctx.alloc_typed(m))
    }

    #[allow(private_bounds)]
    pub fn new_gmem_with_len<T: GpuDataMarker>(
        &self,
        len: usize,
        init: &[T],
    ) -> Result<&'ctx mut CudaMemBox<[T], N>, CudaError> {
        let ret: &'ctx mut CudaMemBox<[T], N> = unsafe { self.__new_gmem_with_len(len) }?;
        ret.copy_from_host(init, len, self)?;
        Ok(ret)
    }
}

impl<T: Sized, N: GpuCtxSpace> CudaMemBox<T, N> {
    pub fn copy_to_host<'ctx, 'a>(
        &self,
        dst: &mut T,
        _ctx: &GpuCtxGuard<'ctx, 'a, N>,
    ) -> Result<(), CudaError> {
        let size = core::mem::size_of::<T>();
        unsafe {
            cuMemcpyDtoH_v2(dst as *mut _ as _, self.as_devptr(), size);
        }
        Ok(())
    }

    pub fn copy_from_host<'ctx, 'a>(
        &self,
        src: &T,
        _ctx: &GpuCtxGuard<'ctx, 'a, N>,
    ) -> Result<(), CudaError> {
        let size = core::mem::size_of::<T>();
        unsafe {
            cuMemcpyHtoD_v2(self.as_devptr(), src as *const _ as _, size);
        }
        Ok(())
    }
}

impl<T, N: GpuCtxSpace> CudaMemBox<[T], N> {
    pub fn copy_to_host<'ctx, 'a>(
        &'ctx self,
        dst: &mut [T],
        len: usize,
        _ctx: &GpuCtxGuard<'ctx, 'a, N>,
    ) -> Result<(), CudaError> {
        if len > dst.len() || len > self.ptr.len() {
            return Err(CudaError::MemCopyOutOfBound);
        }
        let size = core::mem::size_of::<T>() * len;
        unsafe {
            cuMemcpyDtoH_v2(dst as *mut _ as _, self.as_devptr(), size);
        }
        Ok(())
    }

    pub fn copy_from_host<'ctx, 'a>(
        &'ctx self,
        src: &[T],
        len: usize,
        _ctx: &GpuCtxGuard<'ctx, 'a, N>,
    ) -> Result<(), CudaError> {
        if len > src.len() || len > self.ptr.len() {
            return Err(CudaError::MemCopyOutOfBound);
        }
        let size = core::mem::size_of::<T>() * len;
        unsafe {
            cuMemcpyHtoD_v2(self.as_devptr(), src as *const _ as _, size);
        }
        Ok(())
    }
}

impl<T: ?Sized, N: GpuCtxSpace> CudaMemBox<T, N> {
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_devptr(&self) -> CUdeviceptr {
        self.ptr as *const c_void as CUdeviceptr
    }
}

impl<T: ?Sized> core::ops::Deref for CudaMemBox<T> {
    type Target = T;

    /// This should never be used in host code, as it would panic.
    /// This is only here to check API correctness in GPU code.
    /// Since &CudaMemBox<T> is treated as &T in GPU code.
    fn deref(&self) -> &Self::Target {
        // SAFETY: we know `ptr` is valid and was allocated via cudaMalloc
        panic!("This should only be called in GPU code, not host code")
    }
}

impl<T: ?Sized> core::ops::DerefMut for CudaMemBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: we know `ptr` is valid and was allocated via cudaMalloc
        panic!("This should only be called in GPU code, not host code")
    }
}

impl<T: ?Sized, N: GpuCtxSpace> Drop for CudaMemBox<T, N> {
    fn drop(&mut self) {
        // Why push and pop current context?
        // Because we may have multiple contexts in the same CPU thread.
        // If we push ctx1, create cudamem1, and then push ctx2, cudamem2, and then push ctx1 and use cudamem1.
        // The cudamem1 and cudamem2 may be dropped in the same context, which may cause problems.
        // This also introduce 8bytes extra overhead in cudamem, to avoid that
        // we may need to maintain N -> ctx mapping?.
        let ret = unsafe { cuMemFree_v2(self.as_ptr() as *const u8 as _) };
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to free GPU memory: {} {:?}", CudaError::Err(ret), self.as_ptr());
        }
    }
}
