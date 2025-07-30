use core::ffi::c_void;
use core::marker::PhantomData;

use super::unsafe_bindings::*;
use super::{CUDA_SUCCESS, CudaError};
use crate::ctx::{CtxSpaceZero, GpuCtxArenaTrait, GpuCtxGuard, GpuCtxSpace};

macro_rules! eprintln {
    () => {
        panic!()
    };
    ($($arg:tt)*) => {{
        panic!($($arg)*);
    }};
}

/// N is the namespace ID for the GPU context.
#[derive(Debug)]
pub struct CudaMemBox<T: ?Sized, N: GpuCtxSpace + 'static = CtxSpaceZero> {
    pub(crate) ptr: *mut T,
    _marker: PhantomData<N>,
}

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
    pub fn new_gmem<T: 'static>(&self) -> Result<&'ctx mut CudaMemBox<T, N>, CudaError> {
        let size = core::mem::size_of::<T>();
        let ptr = unsafe { crate::mem::gpu_memalloc(size)? as *mut T };
        let m = CudaMemBox::<T, N> { ptr, _marker: PhantomData };
        Ok(self.ctx.alloc_typed(m))
    }

    pub fn new_gmem_with_len<T: 'static>(
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
