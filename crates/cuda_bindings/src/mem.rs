use core::ffi::c_void;
use core::marker::PhantomData;

use super::unsafe_bindings::*;
use super::{CUDA_SUCCESS, CudaError};
use crate::ctx::{CtxSpaceZero, GpuCtxArenaTrait, GpuCtxGuard, GpuCtxSpace};
use crate::sized_or_slice::{SizedOrSlice, SizedOrSliceClone};

/// Marker trait for GPU vector types that can be flattened to their scalar element type.
/// Implementors guarantee proper alignment and size relationships.
///
/// # Safety
/// - `size_of::<Self>()` must equal `VEC_LEN * size_of::<Elem>()`
/// - `align_of::<Self>()` must be >= `align_of::<Elem>()`
pub unsafe trait DeviceVecType: Copy {
    type Elem: Copy;
    const VEC_LEN: usize;
}

#[cfg(feature = "gpu")]
#[macro_export]
macro_rules! eprintln {
    ($($arg:tt)*) => {{}};
}

/// No Copy/Clone
/// N is the namespace ID for the GPU context.
#[derive(Debug)]
pub struct CudaMemBox<T: ?Sized, N: GpuCtxSpace = CtxSpaceZero> {
    pub(crate) ptr: *mut T,
    _marker: PhantomData<N>,
}

impl<T: ?Sized, N: GpuCtxSpace> Drop for CudaMemBox<T, N> {
    fn drop(&mut self) {
        // Why push and pop current context?
        // Because we may have multiple contexts in the same CPU thread.
        // If we push ctx1, create cudamem1, and then push ctx2, cudamem2, and then push ctx1 and use cudamem1.
        // The cudamem1 and cudamem2 may be dropped in the same context, which may cause problems.
        // This also introduce 8bytes extra overhead in cudamem, to avoid that
        // we may need to maintain N -> ctx mapping?.
        let dev_addr = self.ptr as *const u8 as _;
        let ret = unsafe { cuMemFree_v2(dev_addr) };
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to free GPU memory: {} {:?}", CudaError::Err(ret), dev_addr);
        }
    }
}

/// This struct is used to represent a tensor allocated on the GPU for a given context.
/// The lifetime of the tensor is tied to the lifetime of the GPU context it was created from.
/// Even when the current context switches to another context, the tensor's data must remain valid.
/// It only got dropped when the original context is dropped.
/// Since user may create multiple GPU contexts, the use of tensor must be tied to ctx
/// and user should treat it as TensorView/TensorViewMut to avoid passing ctx around.
pub type TensorMut<'ctx, T, N> = &'ctx mut CudaMemBox<T, N>;
pub type TensorRef<'ctx, T, N> = &'ctx CudaMemBox<T, N>;

/// TensorView represents an immutable view of a tensor allocated on the GPU.
/// It is tied to the lifetime of the GPU context guard for current context.
/// When the current context switches to another context, the TensorView's lifetime ends.
/// Since it is bounded to current context, we do not need to carry GpuCtxSpace here.
///
/// The lifetime of TensorView cannot outlive the lifetime of the GPU context.
/// index cannot outlive the lifetime of the parent TensorView.
/// ```rust,compile_fail,E0597
/// use cuda_bindings::*;
/// fn test_tensor_view<'ctx, N: GpuCtxSpace>(t: TensorRef<'ctx, [f32], N>, ctx: &GpuCtxGuard<'ctx, '_, N>) {
///     // `t` does not live long enough
///     let t0 = {
///         let t = t.as_tensor_view(ctx);
///         let t0 = t.index(0..4);
///         t0
///     };
///     let t1 = t0.index(1);
/// }
/// ```
/// No Copy/Clone
#[derive(Debug)]
pub struct TensorView<'a, T: ?Sized> {
    pub(crate) devptr: *const T,
    _marker: PhantomData<&'a ()>,
}

/// TensorViewMut represents a mutable view of a tensor allocated on the GPU.
/// It is tied to the lifetime of the GPU context guard it was created from.
/// When the current context switches to another context, the TensorViewMut's lifetime ends.
/// Since it is bounded to current context, we do not need to carry GpuCtxSpace here.
///
/// The lifetime of TensorViewMut cannot outlive the lifetime of the GPU context.
/// index_mut cannot outlive the lifetime of the parent TensorViewMut.
/// ```rust,compile_fail,E0499
/// use cuda_bindings::*;
/// fn test_tensor_view_mut<'ctx, N: GpuCtxSpace>(t: TensorMut<'ctx, [f32], N>, ctx: &GpuCtxGuard<'ctx, '_, N>) {
///     let mut t = t.as_tensor_view_mut(ctx);
///     let mut t0 = t.index_mut(0..4);
///     let mut t1 = t0.index_mut(0..2);
///     let t2 = t.index_mut(0);
///     let t3 = t1.index_mut(0);
/// }
/// ```
/// ```rust,compile_fail,E0499
/// use cuda_bindings::*;
/// fn f(x: &mut TensorViewMut<'_, [f32]>) {}
/// fn test_tensor_view_mut_index<'a, N: GpuCtxSpace>(t: &mut TensorViewMut<'a, [f32], N>) {
///     let mut t0 = t.index_mut(0..4);
///     let mut t1 = t.index_mut(0..1);
///     f(&mut t1);
///     f(&mut t0);
/// }
/// ```
/// No Copy/Clone
#[derive(Debug)]
pub struct TensorViewMut<'a, T: ?Sized> {
    pub(crate) inner: TensorView<'a, T>,
}

impl<'ctx, T: ?Sized, N: GpuCtxSpace> CudaMemBox<T, N> {
    pub fn as_tensor_view<'a>(&'ctx self, _ctx: &GpuCtxGuard<'ctx, 'a, N>) -> TensorView<'a, T> {
        TensorView { devptr: self.ptr, _marker: PhantomData }
    }

    pub fn as_tensor_view_mut<'a>(
        &'ctx mut self,
        ctx: &GpuCtxGuard<'ctx, 'a, N>,
    ) -> TensorViewMut<'a, T> {
        TensorViewMut { inner: self.as_tensor_view(ctx) }
    }
}

pub(crate) trait GpuDataMarker: 'static {}

/// This is to ensure that the data can be safely transferred to the GPU device.
/// This helps to prevent accidental transfer of non-Sync types (e.g., *const T)
/// or non-'static types (e.g., &'a T) to the GPU device. But user still need to
/// ensure that the data is valid for GPU usage. For example, a struct with
/// &'static T or &'static mut T is still Sync when T is Sync, but it may not
/// valid for GPU usage if we do not enable HMM. Thus, we statically ruled out
/// most risky types; If somehow the user accidentally passed &'static T to
/// CudaMemBox, the GPU should return CUDA_ERROR_ILLEGAL_ADDRESS error when
/// accessing the pointer.
impl<T: Sync + 'static> GpuDataMarker for T {}

impl<T: Sync + 'static> GpuDataMarker for [T] {}

/// If the data is a CudaMemBox, it is safe to transfer to the GPU device.
/// Thus, we can store a CudaMemBox inside another CudaMemBox.
impl<T: ?Sized + Sync + 'static, N: GpuCtxSpace> GpuDataMarker for CudaMemBox<T, N> {}

/// Prevent CudaMemBox from being sent to other threads, as the underlying
/// CUDA memory is tied to a specific GPU context which is not thread-safe.
impl<T: ?Sized, N: GpuCtxSpace + 'static> !Sync for CudaMemBox<T, N> {}

impl<T: ?Sized + 'static, N: GpuCtxSpace + 'static> GpuCtxArenaTrait for CudaMemBox<T, N> {
    fn as_any(&mut self) -> &mut (dyn core::any::Any) {
        self
    }
}

impl<'a, T: ?Sized> TensorView<'a, T> {
    #[inline(always)]
    pub fn as_devptr(&self) -> CUdeviceptr {
        self.devptr as *const c_void as CUdeviceptr
    }

    #[inline(always)]
    pub fn as_flat_devptr(&self) -> *const T {
        self.devptr
    }
}

impl<'a, T: Default + Clone + core::fmt::Debug + 'static> core::fmt::Display
    for TensorView<'a, [T]>
{
    fn fmt(&self, f: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        let max_print_len = core::cmp::min(100, self.len());
        let mut host_data = alloc::vec![T::default(); max_print_len];
        self.index(..max_print_len).copy_to_host(&mut host_data).expect("copy to host failed");
        write!(f, "TensorView {{ devptr: {:?} }}: {:?}", self.as_devptr(), host_data)?;
        Ok(())
    }
}

impl<'a, T: Default + Clone + core::fmt::Debug + 'static> core::fmt::Display
    for TensorViewMut<'a, [T]>
{
    fn fmt(&self, f: &mut alloc::fmt::Formatter<'_>) -> alloc::fmt::Result {
        (self as &TensorView<'a, [T]>).fmt(f)
    }
}

#[expect(private_bounds)]
impl<'a, T: ?Sized + SizedOrSlice> TensorView<'a, T> {
    pub fn index<'b, I>(&'b self, index: I) -> TensorView<'b, <T as core::ops::Index<I>>::Output>
    where
        T: core::ops::Index<I>,
    {
        let dev_ptr = unsafe { &(&(*self.devptr))[index] };
        TensorView { devptr: dev_ptr as *const _ as *mut _, _marker: PhantomData }
    }

    /// Return the length of the tensor if it is a slice, otherwise return None.
    pub(crate) fn try_get_slice_len(&self) -> Option<usize> {
        unsafe { (*self.devptr).len_if_slice() }
    }

    /// Return the number of base elements in the tensor.
    /// If the tensor is not a slice, return 1.
    /// If the tensor is a slice, return the length of the slice.
    pub fn len(&self) -> usize {
        self.try_get_slice_len().unwrap_or(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T: ?Sized> core::ops::Deref for TensorViewMut<'a, T> {
    type Target = TensorView<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[expect(private_bounds)]
impl<'a, T: ?Sized + SizedOrSlice> TensorViewMut<'a, T> {
    pub fn index_mut<I>(
        &mut self,
        index: I,
    ) -> TensorViewMut<'_, <T as core::ops::Index<I>>::Output>
    where
        T: core::ops::IndexMut<I>,
    {
        TensorViewMut { inner: self.inner.index(index) }
    }
}

impl<'a, T> TensorView<'a, [T]> {
    /// Split the tensor into two at the given index.
    /// The returned tensors will have the same lifetime as the original tensor.
    /// Since split_at allows multiple immutable borrows, user does not need to split.
    /// Not useful outside but as a helper for TensorViewMut.
    fn split(self, mid: usize) -> (TensorView<'a, [T]>, TensorView<'a, [T]>) {
        let (left, right) = unsafe { (*self.devptr).split_at(mid) };
        (
            TensorView { devptr: left as _, _marker: PhantomData },
            TensorView { devptr: right as _, _marker: PhantomData },
        )
    }

    /// Split the tensor into two at the given index.
    /// The returned tensors will have the same lifetime as the ref of the
    /// tensor. which is live only for the scope of the borrow, and is usually
    /// shorter than the lifetime of the original tensor.
    pub fn split_at(&self, mid: usize) -> (TensorView<'_, [T]>, TensorView<'_, [T]>) {
        let (left, right) = unsafe { (*self.devptr).split_at(mid) };
        (
            TensorView { devptr: left as _, _marker: PhantomData },
            TensorView { devptr: right as _, _marker: PhantomData },
        )
    }

    /// Flatten this tensor view from a vector element type to its scalar element type.
    /// Only available when `T` implements `DeviceVecType` (e.g., `U32_4` → `u32`).
    pub fn flatten(&self) -> TensorView<'_, [T::Elem]>
    where
        T: DeviceVecType,
    {
        let old_len = self.devptr.len();
        let new_len = old_len * T::VEC_LEN;
        let ptr = self.devptr as *const T::Elem;
        let fat_ptr = core::ptr::slice_from_raw_parts(ptr, new_len);
        TensorView { devptr: fat_ptr, _marker: PhantomData }
    }
}

impl<'a, T> TensorViewMut<'a, [T]> {
    /// Split the tensor into two at the given index.
    /// The returned tensors will have the same lifetime as the lifetime of the
    /// reference, which usually be live only for the scope of the mutable
    /// borrow, and is usually shorter than the lifetime of the original tensor.
    /// Since method borrows self mutably, it prevents other mutable borrows.
    /// If in one scope, user want to get multiple mutable sub tensor views,
    /// user should use split() after split_mut_at to get more than two sub views.
    ///
    /// ```rust,compile_fail,E0515
    /// use cuda_bindings::*;
    /// fn test_tensor_view_mut_split<'a, 'b>(t: &'b mut TensorViewMut<'a, [f32]>) -> (TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>) {
    ///     let (mut t1, mut t2) = t.split_at_mut(2);
    ///     let (t3, t4) = t1.split_at_mut(1);
    ///     let (t5, t6) = t2.split_at_mut(1);
    ///     // error[E0515]: cannot return value referencing local variable `t2`
    ///     // error[E0515]: cannot return value referencing local variable `t1`
    ///     (t3, t4, t5, t6)
    /// }
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (TensorViewMut<'_, [T]>, TensorViewMut<'_, [T]>) {
        let (left, right) = self.inner.split_at(mid);
        (TensorViewMut { inner: left }, TensorViewMut { inner: right })
    }

    /// Split the tensor into two at the given index.
    /// The returned tensors will have the same lifetime as the original tensor.
    /// Guaranteed to be non-overlapping.
    ///
    /// ```rust
    /// use cuda_bindings::*;
    /// fn test_tensor_view_split<'a, 'b>(t: &'b mut TensorViewMut<'a, [f32]>) -> (TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>, TensorViewMut<'b, [f32]>) {
    ///     let (mut t1, mut t2) = t.split_at_mut(2);
    ///     let (t3, t4) = t1.split(1);
    ///     let (t5, t6) = t2.split(1);
    ///     (t3, t4, t5, t6)
    /// }
    /// ```
    pub fn split(self, mid: usize) -> (TensorViewMut<'a, [T]>, TensorViewMut<'a, [T]>) {
        let (left, right) = self.inner.split(mid);
        (TensorViewMut { inner: left }, TensorViewMut { inner: right })
    }

    /// Flatten this mutable tensor view from a vector element type to its scalar element type.
    /// Only available when `T` implements `DeviceVecType` (e.g., `U32_4` → `u32`).
    pub fn flatten(&mut self) -> TensorViewMut<'_, [T::Elem]>
    where
        T: DeviceVecType,
    {
        let old_len = self.inner.devptr.len();
        let new_len = old_len * T::VEC_LEN;
        let ptr = self.inner.devptr as *const T::Elem;
        let fat_ptr = core::ptr::slice_from_raw_parts(ptr, new_len);
        TensorViewMut { inner: TensorView { devptr: fat_ptr, _marker: PhantomData } }
    }
}

impl<'ctx: 'a, 'a, N: GpuCtxSpace + 'static> GpuCtxGuard<'ctx, 'a, N> {
    unsafe fn alloc_gmem_ignore_init<T: ?Sized + SizedOrSlice + 'static>(
        &self,
        init: &T,
    ) -> Result<TensorMut<'ctx, T, N>, CudaError> {
        let len = init.len_if_slice().unwrap_or(1);
        let size = core::mem::size_of::<T::UnitType>() * len;
        let align: usize = core::mem::align_of::<T::UnitType>();
        if 256 % align != 0 {
            return Err(CudaError::MemAlignmentTooHigh(256, align));
        }

        let mut raw: *mut c_void = core::ptr::null_mut();
        // This is safe since we have checked the error code to ensure the allocation is successful.
        let ret = unsafe { cuMemAlloc_v2(&mut raw as *mut _ as _, size) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Err(ret));
        }
        let ptr = T::build_const_ptr(raw as _, len) as _;
        let m = CudaMemBox { ptr, _marker: PhantomData };
        Ok(self.ctx.alloc_typed(m))
    }

    /// Allocate a tensor on the GPU and initialize it with the data from `init`.
    /// Return the tensor, which only lives as long as the GPU context.
    /// But the use of tensor must be tied to ctx and user may treat it as tensor_view
    /// to avoid passing ctx around.
    #[expect(private_bounds)]
    pub fn new_tensor<T: ?Sized + SizedOrSliceClone + 'static>(
        &self,
        init: &T,
    ) -> Result<TensorMut<'ctx, T, N>, CudaError> {
        if init.len_if_slice().unwrap_or(1) == 0 {
            return Ok(self.ctx.alloc_typed(CudaMemBox {
                ptr: T::build_const_ptr(core::ptr::null_mut(), 0) as _,
                _marker: PhantomData,
            }));
        }
        // This is safe because we have initizalied the memory with init.
        let tensor = unsafe { self.alloc_gmem_ignore_init(init)? };
        {
            let mut view = tensor.as_tensor_view_mut(self);
            view.copy_from_host(init).expect("copy from host failed");
        }
        Ok(tensor)
    }

    /// Allocate a tensor on the GPU and initialize it with the data from `init`.
    /// Return the mutable view of the tensor, which only lives as long as the GPU context guard.
    #[allow(private_bounds)]
    pub fn new_tensor_view<T: ?Sized + SizedOrSliceClone + GpuDataMarker>(
        &self,
        init: &T,
    ) -> Result<TensorViewMut<'a, T>, CudaError> {
        self.new_tensor(init).map(|t| t.as_tensor_view_mut(self))
    }
}

#[expect(private_bounds)]
impl<'a, T: ?Sized + SizedOrSliceClone> TensorView<'a, T> {
    pub fn copy_to_host(&self, dst: &mut T) -> Result<(), CudaError> {
        let len = self.len();
        let dst_len = (dst as &T).len_if_slice().unwrap_or(1);
        let size = core::mem::size_of::<T::UnitType>() * len;
        if dst_len < len {
            return Err(CudaError::MemCopyOutOfBound);
        }
        unsafe {
            let err = cuMemcpyDtoH_v2(dst as *mut _ as _, self.as_devptr(), size);
            if err != CUDA_SUCCESS {
                return Err(CudaError::Err(err));
            }
        }
        Ok(())
    }
}

#[expect(private_bounds)]
impl<'a, T: ?Sized + SizedOrSliceClone> TensorViewMut<'a, T> {
    pub fn copy_from_host(&mut self, src: &T) -> Result<(), CudaError> {
        let len = self.len();
        let src_len = src.len_if_slice().unwrap_or(1);
        let size = core::mem::size_of::<T::UnitType>() * src_len;
        if src_len > len {
            return Err(CudaError::MemCopyOutOfBound);
        }
        unsafe {
            let err = cuMemcpyHtoD_v2(self.as_devptr(), src as *const _ as _, size);
            if err != CUDA_SUCCESS {
                return Err(CudaError::Err(err));
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn memset(&mut self, value: u8) -> Result<(), CudaError> {
        let len = self.len();
        let size = core::mem::size_of::<T::UnitType>() * len;
        unsafe {
            let ret = cuMemsetD8_v2(self.as_devptr(), value, size);
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
        }
        Ok(())
    }
}
