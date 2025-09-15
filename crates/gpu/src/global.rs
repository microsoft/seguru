/// Used to distinguish different memory spaces in GPU programming.
/// GpuGlobal represents global memory space.
/// See shared::GpuShared for shared memory space.
/// When chunking or atomic operations are needed, GpuGlobal is owned by
/// chunk or atomic struct.
/// This ensures that the user cannot access the data without using chunk or
/// atomic operations.
#[rustc_diagnostic_item = "gpu::global::GpuGlobal"]
pub struct GpuGlobal<'a, T: ?Sized> {
    pub(crate) data: &'a mut T, // Accessed only by chunk or atomic constructor.
}

impl<'a, T: ?Sized> GpuGlobal<'a, T> {
    // This is a host-side function.
    #[cfg(not(feature = "codegen_tests"))]
    pub fn new(slice: &'a mut cuda_bindings::CudaMemBox<T>) -> Self {
        unsafe { GpuGlobal { data: &mut *(slice.as_ptr() as *mut T) } }
    }
}

/// Never implement Deref to prevent direct read access to mutable data.
/// When the global mem is immutable, use &T directly instead of &mut T which will be converted to GpuGlobal.
///
/// Can I read global data before write to unique chunk?
/// Yes, but it is not common and requires us to syncronize the read for all running threads from future write access.
/// otherwide, the read may get old or new data indeterministically.
/// This is not a common pattern in GPU programming.
/// So we disallow it for simplicity.
///
/// Can I read the cross-thread global data after write to unique chunk?
/// Yes, but it requires us to syncronize the read for all running threads after write access.
/// otherside, the read may get old or new data indeterministically.
/// This is not a common pattern in GPU programming.
/// So we disallow it for simplicity.
impl<'a, T: ?Sized> !core::ops::Deref for GpuGlobal<'a, T> {}

/// Never implement DerefMut to prevent direct mutable access to the data.
/// This ensures that the user cannot access the data without using chunk or
/// atomic operations.
impl<'a, T: Sized> !core::ops::DerefMut for GpuGlobal<'a, T> {}
