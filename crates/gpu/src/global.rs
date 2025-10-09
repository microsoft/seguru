use core::ptr::slice_from_raw_parts_mut;

#[cfg(not(feature = "codegen_tests"))]
use cuda_bindings::TensorViewMut;

use crate::GlobalGroupChunk;
use crate::chunk::ScopeUniqueMap;
use crate::chunk_scope::{ChunkScope, Grid};

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
    pub fn new<'b: 'a>(slice: TensorViewMut<'a, T>) -> Self {
        unsafe { GpuGlobal { data: &mut *(slice.as_flat_devptr() as *mut T) } }
    }
}

impl<'a, T> GpuGlobal<'a, [T]> {
    /// Convert GpuGlobal to GlobalGroupChunk in one step.
    /// See `ChunkScope` for more details about chunk scope.
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data(0, 1, 2)]
    #[inline(always)]
    pub fn chunk_to_scope<CS, Map: ScopeUniqueMap<CS>>(
        self,
        _cs: CS,
        m: Map,
    ) -> GlobalGroupChunk<'a, T, CS, Map>
    where
        CS: ChunkScope<FromScope = Grid>,
    {
        GlobalGroupChunk::new(self, m)
    }

    /// Useful to optimize code with vector load/store.
    /// If length of the slice is not a multiple of N,
    /// the remaining elements will be ignored.
    /// For now, we only use reshape for global memory.
    /// For shared memory, user can use GpuShared<[[T; N]]> directly.
    #[gpu_codegen::device]
    #[inline(always)]
    pub fn reshape<const N: usize>(self) -> GpuGlobal<'a, [[T; N]]> {
        // SAFETY: the returned slice will be at same size or shorter, so it is safe.
        unsafe {
            GpuGlobal {
                data: &mut *slice_from_raw_parts_mut(self.data.as_mut_ptr() as _, self.len() / N),
            }
        }
    }

    #[inline(always)]
    #[gpu_codegen::device]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    #[gpu_codegen::device]
    pub fn len(&self) -> usize {
        self.data.len()
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
#[cfg(not(doc))]
impl<'a, T: ?Sized> !core::ops::Deref for GpuGlobal<'a, T> {}

/// Never implement DerefMut to prevent direct mutable access to the data.
/// This ensures that the user cannot access the data without using chunk or
/// atomic operations.
impl<'a, T: Sized> !core::ops::DerefMut for GpuGlobal<'a, T> {}
