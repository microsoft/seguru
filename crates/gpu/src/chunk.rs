use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use crate::chunk_scope::{ChunkScope, GlobalMemScope};
use crate::{GpuGlobal, assert_ptr};

/// Thread unique mapping trait
///
/// This trait guarantees that each thread produces a unique index mapping,
/// so no two distinct threads map to the same global index.
///
/// # Type Parameters
/// - `CS`: The memory space: GlobalMemScope or SharedMemScope
///
/// # Safety
/// Implementors must ensure:
/// ```text
/// forall |idx1, idx2, thread_ids1, thread_ids2|
///     thread_ids1 != thread_ids2 ==>
///         map(idx1, thread_ids1) !=  map(idx2, thread_ids2)
/// ```
pub unsafe trait ThreadUniqueMap<CS: ChunkScope>: Clone {
    type IndexType;
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        true
    }

    /// Returns the extra precondition of indexing operation and the global
    /// index. Without providing extra precondition, index will always check the
    /// OOB error with global idx.
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; CS::TID_LEN]) -> (bool, usize);
}

/// Provide local_to_global_index for chunking.
/// This is a private trait that should not be used outside this crate.
pub(crate) trait ThreadUniqueMapProvidedMethods<CS: ChunkScope>:
    ThreadUniqueMap<CS>
{
    #[inline]
    #[gpu_codegen::device]
    fn local_to_global_index(&self, idx: Self::IndexType) -> (bool, usize)
    where
        [(); CS::TID_LEN]:,
    {
        self.map(idx, CS::thread_ids())
    }
}

impl<T: ThreadUniqueMap<CS>, CS: ChunkScope> ThreadUniqueMapProvidedMethods<CS> for T {}

/// Represent a chunk of global memory that is uniquely mapped to each thread.
/// It supports both continuous and non-continuous mapping strategies.
/// - T: element type
/// - Map: mapping strategy type
/// - 'a: lifetime of the underlying slice
/// - map_params: parameters for the mapping strategy
pub struct GlobalThreadChunk<'a, T, Map: ThreadUniqueMap<GlobalMemScope>> {
    data: &'a mut [T], // Must be private.
    pub map_params: Map,
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope>> GlobalThreadChunk<'a, T, Map> {
    #[inline]
    #[rustc_diagnostic_item = "gpu::chunk_mut"]
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data(0, 1)]
    /// TODO: We will prevent rechunking of global mem in the mir_analysis.
    /// For now, we just leave it to the user.
    pub fn new(global: GpuGlobal<'a, [T]>, map_params: Map) -> Self {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        Self { data: global.data, map_params }
    }

    /// In some cases, passing GlobalThreadChunk from host to device is more
    /// convenient. Due to unknown optimization-related factors, passing
    /// GlobalThreadChunk to kernel function may make the kernel faster (see
    /// examples/matmul). In addition, directly passing GlobalThreadChunk from
    /// host to device naturally avoids the problem of rechunking the global
    /// mem.
    #[cfg(not(feature = "codegen_tests"))]
    pub fn new_from_host(slice: &'a cuda_bindings::CudaMemBox<[T]>, map_params: Map) -> Self {
        unsafe { Self { data: &mut *(slice.as_ptr() as *mut [T]), map_params } }
    }
}

#[cfg(not(feature = "codegen_tests"))]
unsafe impl<'a, T: Send, Map: ThreadUniqueMap<GlobalMemScope> + 'static + Send>
    cuda_bindings::AsHostKernelParams for GlobalThreadChunk<'a, T, Map>
where
    // Ensure Map is small enough to be passed by value.
    [(); 16 - core::mem::size_of::<Map>()]:,
{
    fn as_kernel_param_data(&self) -> Vec<alloc::boxed::Box<dyn core::any::Any>> {
        assert!(core::mem::size_of::<Map>() <= 16);
        vec![
            Box::new((self.data as *const [T] as *const T) as usize),
            Box::new(self.data.len()),
            Box::new(self.map_params.clone()),
        ]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope>> Index<Map::IndexType>
    for GlobalThreadChunk<'a, T, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: Map::IndexType) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        assert_ptr(self.map_params.precondition() & idx_precondition, &self.data[idx])
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope>> IndexMut<Map::IndexType>
    for GlobalThreadChunk<'a, T, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: Map::IndexType) -> &mut T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        assert_ptr(self.map_params.precondition() & idx_precondition, &mut self.data[idx])
    }
}

/// Chunk a global memory slice into unique chunks.
/// supports mapping strategies for global memory.
/// For example,
/// - chunk_mut(&mut data, MapLinear::new(width))
/// - chunk_mut(&mut data, Map2D::new(x_width))
#[gpu_codegen::device]
#[gpu_codegen::sync_data(0, 1, 2)]
#[inline(always)]
pub fn chunk_mut<'a, T, Map: ThreadUniqueMap<GlobalMemScope>>(
    input: GpuGlobal<'a, [T]>,
    map: Map,
) -> crate::GlobalThreadChunk<'a, T, Map> {
    crate::GlobalThreadChunk::new(input, map)
}
