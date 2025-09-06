use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::assert_before_index;
use crate::chunk_scope::{ChunkScope, GlobalMemScope};

/// Thread unique mapping trait
/// N: number of index dimensions.
/// N can be different from GPU thread dimensions.
/// # Safety
/// requires idx -> thread-unique idx
/// forall |idx1, idx2, thread_ids1, thread_ids2| thread_ids1 != thread_ids2 ==> map(idx1, thread_ids1) !=  map(idx2, thread_ids2)
pub unsafe trait ThreadUniqueMap<CS: ChunkScope, const N: usize>: Clone {
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        true
    }

    /// Returns the extra precondition of indexing operation and the global
    /// index. Without providing extra precondition, index will always check the
    /// OOB error with global idx.
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; N], thread_ids: [usize; CS::TID_LEN]) -> (bool, usize);
}

/// Provide local_to_global_index for chunking.
/// This is a private trait that should not be used outside this crate.
pub(crate) trait ThreadUniqueMapProvidedMethods<CS: ChunkScope, const N: usize>:
    ThreadUniqueMap<CS, N>
{
    #[inline]
    #[gpu_codegen::device]
    fn local_to_global_index(&self, idx: [usize; N]) -> (bool, usize)
    where
        [(); CS::TID_LEN]:,
    {
        self.map(idx, CS::thread_ids())
    }
}

impl<T: ThreadUniqueMap<CS, N>, CS: ChunkScope, const N: usize>
    ThreadUniqueMapProvidedMethods<CS, N> for T
{
}

/// N: Index dimension, 1, 2, 3
/// Map: Mapping strategy
pub struct GlobalThreadChunk<'a, T, const N: usize, Map: ThreadUniqueMap<GlobalMemScope, N>> {
    data: &'a mut [T], // Must be private.
    pub map_params: Map,
    dummy: PhantomData<[u8; N]>,
}

impl<'a, T, const N: usize, Map: ThreadUniqueMap<GlobalMemScope, N>>
    GlobalThreadChunk<'a, T, N, Map>
{
    #[inline]
    #[rustc_diagnostic_item = "gpu::chunk_mut"]
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data(0, 1)]
    /// TODO: We will prevent rechunking of global mem in the mir_analysis.
    /// For now, we just leave it to the user.
    pub fn new(data: &'a mut [T], map_params: Map) -> Self {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        Self { data, map_params, dummy: PhantomData }
    }

    /// In some cases, passing GlobalThreadChunk from host to device is more
    /// convenient. Due to unknown optimization-related factors, passing
    /// GlobalThreadChunk to kernel function may make the kernel faster (see
    /// examples/matmul). In addition, directly passing GlobalThreadChunk from
    /// host to device naturally avoids the problem of rechunking the global
    /// mem.
    #[cfg(not(feature = "codegen_tests"))]
    pub fn new_from_host(slice: &'a cuda_bindings::CudaMemBox<[T]>, map_params: Map) -> Self {
        unsafe { Self { data: &mut *(slice.as_ptr() as *mut [T]), map_params, dummy: PhantomData } }
    }
}

#[cfg(not(feature = "codegen_tests"))]
unsafe impl<'a, T: Send, const N: usize, Map: ThreadUniqueMap<GlobalMemScope, N> + 'static + Send>
    cuda_bindings::AsHostKernelParams for GlobalThreadChunk<'a, T, N, Map>
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

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 1>> Index<usize>
    for GlobalThreadChunk<'a, T, 1, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: usize) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 1>> IndexMut<usize>
    for GlobalThreadChunk<'a, T, 1, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 2>> Index<(usize, usize)>
    for GlobalThreadChunk<'a, T, 2, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: (usize, usize)) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx.0, idx.1]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 2>> IndexMut<(usize, usize)>
    for GlobalThreadChunk<'a, T, 2, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx.0, idx.1]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 3>> Index<(usize, usize, usize)>
    for GlobalThreadChunk<'a, T, 3, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: (usize, usize, usize)) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx.0, idx.1, idx.2]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<GlobalMemScope, 3>> IndexMut<(usize, usize, usize)>
    for GlobalThreadChunk<'a, T, 3, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: (usize, usize, usize)) -> &mut T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index([idx.0, idx.1, idx.2]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}
