use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::{assert_before_index, block_thread_ids};

/// Thread unique mapping trait
/// N: number of index dimensions.
/// N can be different from GPU thread dimensions.
/// # Safety
/// requires idx -> thread-unique idx
/// forall |idx1, idx2, thread_ids1, thread_ids2| thread_ids1 != thread_ids2 ==> map(idx1, thread_ids1) !=  map(idx2, thread_ids2)
pub(crate) unsafe trait ThreadUniqueMap<const N: usize>: Clone {
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        true
    }

    /// Returns the extra precondition of indexing operation and the global
    /// index. Without providing extra precondition, index will always check the
    /// OOB error with global idx.
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; N], thread_ids: [usize; 6]) -> (bool, usize);
}

/// N: Index dimension, 1, 2, 3
/// Map: Mapping strategy
#[allow(private_bounds)]
pub struct GlobalThreadChunk<'a, T, const N: usize, Map: ThreadUniqueMap<N>> {
    data: &'a mut [T], // Must be private.
    pub map_params: Map,
    dummy: PhantomData<[u8; N]>,
}

#[expect(private_bounds)]
impl<'a, T, const N: usize, Map: ThreadUniqueMap<N>> GlobalThreadChunk<'a, T, N, Map> {
    #[inline]
    #[rustc_diagnostic_item = "gpu::chunk_mut"]
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data(0, 1)]
    pub fn new(data: &'a mut [T], map_params: Map) -> Self {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        Self { data, map_params, dummy: PhantomData }
    }

    #[inline]
    #[gpu_codegen::device]
    fn local_to_global_index(&self, i: [usize; N]) -> (bool, usize) {
        let ids = block_thread_ids();
        self.map_params.map(i, ids)
    }
}

impl<'a, T, Map: ThreadUniqueMap<1>> Index<usize> for GlobalThreadChunk<'a, T, 1, Map> {
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: usize) -> &T {
        let (idx_precondition, idx) = self.local_to_global_index([idx]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<1>> IndexMut<usize> for GlobalThreadChunk<'a, T, 1, Map> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        let (idx_precondition, idx) = self.local_to_global_index([idx]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<2>> Index<(usize, usize)> for GlobalThreadChunk<'a, T, 2, Map> {
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: (usize, usize)) -> &T {
        let (idx_precondition, idx) = self.local_to_global_index([idx.0, idx.1]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<2>> IndexMut<(usize, usize)> for GlobalThreadChunk<'a, T, 2, Map> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        let (idx_precondition, idx) = self.local_to_global_index([idx.0, idx.1]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<3>> Index<(usize, usize, usize)>
    for GlobalThreadChunk<'a, T, 3, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: (usize, usize, usize)) -> &T {
        let (idx_precondition, idx) = self.local_to_global_index([idx.0, idx.1, idx.2]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &self.data[idx]
    }
}

impl<'a, T, Map: ThreadUniqueMap<3>> IndexMut<(usize, usize, usize)>
    for GlobalThreadChunk<'a, T, 3, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: (usize, usize, usize)) -> &mut T {
        let (idx_precondition, idx) = self.local_to_global_index([idx.0, idx.1, idx.2]);
        assert_before_index(self.map_params.precondition() & idx_precondition, idx);
        &mut self.data[idx]
    }
}
