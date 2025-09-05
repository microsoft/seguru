use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::{
    DimType, assert_before_index, block_dim, block_id, block_thread_ids, dim, grid_dim, thread_id,
};
#[cfg(not(feature = "codegen_tests"))]
use crate::{GpuChunkable2D, GpuChunkableMut2D};

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

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_mut_2d"]
#[gpu_codegen::device]
pub fn get_local_mut_2d<'a, T>(a: &mut GpuChunkableMut2D<'a, T>, x: usize, y: usize) -> &'a mut T {
    // Must check if col is smaller than a.size_x

    let row =
        y * dim(DimType::Y) + block_dim(DimType::Y) * block_id(DimType::Y) + thread_id(DimType::Y);
    let col =
        x * dim(DimType::X) + block_dim(DimType::X) * block_id(DimType::X) + thread_id(DimType::X);
    let z_size = dim(DimType::Z);
    let idx = a.size_x() * row + col;
    assert_before_index(z_size == 1, idx);
    assert_before_index(col <= a.size_x(), idx);

    // Here Rust will automatic generate an SFI
    unsafe { &mut (&mut *(a.as_ptr() as *mut [T]))[a.size_x() * row + col] }
}

#[cfg(not(feature = "codegen_tests"))]
#[rustc_diagnostic_item = "gpu::get_local_2d"]
#[gpu_codegen::device]
pub fn get_local_2d<'a, T>(a: &'a GpuChunkable2D<'a, T>, x: usize, y: usize) -> &'a T {
    // Must check if col is smaller than a.size_x

    use crate::assert_before_index;
    let row = y * grid_dim(DimType::Y) * block_dim(DimType::Y)
        + block_dim(DimType::Y) * block_id(DimType::Y)
        + thread_id(DimType::Y);
    let col = x * grid_dim(DimType::X) * block_dim(DimType::X)
        + block_dim(DimType::X) * block_id(DimType::X)
        + thread_id(DimType::X);
    let z_size = grid_dim(DimType::Z) * block_dim(DimType::Z);
    let idx = a.size_x() * row + col;
    assert_before_index(z_size == 1, idx);
    assert_before_index(col <= a.size_x(), idx);

    // Here Rust will automatic generate an SFI
    unsafe { &(&*a.as_ptr())[idx] }
}
