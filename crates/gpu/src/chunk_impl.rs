use crate::chunk::ThreadUniqueMap;
use crate::{DimType, block_dim, dim};

/// Unique mapping for continuous memory
/// N is the number of thread dimensions.
#[derive(Copy, Clone)]
pub struct MapLinearWithDim<const N: usize = 3> {
    width: usize,
}

pub type MapLinear = MapLinearWithDim<3>;

#[gpu_codegen::device]
#[inline]
fn global_id(thread_ids: [usize; 6]) -> usize {
    let x_id = thread_ids[3] * block_dim(DimType::X) + thread_ids[0];
    let y_id = thread_ids[4] * block_dim(DimType::Y) + thread_ids[1];
    let z_id = thread_ids[5] * block_dim(DimType::Z) + thread_ids[2];
    x_id + (z_id * dim(DimType::Y) + y_id) * dim(DimType::X)
}

impl<const N: usize> MapLinearWithDim<N> {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(0)]
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

/// # Safety
/// It is safe to use this mapping as long as the thread dimensions are properly
/// configured.
unsafe impl ThreadUniqueMap<1> for MapLinearWithDim<1> {
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        dim(DimType::Y) == 1 && dim(DimType::Z) == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; 1], thread_ids: [usize; 6]) -> (bool, usize) {
        MapLinearWithDim::<3>::new(self.width).map(idx, thread_ids)
    }
}

unsafe impl ThreadUniqueMap<1> for MapLinearWithDim<2> {
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        dim(DimType::Z) == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; 1], thread_ids: [usize; 6]) -> (bool, usize) {
        MapLinearWithDim::<3>::new(self.width).map(idx, thread_ids)
    }
}

unsafe impl ThreadUniqueMap<1> for MapLinearWithDim<3> {
    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; 1], thread_ids: [usize; 6]) -> (bool, usize) {
        let global_thread_id = global_id(thread_ids);
        (true, (idx[0] + global_thread_id) * self.width)
    }
}

#[gpu_codegen::device]
#[gpu_codegen::sync_data(0, 1, 2)]
#[inline(always)]
pub fn chunk_mut<'a, T>(
    input: &'a mut [T],
    window: usize,
    _idx: crate::GpuChunkIdx,
) -> crate::GlobalThreadChunk<'a, T, 1, MapLinear> {
    crate::GlobalThreadChunk::new(input, MapLinear::new(window))
}
