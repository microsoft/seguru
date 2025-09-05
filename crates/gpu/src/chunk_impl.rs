use crate::chunk::ThreadUniqueMap;
use crate::{DimX, DimY, DimZ, block_dim, dim};

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
    let x_id = thread_ids[3] * block_dim::<DimX>() + thread_ids[0];
    let y_id = thread_ids[4] * block_dim::<DimY>() + thread_ids[1];
    let z_id = thread_ids[5] * block_dim::<DimZ>() + thread_ids[2];
    x_id + (z_id * dim::<DimY>() + y_id) * dim::<DimX>()
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
        dim::<DimY>() == 1 && dim::<DimZ>() == 1
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
        dim::<DimZ>() == 1
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

/// This mapping strategy is useful when we want to reshape a 1D array into a 2D
/// array and then distribute one element to a thread one by one until consuming
/// all. It creates a new non-continuous partition for each thread.
/*
Example:
array: [T; 20]
x_size = 5 => y_size = 4
dim: x=2, y=2, z=1
0     2     4     6   7
┌───┬───┬───┬───┬───┐
│0,0│1,0│0,0│1,0│0,0│
├───┼───┼───┼───┼───┤
│0,1│1,1│0,1│1,1│0,1│
├───┼───┼───┼───┼───┤
│0,0│1,0│0,0│1,0│0,0│
├───┼───┼───┼───┼───┤
│0,1│1,1│0,1│1,1│0,1│
└───┴───┴───┴───┴───┘
In this case, thread(1,0) and (1,1) should
only have access to a shape of 2*2 = 4 elements,
while thread (0,0) and (0,1) have access to a shape of 3*2 = 6 elements.
*/
#[derive(Copy, Clone)]
pub struct Map2D {
    pub x_size: usize,
}

impl Map2D {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(0)]
    pub fn new(x_size: usize) -> Self {
        Self { x_size }
    }
}

unsafe impl ThreadUniqueMap<2> for Map2D {
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        dim::<DimZ>() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: [usize; 2], thread_ids: [usize; 6]) -> (bool, usize) {
        let shape_x = self.x_size;
        let inner_x = idx[0];
        let inner_y = idx[1];
        let x = inner_x * dim::<DimX>() + block_dim::<DimX>() * thread_ids[3] + thread_ids[0];
        let y = inner_y * dim::<DimY>() + block_dim::<DimY>() * thread_ids[4] + thread_ids[1];
        (x < shape_x, shape_x * y + x)
    }
}
