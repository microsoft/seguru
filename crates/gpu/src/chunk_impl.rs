use crate::chunk::ScopeUniqueMap;
use crate::chunk_scope::{ChunkScope, TID_MAX_LEN};

/// Linear mapping for 1D array.
/// N is the number of thread dimensions.
/// width is the chunking window.
/// The array is divided into chunks along threads until all elements are covered.
///
/// ## Examples
///
/// Given:
/// bdim = (x=4, y=1, z=1)
/// gdim = (x=1, y=1, z=1)
/// arr: [T; 8]
/// width = 2
///
/// With `width = 2`, the mapping assign two continuous elements to thread 0-3:
/// [0, 0, 1, 1, 2, 2, 3, 3]
///
/// width = 1, the mapping assign two non-continuous elements to thread 0-3:
/// [0, 1, 2, 3, 0, 1, 2, 3]
#[derive(Copy, Clone)]
pub struct MapLinearWithDim<const N: usize = 3> {
    width: usize,
}

pub type MapLinear = MapLinearWithDim<3>;

impl<const N: usize> MapLinearWithDim<N> {
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

/// # Safety
/// It is safe to use this mapping as long as the thread dimensions are properly
/// configured.
unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<1> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_y() == 1 && CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<2> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<3> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: usize, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        let x_id = CS::global_id_x(thread_ids);
        let y_id = CS::global_id_y(thread_ids);
        let z_id = CS::global_id_z(thread_ids);
        let global_thread_id = x_id + (z_id * CS::global_dim_y() + y_id) * CS::global_dim_x();
        let stride = self.width;
        let total_dim = CS::global_dim_x() * CS::global_dim_y() * CS::global_dim_z();
        (true, idx % stride + (idx / stride) * stride * total_dim + global_thread_id * stride)
    }
}

/// This mapping strategy is useful when we want to reshape a 1D array into a 2D
/// array and then distribute one element to a thread one by one until consuming
/// all. It creates a new non-continuous partition for each thread.
/// - IndexType is (usize, usize)
///
/// Example:
/// Example:
/// - array: [T; 20]
/// - x_size = 5 => y_size = 4
/// - dim: x=2, y=2, z=1
/// ```text
/// 0   1   2   3   4   5
/// ┌───┬───┬───┬───┬───┐
/// │0,0│1,0│0,0│1,0│0,0│
/// ├───┼───┼───┼───┼───┤
/// │0,1│1,1│0,1│1,1│0,1│
/// ├───┼───┼───┼───┼───┤
/// │0,0│1,0│0,0│1,0│0,0│
/// ├───┼───┼───┼───┼───┤
/// │0,1│1,1│0,1│1,1│0,1│
/// └───┴───┴───┴───┴───┘
/// ```
/// In this case,
/// - thread(1,0) and (1,1) should only have access to a shape of 2*2 = 4 elements.
/// - thread (0,0) and (0,1) have access to a shape of 3*2 = 6 elements.
#[derive(Copy, Clone)]
pub struct Map2D {
    pub x_size: usize,
}

impl Map2D {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(x_size: usize) -> Self {
        Self { x_size }
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for Map2D {
    type IndexType = (usize, usize);

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        let shape_x = self.x_size;
        let inner_x = idx.0;
        let inner_y = idx.1;
        let x = inner_x * CS::global_dim_x() + CS::global_id_x(thread_ids);
        let y = inner_y * CS::global_dim_y() + CS::global_id_y(thread_ids);
        (x < shape_x, shape_x * y + x)
    }
}
