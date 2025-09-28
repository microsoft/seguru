use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::chunk_scope::{
    ChainedMap, ChainedScope, ChunkScope, Grid, Grid2ThreadScope, TID_MAX_LEN, Thread,
};
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
pub unsafe trait ScopeUniqueMap<CS: ChunkScope>: Clone {
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
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize);
}

/// Provide local_to_global_index for chunking.
/// This is a private trait that should not be used outside this crate.
pub(crate) trait ScopeUniqueMapProvidedMethods<CS: ChunkScope>: ScopeUniqueMap<CS> {
    #[inline]
    #[gpu_codegen::device]
    fn local_to_global_index(&self, idx: Self::IndexType) -> (bool, usize) {
        self.map(idx, CS::thread_ids())
    }
}

impl<T: ScopeUniqueMap<CS>, CS: ChunkScope> ScopeUniqueMapProvidedMethods<CS> for T {}

/// Represent a chunk of global memory that is uniquely mapped to each thread group.
/// It supports both continuous and non-continuous mapping strategies.
/// - T: element type
/// - Map: mapping strategy type
/// - 'a: lifetime of the underlying slice
/// - map_params: parameters for the mapping strategy
pub struct GlobalGroupChunk<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> {
    data: &'a mut [T], // Must be private.
    pub map_params: Map,
    dummy: PhantomData<CS>,
}

pub type GlobalThreadChunk<'a, T, Map> = GlobalGroupChunk<'a, T, Grid2ThreadScope, Map>;

/// Creating global chunk from GpuGlobal.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> GlobalGroupChunk<'a, T, CS, Map>
where
    CS: ChunkScope<FromScope = Grid>,
{
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
        Self { data: global.data, map_params, dummy: PhantomData }
    }

    /// In some cases, passing GlobalThreadChunk from host to device is more
    /// convenient. Due to unknown optimization-related factors, passing
    /// GlobalThreadChunk to kernel function may make the kernel faster (see
    /// examples/matmul). In addition, directly passing GlobalThreadChunk from
    /// host to device naturally avoids the problem of rechunking the global
    /// mem.
    #[cfg(not(feature = "codegen_tests"))]
    pub fn new_from_host<'b: 'a>(
        slice: &'a mut cuda_bindings::TensorViewMut<'b, [T]>,
        map_params: Map,
    ) -> Self {
        unsafe {
            Self {
                data: &mut *(slice.as_flat_devptr() as *mut [T]),
                map_params,
                dummy: PhantomData,
            }
        }
    }
}

/// Creating global chunk from GpuGlobal.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> GlobalGroupChunk<'a, T, CS, Map> {
    /// Convert GlobalGroupChunk to another GlobalGroupChunk with different ChunkScope and Map.
    /// See `ChunkScope` for more details about chunk scope.
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data(2)]
    pub fn chunk_to_scope<CS2: ChunkScope, Map2: ScopeUniqueMap<CS2>>(
        self,
        _cs: CS2,
        map: Map2,
    ) -> GlobalGroupChunk<'a, T, ChainedScope<CS, CS2>, ChainedMap<CS, CS2, Map, Map2>>
    where
        Map: ScopeUniqueMap<CS, IndexType = usize>,
        CS: ChunkScope<ToScope = CS2::FromScope>,
    {
        GlobalGroupChunk {
            data: self.data,
            map_params: ChainedMap::new(self.map_params, map),
            dummy: PhantomData,
        }
    }

    #[gpu_codegen::device]
    #[inline]
    pub fn local2global(&self, idx: <Map as ScopeUniqueMap<CS>>::IndexType) -> usize {
        self.map_params.local_to_global_index(idx).1
    }
}

#[cfg(not(feature = "codegen_tests"))]
unsafe impl<'a, T: Send, CS: ChunkScope, Map: ScopeUniqueMap<CS> + 'static + Send>
    cuda_bindings::AsHostKernelParams for GlobalGroupChunk<'a, T, CS, Map>
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

// Read-only access is always allowed.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> Index<Map::IndexType>
    for GlobalGroupChunk<'a, T, CS, Map>
where
    [(); TID_MAX_LEN]:,
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: Map::IndexType) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        assert_ptr(self.map_params.precondition() & idx_precondition, &self.data[idx])
    }
}

// Mutable access is only allowed when ToScope is Thread,
// indicating that each thread has a unique chunk.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> IndexMut<Map::IndexType>
    for GlobalGroupChunk<'a, T, CS, Map>
where
    [(); TID_MAX_LEN]:,
    CS: ChunkScope<ToScope = Thread>,
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
pub fn chunk_mut<'a, T, Map: ScopeUniqueMap<Grid2ThreadScope>>(
    input: GpuGlobal<'a, [T]>,
    map: Map,
) -> GlobalThreadChunk<'a, T, Map> {
    GlobalThreadChunk::new(input, map)
}
