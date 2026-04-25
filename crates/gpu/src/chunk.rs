use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use num_traits::AsPrimitive;

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
    type IndexType: Copy + 'static;
    type GlobalIndexType: AsPrimitive<usize>;
    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        true
    }

    /// Returns the extra precondition of indexing operation and the global
    /// index. Without providing extra precondition, index will always check the
    /// OOB error with global idx.
    #[gpu_codegen::device]
    fn map(
        &self,
        idx: Self::IndexType,
        thread_ids: [u32; TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType);
}

/// Provide local_to_global_index for chunking.
/// This is a private trait that should not be used outside this crate.
pub(crate) trait ScopeUniqueMapProvidedMethods<CS: ChunkScope>: ScopeUniqueMap<CS> {
    #[inline]
    #[gpu_codegen::device]
    fn local_to_global_index(&self, idx: Self::IndexType) -> (bool, Self::GlobalIndexType) {
        self.map(idx, CS::thread_ids())
    }
}

impl<T: ScopeUniqueMap<CS>, CS: ChunkScope> ScopeUniqueMapProvidedMethods<CS> for T {}

/// Companion to ScopeUniqueMap for Maps that can split the linear-offset
/// computation into a thread-invariant "tile base" and a per-access lid offset.
///
/// # Purpose
/// When kernels unroll a per-thread tile, the thread-portion of the index is
/// invariant across all unrolled accesses. The default `map` path re-computes
/// the full i32 offset + zext + shl + gep per access, which inflates the
/// register footprint of kernels and lowers occupancy. A Map implementing
/// `MapWithLidOffset` lets `GlobalGroupChunk::open_tile` materialize the tile
/// base as a 64-bit pointer once and pay only compile-time-constant pointer
/// arithmetic per access.
///
/// # Safety
/// Implementors must ensure:
/// ```text
/// forall lid, tid:
///     map(lid, tid).1 == map(<lid=0>, tid).1 + map_lid_offset(lid).1
/// ```
/// i.e. the lid and tid contributions to the linear offset are independent
/// and additive, and `map_lid_offset` includes no thread-dependent terms.
/// The validity flag returned by `map_lid_offset` must match the lid-portion
/// of bounds checks in `map`.
pub unsafe trait MapWithLidOffset<CS: ChunkScope>: ScopeUniqueMap<CS> {
    #[gpu_codegen::device]
    fn map_lid_offset(&self, idx: Self::IndexType) -> (bool, Self::GlobalIndexType);

    /// Thread-invariant portion of the linear index, including the struct's
    /// base offset. By contract:
    ///   `map(lid, tid).1 == thread_base(tid).1 + map_lid_offset(lid).1`
    /// and `thread_base(tid).0 == (tid-portion of map's validity)`.
    #[gpu_codegen::device]
    fn thread_base(
        &self,
        thread_ids: [u32; crate::chunk_scope::TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType);
}

/// Companion to `MapWithLidOffset` for maps whose local index is a 2D
/// `(col, row)` pair that can be split into a row base and an in-row offset.
///
/// # Safety
/// Implementors must ensure:
/// ```text
/// forall col, row:
///     map_lid_offset((col, row)).1 == row_lid_offset(row).1 + in_row_lid_offset(col).1
/// ```
/// Implementors must also preserve the validity-flag algebra:
/// ```text
/// map_lid_offset((col, row)).0 == (row_lid_offset(row).0 & in_row_lid_offset(col).0)
/// ```
/// and the validity flags returned by the split methods must match the row and
/// in-row portions of the original local-index bounds checks. Neither method
/// may include thread-dependent terms.
pub unsafe trait MapWithRows<CS: ChunkScope>:
    MapWithLidOffset<CS, IndexType = (u32, u32)>
{
    #[gpu_codegen::device]
    fn row_lid_offset(&self, row: u32) -> (bool, Self::GlobalIndexType);

    #[gpu_codegen::device]
    fn in_row_lid_offset(&self, col: u32) -> (bool, Self::GlobalIndexType);
}

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
        Map: ScopeUniqueMap<CS>,
        CS: ChunkScope<ToScope = CS2::FromScope>,
        Map2::GlobalIndexType: AsPrimitive<Map::IndexType>,
    {
        GlobalGroupChunk {
            data: self.data,
            map_params: ChainedMap::new(self.map_params, map),
            dummy: PhantomData,
        }
    }

    #[gpu_codegen::device]
    #[inline]
    pub fn local2global(
        &self,
        idx: <Map as ScopeUniqueMap<CS>>::IndexType,
    ) -> Map::GlobalIndexType {
        self.map_params.local_to_global_index(idx).1
    }

    /// Raw mutable pointer to the underlying slice. Used by `open_tile` in
    /// `tile.rs` to derive a per-thread pre-offset pointer.
    #[gpu_codegen::device]
    #[inline(always)]
    pub(crate) fn data_ptr_mut(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

// Read-only access is always allowed.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> Index<Map::IndexType>
    for GlobalGroupChunk<'a, T, CS, Map>
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: Map::IndexType) -> &T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let idx = idx.as_();
        assert_ptr(self.map_params.precondition() & idx_precondition, &self.data[idx])
    }
}

// Mutable access is only allowed when ToScope is Thread,
// indicating that each thread has a unique chunk.
impl<'a, T, CS: ChunkScope, Map: ScopeUniqueMap<CS>> IndexMut<Map::IndexType>
    for GlobalGroupChunk<'a, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: Map::IndexType) -> &mut T {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let idx = idx.as_();
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
