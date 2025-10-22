use core::marker::PhantomData;

use num_traits::AsPrimitive;

pub use crate::cg::{Block, Grid, Thread, ThreadWarpTile};
use crate::chunk::ScopeUniqueMap;
use crate::dim::{
    DimType, DimTypeID, DimX, DimY, DimZ, block_dim, block_id, block_size, dim, num_blocks,
    thread_id,
};
use crate::grid_dim;

trait PrivateTraitGuard {}

trait SyncScope {}

/// The length of thread_ids array.
/// A thread can be indexed by one of following ways:
/// - thread_id_{x,y,z}, block_id_{x,y,z}
/// - _, lane_id, warp_id, block_id_{x,y,z}
pub const TID_MAX_LEN: usize = 6;

impl SyncScope for Thread {}

impl<const SIZE: usize> SyncScope for ThreadWarpTile<SIZE> {}
impl SyncScope for Block {}
impl SyncScope for Grid {}

#[expect(private_bounds)]
pub trait BuildChunkScope<S2: SyncScope>: SyncScope {
    type CS: ChunkScope<FromScope = Self, ToScope = S2>;
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    fn build_chunk_scope(&self, to: S2) -> Self::CS;
}

// Block -> Warp
impl<const SIZE: usize> BuildChunkScope<ThreadWarpTile<SIZE>> for Block {
    type CS = Block2WarpScope<SIZE>;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: ThreadWarpTile<SIZE>) -> Block2WarpScope<SIZE> {
        Block2WarpScope
    }
}

// Block -> Thread
impl BuildChunkScope<Thread> for Block {
    type CS = Block2ThreadScope;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: Thread) -> Block2ThreadScope {
        Block2ThreadScope
    }
}

// Grid -> Block
impl BuildChunkScope<Block> for Grid {
    type CS = Grid2BlockScope;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: Block) -> Grid2BlockScope {
        Grid2BlockScope
    }
}

// Grid -> Warp
impl<const SIZE: usize> BuildChunkScope<ThreadWarpTile<SIZE>> for Grid {
    type CS = Grid2WarpScope<SIZE>;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: ThreadWarpTile<SIZE>) -> Grid2WarpScope<SIZE> {
        Grid2WarpScope
    }
}

// Grid -> Thread
impl BuildChunkScope<Thread> for Grid {
    type CS = Grid2ThreadScope;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: Thread) -> Grid2ThreadScope {
        Grid2ThreadScope
    }
}

/// Warp -> Thread
impl<const SIZE: usize> BuildChunkScope<Thread> for ThreadWarpTile<SIZE> {
    type CS = Warp2ThreadScope<SIZE>;
    #[inline]
    #[gpu_codegen::device]
    fn build_chunk_scope(&self, _to: Thread) -> Warp2ThreadScope<SIZE> {
        Warp2ThreadScope
    }
}

#[inline]
#[gpu_codegen::device]
#[expect(private_bounds)]
#[gpu_codegen::ret_sync_data(1000)]
pub fn build_chunk_scope<S1, S2>(from: S1, to: S2) -> <S1 as BuildChunkScope<S2>>::CS
where
    S2: SyncScope,
    S1: BuildChunkScope<S2> + SyncScope,
{
    from.build_chunk_scope(to)
}

/// This trait provides chunking scope information,
/// indicating chunking from a larger scope to a smaller scope:
/// Grid -> Cluster -> Block -> Warp -> Thread
///
/// ## Primitive Chunk Scopes
///
/// Cluster is now mostly out of scope due to limited hardware support.
/// We currently consider the following 6 scope transitions:
///
/// - `Grid2Block`: chunking from Grid scope to Block scope.
/// - `Block2Thread`: chunking from Block scope to Thread scope.
/// - `Grid2Thread`: chunking from Grid scope to Thread scope.
/// - `Grid2Warp`: chunking from Grid scope to Warp scope.
/// - `Block2Warp`: chunking from Block scope to Warp scope.
/// - `Warp2Thread`: chunking from Warp scope to Thread scope.
///
/// ## Chained Scope and Chained Map
///
/// In addition to primitive chunk scopes, we provide `ChainedScope`,
/// which allows chaining two scopes together. `ChainedScope` is always used
/// together with `ChainedMap` to chain two mapping strategies.
///
/// For example, `Grid2Block + Block2Thread` is similar to `Grid2Thread`
/// but differs slightly in usage.
///
/// ### When should I use Primitive Scopes instead of Chained Scopes?
///
/// **_Chained scope/mapping should not be over-used._**
///
/// For simplicity, you may not always need to chunk from a scope to its next scope.
/// For example, use `Grid2Warp` directly instead of `Grid2Block + Block2Warp`
/// if the intermediate chunking strategy does not matter much.
///
/// This is especially true when access patterns switch between scopes,
/// e.g., from grid to warp for read and then from warp to thread for write.
///
/// ### When should I use Chained Scopes?
///
/// `ChainedScope` is useful for:
/// - Removing some threads from work, and
/// - Applying different mapping strategies at different levels.
///
/// ## Examples
///
/// ### Direct chunk: Grid -> Thread
///
/// The following example is similar to chunk_mut(...) with MapLinear.
///
/// ```rust
/// use gpu::MapLinear;
/// use gpu::chunk_scope::{build_chunk_scope, Grid, Thread, ThreadWarpTile};
///
/// fn kernel(input: gpu::GpuGlobal<[f32]>) {
///     let g2w = build_chunk_scope(Grid, Thread);
///     let _ = input.chunk_to_scope(g2w, MapLinear::new(2));
/// }
/// ```
///
/// ### Flexible chunking: Grid -> Warp -> Thread
///
/// With `Grid2Warp + Warp2Thread`, only lane0 of each warp may access elements.
///
/// ```rust
/// use gpu::MapLinear;
/// use gpu::chunk_scope::{build_chunk_scope, Grid, Thread, ThreadWarpTile};
///
/// fn kernel(input: gpu::GpuGlobal<[f32]>) {
///     let warp = ThreadWarpTile::<32>;
///     let g2w = build_chunk_scope(Grid, warp);
///     let w2t = build_chunk_scope(warp, Thread);
///     input
///         .chunk_to_scope(g2w, MapLinear::new(2))
///         .chunk_to_scope(w2t, MapLinear::new(2));
/// }
/// ```
///
/// ### Invalid scope transitions will be rejected
///
/// The `chunk_to_scope` API guarantees valid scope transitions.
/// ```rust,compile_fail,E0271
/// use gpu::MapLinear;
/// use gpu::chunk_scope::{build_chunk_scope, Grid, Block, Thread, ThreadWarpTile};
///
/// fn kernel(input: gpu::GpuGlobal<[f32]>) {
///     let warp = ThreadWarpTile::<32>;
///     let g2w = build_chunk_scope(Grid, warp);
///     let b2t = build_chunk_scope(Block, Thread);
///     // This should not compile, as the scope transition is invalid.
///     // Type mismatch resolving `<Block2ThreadScope as ChunkScope>::FromScope == ThreadWarpTile`
///     input.chunk_to_scope(g2w, MapLinear::new(2))
///          .chunk_to_scope(b2t, MapLinear::new(2));
/// }
/// ```
///
/// TODO: `ToScope` can be leveraged for static analysis of memory access patterns.
/// It may be used to check required synchronization scopes.
#[expect(private_bounds)]
pub trait ChunkScope: PrivateTraitGuard + Clone {
    type FromScope: SyncScope;
    type ToScope: SyncScope;

    fn thread_ids() -> [u32; TID_MAX_LEN];

    fn global_dim<D: DimType>() -> u32;
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32;

    /// Provided methods.
    #[inline]
    #[gpu_codegen::device]
    fn global_id_x(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        Self::global_id::<DimX>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id_y(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        Self::global_id::<DimY>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id_z(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        Self::global_id::<DimZ>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_x() -> u32 {
        Self::global_dim::<DimX>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_y() -> u32 {
        Self::global_dim::<DimY>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_z() -> u32 {
        Self::global_dim::<DimZ>()
    }
}

#[derive(Copy, Clone)]
pub struct Grid2ThreadScope;
impl PrivateTraitGuard for Grid2ThreadScope {}
impl ChunkScope for Grid2ThreadScope {
    type FromScope = Grid;
    type ToScope = Thread;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        // global memory is accessible across blocks,
        // so we need block_id as well as thread_id.
        [
            thread_id::<DimX>(),
            thread_id::<DimY>(),
            thread_id::<DimZ>(),
            block_id::<DimX>(),
            block_id::<DimY>(),
            block_id::<DimZ>(),
        ]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        thread_ids[(D::DIM_ID + DimTypeID::Max as u8) as usize] * block_dim::<D>()
            + thread_ids[D::DIM_ID as usize]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        dim::<D>()
    }
}

#[derive(Copy, Clone)]
pub struct Block2ThreadScope;
impl PrivateTraitGuard for Block2ThreadScope {}
impl ChunkScope for Block2ThreadScope {
    type FromScope = Block;
    type ToScope = Thread;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        // shared memory is shared within a block and so no block_id.
        [thread_id::<DimX>(), thread_id::<DimY>(), thread_id::<DimZ>(), 0, 0, 0]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        block_dim::<D>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        thread_ids[D::DIM_ID as usize]
    }
}

#[derive(Copy, Clone)]
pub struct Grid2BlockScope;
impl PrivateTraitGuard for Grid2BlockScope {}
impl ChunkScope for Grid2BlockScope {
    type FromScope = Grid;
    type ToScope = Block;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        [0, 0, 0, block_id::<DimX>(), block_id::<DimY>(), block_id::<DimZ>()]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        thread_ids[(D::DIM_ID + DimTypeID::Max as u8) as usize]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        grid_dim::<D>()
    }
}

#[derive(Copy, Clone)]
pub struct Grid2WarpScope<const SIZE: usize>;

impl<const SIZE: usize> PrivateTraitGuard for Grid2WarpScope<SIZE> {}

impl<const SIZE: usize> Grid2WarpScope<SIZE> {
    pub const CHECKED_SIZE: u32 = ThreadWarpTile::<SIZE>::CHECKED_SIZE;
}

impl<const SIZE: usize> ChunkScope for Grid2WarpScope<SIZE> {
    type FromScope = Grid;
    type ToScope = ThreadWarpTile<SIZE>;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        // warp memory is shared within a warp and so only each warp has a
        // unique warp id and block id and we use warp id (`subgroup_id`,
        // `block_id_{x,y,z}` to index the group.
        [
            0,
            0,
            Self::ToScope::_subgroup_id(),
            block_id::<DimX>(),
            block_id::<DimY>(),
            block_id::<DimZ>(),
        ]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        if D::DIM_ID == 0 { (block_size() * num_blocks()) / Self::CHECKED_SIZE } else { 1 }
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        if D::DIM_ID == 0 {
            Grid2BlockScope::global_id::<D>(thread_ids) * Self::ToScope::_meta_group_size()
                + thread_ids[2]
        } else {
            0
        }
    }
}

#[derive(Copy, Clone)]
pub struct Block2WarpScope<const SIZE: usize>;

impl<const SIZE: usize> PrivateTraitGuard for Block2WarpScope<SIZE> {}

impl<const SIZE: usize> ChunkScope for Block2WarpScope<SIZE> {
    type FromScope = Block;
    type ToScope = ThreadWarpTile<SIZE>;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        // warp memory is shared within a warp and so only warp_id.
        [0, 0, Self::ToScope::_subgroup_id(), 0, 0, 0]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        if D::DIM_ID == 0 { Self::ToScope::_meta_group_size() } else { 1 }
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        if D::DIM_ID == 0 { thread_ids[2] } else { 0 }
    }
}

#[derive(Copy, Clone)]
pub struct Warp2ThreadScope<const SIZE: usize>;

impl<const SIZE: usize> PrivateTraitGuard for Warp2ThreadScope<SIZE> {}

impl<const SIZE: usize> Warp2ThreadScope<SIZE> {
    pub const CHECKED_SIZE: u32 = ThreadWarpTile::<SIZE>::CHECKED_SIZE;
}

impl<const SIZE: usize> ChunkScope for Warp2ThreadScope<SIZE> {
    type FromScope = ThreadWarpTile<SIZE>;
    type ToScope = Thread;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [u32; TID_MAX_LEN] {
        // warp memory is shared within a warp and so only lane_id.
        [0, Self::FromScope::_thread_rank(), 0, 0, 0, 0]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> u32 {
        if D::DIM_ID == 0 { Self::CHECKED_SIZE } else { 1 }
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        if D::DIM_ID == 0 { thread_ids[1] } else { 0 }
    }
}

#[derive(Clone)]
pub struct ChainedScope<CS1: ChunkScope, CS2: ChunkScope>
where
    CS2: ChunkScope<FromScope = CS1::ToScope>,
{
    _cs1: PhantomData<CS1>,
    _cs2: PhantomData<CS2>,
}

impl<CS1: ChunkScope, CS2: ChunkScope> PrivateTraitGuard for ChainedScope<CS1, CS2> where
    CS2: ChunkScope<FromScope = CS1::ToScope>
{
}
impl<CS1: ChunkScope, CS2: ChunkScope> ChunkScope for ChainedScope<CS1, CS2>
where
    CS2: ChunkScope<FromScope = CS1::ToScope>,
{
    type FromScope = CS1::FromScope;

    type ToScope = CS2::ToScope;

    fn thread_ids() -> [u32; TID_MAX_LEN] {
        let mut ids = CS1::thread_ids();
        let ids2 = CS2::thread_ids();
        for i in 0..TID_MAX_LEN {
            ids[i] += ids2[i];
        }
        ids
    }

    fn global_dim<D: DimType>() -> u32 {
        CS1::global_dim::<D>() * CS2::global_dim::<D>()
    }

    fn global_id<D: DimType>(thread_ids: [u32; TID_MAX_LEN]) -> u32 {
        CS1::global_id::<D>(thread_ids) * CS2::global_dim::<D>() + CS2::global_id::<D>(thread_ids)
    }
}

#[derive(Copy, Clone)]
pub struct ChainedMap<
    CS1: ChunkScope,
    CS2: ChunkScope,
    Map1: ScopeUniqueMap<CS1>,
    Map2: ScopeUniqueMap<CS2>,
> {
    _cs1: PhantomData<CS1>,
    _cs2: PhantomData<CS2>,
    map1: Map1,
    map2: Map2,
}

impl<CS1: ChunkScope, CS2: ChunkScope, Map1: ScopeUniqueMap<CS1>, Map2: ScopeUniqueMap<CS2>>
    ChainedMap<CS1, CS2, Map1, Map2>
where
    CS2: ChunkScope<FromScope = CS1::ToScope>,
{
    pub fn new(m1: Map1, m2: Map2) -> Self {
        Self { _cs1: PhantomData, _cs2: PhantomData, map1: m1, map2: m2 }
    }
}

impl<CS1: ChunkScope, CS2: ChunkScope, Map1: ScopeUniqueMap<CS1>, Map2: ScopeUniqueMap<CS2>>
    PrivateTraitGuard for ChainedMap<CS1, CS2, Map1, Map2>
where
    CS2: ChunkScope<FromScope = CS1::ToScope>,
{
}

/// Chain two mapping strategies.
/// For example,
/// block_size = (32 * 4, 1, 1),
/// MapLinear<Grid2Warp>(64) + MapLinear<Warp2Thread>(2) = MapLinear<Grid2Thread>
/// idx  -> map2(idx) -> map1(map2(idx))
/// 0 -> 0 -> 0
/// 1 -> 1 -> 1
/// 2 -> 64 -> 4 * 64
/// 3 -> 65 -> 4 * 64 + 1
///
unsafe impl<CS1: ChunkScope, CS2: ChunkScope, Map1: ScopeUniqueMap<CS1>, Map2: ScopeUniqueMap<CS2>>
    ScopeUniqueMap<ChainedScope<CS1, CS2>> for ChainedMap<CS1, CS2, Map1, Map2>
where
    CS2: ChunkScope<FromScope = CS1::ToScope>,
    Map1: ScopeUniqueMap<CS1>,
    Map2: ScopeUniqueMap<CS2>,
    Map2::GlobalIndexType: AsPrimitive<Map1::IndexType>,
{
    type IndexType = Map2::IndexType;
    type GlobalIndexType = Map1::GlobalIndexType;

    fn map(
        &self,
        idx: Self::IndexType,
        thread_ids: [u32; TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType) {
        let (valid2, idx2) = self.map2.map(idx, thread_ids);
        let (valid1, idx1) = self.map1.map(idx2.as_(), thread_ids);
        (valid1 & valid2, idx1)
    }
}

#[cfg(any(test, doctest))]
pub mod test {
    use super::*;
    #[derive(Clone)]
    pub struct MockBlock2WarpScope<const SIZE: usize, const WARP_ID: usize, const BLOCK_SIZE: usize>;
    impl<const SIZE: usize, const WARP_ID: usize, const BLOCK_SIZE: usize> PrivateTraitGuard
        for MockBlock2WarpScope<SIZE, WARP_ID, BLOCK_SIZE>
    {
    }
    impl<const SIZE: usize, const WARP_ID: usize, const BLOCK_SIZE: usize> ChunkScope
        for MockBlock2WarpScope<SIZE, WARP_ID, BLOCK_SIZE>
    {
        type FromScope = Block;
        type ToScope = ThreadWarpTile<SIZE>;
        fn thread_ids() -> [u32; TID_MAX_LEN] {
            [0, 0, WARP_ID as u32, 0, 0, 0]
        }
        fn global_dim<D: DimType>() -> u32 {
            if D::DIM_ID == 0 { (BLOCK_SIZE / SIZE) as u32 } else { 1 }
        }
        fn global_id<D: DimType>(ids: [u32; TID_MAX_LEN]) -> u32 {
            if D::DIM_ID == 0 { ids[2] } else { 0 }
        }
    }

    #[derive(Clone)]
    pub struct MockWarp2ThreadScope<const SIZE: usize, const LANE_ID: u32>;
    impl<const SIZE: usize, const LANE_ID: u32> PrivateTraitGuard
        for MockWarp2ThreadScope<SIZE, LANE_ID>
    {
    }
    impl<const SIZE: usize, const LANE_ID: u32> ChunkScope for MockWarp2ThreadScope<SIZE, LANE_ID> {
        type FromScope = ThreadWarpTile<SIZE>;
        type ToScope = Thread;
        fn thread_ids() -> [u32; TID_MAX_LEN] {
            [0, LANE_ID, 0, 0, 0, 0]
        }
        fn global_dim<D: DimType>() -> u32 {
            if D::DIM_ID == 0 { SIZE as _ } else { 1 }
        }
        fn global_id<D: DimType>(ids: [u32; TID_MAX_LEN]) -> u32 {
            if D::DIM_ID == 0 { ids[1] } else { 0 }
        }
    }

    macro_rules! assert_map {
        ($cs:ty, $m:expr, $idx:expr, $thread_ids:expr, $expected:expr) => {
            let (valid, mapped_idx) = ScopeUniqueMap::<$cs>::map(&$m, $idx, $thread_ids);
            assert!(
                valid == $expected.0 && (mapped_idx == $expected.1 || !valid),
                "idx = {}, mapped_idx = {}, valid = {} expected = {:?}",
                $idx,
                mapped_idx,
                valid,
                $expected
            );
        };
    }

    #[test]
    fn test_mocked_scope() {
        type S1 = MockBlock2WarpScope<32, 1, 128>;
        type S2 = MockWarp2ThreadScope<32, 1>;
        let thread_ids = [0, 1, 1, 0, 0, 0];
        assert!(S1::global_dim_x() == 4, "dimx = {}", S1::global_dim_x());
        assert!(S1::global_id_x(thread_ids) == 1, "id_x = {}", S1::global_id_x(thread_ids));
        assert!(S1::global_dim_y() == 1, "dimy = {}", S1::global_dim_y());
        assert!(S1::global_id_y(thread_ids) == 0, "id_y = {}", S1::global_id_y(thread_ids));
        assert!(S1::global_dim_z() == 1, "dimz = {}", S1::global_dim_z());
        assert!(S1::global_id_z(thread_ids) == 0, "id_z = {}", S1::global_id_z(thread_ids));
        assert!(S2::global_dim_x() == 32, "dimx = {}", S2::global_dim_x());
        assert!(S2::global_id_x(thread_ids) == 1, "id_x = {}", S2::global_id_x(thread_ids));
    }

    #[test]
    fn test_chain_map() {
        let map_warps = crate::MapLinear::new(64);
        let map_warp_threads = crate::MapLinear::new(2);
        type S1 = MockBlock2WarpScope<32, 1, 128>;
        type S2 = MockWarp2ThreadScope<32, 1>;

        let chained_map = ChainedMap::<S1, S2, _, _>::new(map_warps, map_warp_threads);

        let thread_ids0 = [0, 0, 0, 0, 0, 0];
        assert_map!(S2, map_warp_threads, 0, thread_ids0, (true, 0));
        assert_map!(S1, map_warps, 0, thread_ids0, (true, 0));
        assert_map!(_, chained_map.clone(), 0, thread_ids0, (true, 0));

        assert_map!(S2, map_warp_threads, 1, thread_ids0, (true, 1));
        assert_map!(S1, map_warps, 1, thread_ids0, (true, 1));
        assert_map!(_, chained_map.clone(), 1, thread_ids0, (true, 1));

        assert_map!(S2, map_warp_threads, 2, thread_ids0, (true, 64));

        let thread_ids = [0, 0, 1, 0, 0, 0];
        assert_map!(S2, map_warp_threads, 0, thread_ids, (true, 0));
        assert_map!(S1, map_warps, 0, thread_ids, (true, 64));
        assert_map!(_, chained_map.clone(), 0, thread_ids, (true, 64));

        assert_map!(S2, map_warp_threads, 1, thread_ids, (true, 1));
        assert_map!(S1, map_warps, 1, thread_ids, (true, 65));
        assert_map!(_, chained_map.clone(), 1, thread_ids, (true, 65));

        let thread_ids = [0, 1, 1, 0, 0, 0];
        assert_map!(S2, map_warp_threads, 0, thread_ids, (true, 2));
        assert_map!(S1, map_warps, 2, thread_ids, (true, 66));
        assert_map!(_, chained_map.clone(), 0, thread_ids, (true, 66));
    }

    #[test]
    fn test_chain_map_lane_0_only() {
        // If the target array has len = 64 * 128/32 = 256,
        // then the mapping should only affect lane 0.
        // lane != 0 will have out of bound error and so developers should avoid using them.

        const BLOCK_SIZE: usize = 128;
        const WIDTH: usize = 64;
        const WARP_SIZE: usize = 32;
        const N: usize = BLOCK_SIZE / WARP_SIZE * WIDTH;
        let map_warps = crate::MapLinear::new(WIDTH);
        let map_warp_threads = crate::MapLinear::new(WIDTH);
        type S1 = MockBlock2WarpScope<WARP_SIZE, 1, BLOCK_SIZE>;
        type S2 = MockWarp2ThreadScope<WARP_SIZE, 1>;

        let chained_map = ChainedMap::<S1, S2, _, _>::new(map_warps, map_warp_threads);

        let thread_ids0 = [0, 0, 0, 0, 0, 0];
        assert_map!(S2, map_warp_threads, 0, thread_ids0, (true, 0));
        assert_map!(S1, map_warps, 0, thread_ids0, (true, 0));
        assert_map!(_, chained_map.clone(), 0, thread_ids0, (true, 0));

        assert_map!(S2, map_warp_threads, 1, thread_ids0, (true, 1));
        assert_map!(S1, map_warps, 1, thread_ids0, (true, 1));
        assert_map!(_, chained_map.clone(), 1, thread_ids0, (true, 1));

        let thread_ids = [0, 1, 1, 0, 0, 0];
        assert_map!(S2, map_warp_threads, 0, thread_ids, (true, WIDTH));
        assert_map!(S1, map_warps, 64, thread_ids, (true, N + WIDTH)); // 128/32 * 64 + 64
        assert_map!(_, chained_map.clone(), 0, thread_ids, (true, N + WIDTH));
    }
}
