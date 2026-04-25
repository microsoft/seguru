//! OpenedTile: a thread-local view into a GlobalThreadChunk with pre-computed
//! thread-base pointer.
//!
//! # Problem
//! `GlobalGroupChunk::index_mut` calls `map(lid, tid)`, which recomputes the
//! full linear index (struct offset + tid contribution + lid contribution) on
//! every access. In unrolled per-thread tile loops this inflates the register
//! footprint by keeping the tid-derived address-arith temporaries live across
//! the entire tile, hurting occupancy.
//!
//! # Solution
//! `open_tile(self)` computes the tid-invariant pointer once:
//! `tile_ptr = &mut data[thread_base(tid)]`. Each per-access then just adds
//! `map_lid_offset(lid)` — pure lid arithmetic that the compiler can fold
//! into immediate offsets when the unrolled lid is compile-time constant.
//!
//! # Safety
//! `open_tile` consumes the chunk (`self`), so only one `OpenedTile` exists
//! per chunk. The lifetime `'a` on the underlying slice reference is retained
//! so the borrow checker still enforces exclusive access. Per-access bounds
//! are checked using the lid-side validity returned by `map_lid_offset`, and
//! the tid-side validity is checked once at open time. `OpenedRow` follows the
//! same guarded GPU pointer convention as `OpenedTile`: raw pointer offsets are
//! derived from macro-generated map offsets, and the final reference is passed
//! through `assert_ptr` with the combined validity flag expected by the backend.

use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use num_traits::AsPrimitive;

use crate::assert_ptr;
use crate::chunk::{GlobalGroupChunk, MapWithLidOffset, MapWithRows, ScopeUniqueMap};
use crate::chunk_scope::{ChunkScope, Thread};

/// A thread-owned view into a `GlobalGroupChunk` with the tid-invariant
/// portion of the address precomputed.
///
/// Produced by `GlobalGroupChunk::open_tile`. Only available when
/// `ToScope = Thread` (mutable access requirement) and the Map implements
/// `MapWithLidOffset`.
pub struct OpenedTile<'a, T, CS, Map>
where
    CS: ChunkScope,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS>,
{
    /// Base pointer pre-offset to the thread's tile. Keeps the underlying
    /// slice borrow alive via PhantomData below.
    tile_ptr: *mut T,
    /// Combined tid-side validity + struct precondition. Per-access AND's
    /// this with lid-side validity.
    base_ok: bool,
    map_params: Map,
    _data: PhantomData<&'a mut [T]>,
    _cs: PhantomData<CS>,
}

/// A mutable row view borrowed from an `OpenedTile`.
pub struct OpenedRow<'row, T, CS, Map>
where
    CS: ChunkScope,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS> + MapWithRows<CS>,
{
    row_ptr: *mut T,
    row_ok: bool,
    map_params: Map,
    _tile: PhantomData<&'row mut T>,
    _cs: PhantomData<CS>,
}

impl<'a, T, CS, Map> GlobalGroupChunk<'a, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    /// Consume this chunk and return an `OpenedTile` with the tid-invariant
    /// portion of the linear index precomputed as a 64-bit pointer.
    ///
    /// # Safety invariants
    /// - Consumes `self`, so only one OpenedTile exists per chunk.
    /// - Per-access (`IndexMut`) still bounds-checks the lid contribution.
    /// - Tid-side validity and struct precondition are folded into `base_ok`
    ///   and AND'd with the per-access check.
    #[inline(always)]
    #[gpu_codegen::device]
    pub fn open_tile(self) -> OpenedTile<'a, T, CS, Map> {
        let (tb_ok, base) = self.map_params.thread_base(CS::thread_ids());
        let base_usize: usize = base.as_();
        let precondition = self.map_params.precondition();
        // SAFETY: bounds-check folded into `base_ok`; dereference is guarded by
        // `assert_ptr` in `IndexMut::index_mut`. Same lifetime & aliasing
        // guarantees as the underlying `&'a mut [T]` from which it is derived.
        let mut this = self;
        let data_ptr = this.data_ptr_mut();
        let tile_ptr = unsafe { data_ptr.add(base_usize) };
        OpenedTile {
            tile_ptr,
            base_ok: tb_ok & precondition,
            map_params: this.map_params,
            _data: PhantomData,
            _cs: PhantomData,
        }
    }
}

impl<'a, T, CS, Map> OpenedTile<'a, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS> + MapWithRows<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    /// Borrow one row of this opened tile as a mutable in-row view.
    #[inline(always)]
    #[gpu_codegen::device]
    pub fn row_mut<'row>(&'row mut self, row: u32) -> OpenedRow<'row, T, CS, Map> {
        let (row_ok, row_off) = self.map_params.row_lid_offset(row);
        let row_off: usize = row_off.as_();
        let row_ptr = unsafe { self.tile_ptr.add(row_off) };
        OpenedRow {
            row_ptr,
            row_ok: self.base_ok & row_ok,
            map_params: self.map_params.clone(),
            _tile: PhantomData,
            _cs: PhantomData,
        }
    }
}

impl<'a, T, CS, Map> Index<Map::IndexType> for OpenedTile<'a, T, CS, Map>
where
    CS: ChunkScope,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: Map::IndexType) -> &T {
        let (lid_ok, off) = self.map_params.map_lid_offset(idx);
        let off: usize = off.as_();
        // SAFETY: bounds-check guards dereference via assert_ptr (optimizes to
        // a select when DISABLE_GPU_BOUND_CHECK is set).
        let ptr = unsafe { &*self.tile_ptr.add(off) };
        assert_ptr(self.base_ok & lid_ok, ptr)
    }
}

impl<'row, T, CS, Map> Index<u32> for OpenedRow<'row, T, CS, Map>
where
    CS: ChunkScope,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS> + MapWithRows<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    type Output = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, col: u32) -> &T {
        let (col_ok, col_off) = self.map_params.in_row_lid_offset(col);
        let col_off: usize = col_off.as_();
        let ptr = unsafe { &*self.row_ptr.add(col_off) };
        assert_ptr(self.row_ok & col_ok, ptr)
    }
}

impl<'row, T, CS, Map> IndexMut<u32> for OpenedRow<'row, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS> + MapWithRows<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, col: u32) -> &mut T {
        let (col_ok, col_off) = self.map_params.in_row_lid_offset(col);
        let col_off: usize = col_off.as_();
        let ptr = unsafe { &mut *self.row_ptr.add(col_off) };
        assert_ptr(self.row_ok & col_ok, ptr)
    }
}

impl<'a, T, CS, Map> IndexMut<Map::IndexType> for OpenedTile<'a, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
    Map: ScopeUniqueMap<CS> + MapWithLidOffset<CS>,
    Map::GlobalIndexType: AsPrimitive<usize>,
{
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: Map::IndexType) -> &mut T {
        let (lid_ok, off) = self.map_params.map_lid_offset(idx);
        let off: usize = off.as_();
        // SAFETY: exclusivity guaranteed by consuming the chunk in open_tile;
        // bounds-check via assert_ptr.
        let ptr = unsafe { &mut *self.tile_ptr.add(off) };
        assert_ptr(self.base_ok & lid_ok, ptr)
    }
}
