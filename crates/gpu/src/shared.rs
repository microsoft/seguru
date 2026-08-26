use core::marker::PhantomData;
/// This file contains shared memory related APIs.
///
use core::ops::{Deref, DerefMut};

#[cfg(not(feature = "codegen_tests"))]
use cuda_bindings::SafeGpuConfig;
use num_traits::AsPrimitive;

use crate::assert_ptr;
use crate::cg::Block;
use crate::chunk::{ScopeUniqueMap, ScopeUniqueMapProvidedMethods};
use crate::chunk_scope::{Block2ThreadScope, ChainedMap, ChainedScope, ChunkScope, Thread};

/// Static GPU shared memory.
/// NVCC always aligns shared memory to 16 bytes,
/// so we also align to 16 bytes here.
#[rustc_diagnostic_item = "gpu::GpuShared"]
#[repr(C, align(16))]
pub struct GpuShared<T: ?Sized> {
    value: T,
}

impl<T> GpuShared<T> {
    /// Allocate static shared memory **without initialising it**.
    ///
    /// # Safety
    ///
    /// The returned storage is **uninitialised**. CUDA `__shared__` memory has no
    /// static-initialiser semantics: it is uninitialised at block start and the
    /// same physical storage is reused by later blocks and later kernel launches,
    /// so it commonly contains another block's leftovers rather than zeros.
    ///
    /// The caller must ensure every element is written before it is read. When the
    /// initialising writes are performed cooperatively by the block, they must be
    /// followed by [`crate::sync::sync_threads`] before any thread reads an element
    /// written by another thread.
    ///
    /// Prefer [`GpuShared::new`], which does this correctly and is safe.
    #[rustc_diagnostic_item = "gpu::new_shared_mem"]
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data]
    #[inline(never)]
    pub const unsafe fn uninit() -> Self {
        unimplemented!();
    }
}

impl<E: Copy, const N: usize> GpuShared<[E; N]> {
    /// Cooperatively initialise every element to `v`.
    ///
    /// Each thread initialises `PER_THREAD` elements, so `PER_THREAD` must be
    /// `N / block_size` rounded up.
    ///
    /// The caller **must** call [`sync_threads`] afterwards, before any thread
    /// reads an element written by another thread. This function deliberately does
    /// not do it itself: the data-divergence analysis is intra-procedural, so a
    /// `sync_threads()` inside this callee is not credited to the caller and the
    /// caller is still rejected with `MissingSyncThreads`.
    ///
    /// Prefer [`shared_init!`], which pairs the allocation, this call and the
    /// barrier into one safe statement.
    #[gpu_codegen::device]
    #[inline(always)]
    pub fn init<const PER_THREAD: usize>(&mut self, v: E) {
        {
            let mut chunk = self.chunk_mut(crate::chunk_impl::MapLinear::new(1));
            for i in 0..PER_THREAD {
                chunk[i] = v;
            }
        }
    }
}

/// Allocate static shared memory and cooperatively initialise every element,
/// then synchronise. This is the safe replacement for the old `GpuShared::zero()`.
///
/// `$per_thread` must be `N / block_size` rounded up: each thread initialises that
/// many elements. The expansion ends with [`sync_threads`], so on return every
/// element is initialised and visible block-wide.
///
/// Because it is a block-wide collective, **every thread of the block must reach
/// it**; using it in thread-divergent control flow will deadlock, exactly as a
/// bare `sync_threads()` would.
///
/// The per-element work itself lives in the ordinary function [`GpuShared::init`];
/// this macro only bundles it with the allocation and the barrier.
///
/// It is a statement macro that declares `$name`, rather than a function returning
/// a fresh `GpuShared`, because the allocation handle must not be moved after it is
/// bound. `uninit()` is `#[inline(never)]` and its call site *is* the allocation, so
/// binding the result and then moving it again makes the backend emit a *second*
/// shared allocation - observed as a ptxas `uses too much shared data` failure at
/// exactly 2x32 KB in the NTT kernels. Taking `&mut self`, as [`GpuShared::init`]
/// does, involves no move and works fine.
///
/// ```ignore
/// gpu::shared_init!(smem: [u64; 4096], 8, 0u64);
/// // `smem` is now a zeroed `GpuShared<[u64; 4096]>`.
/// ```
#[macro_export]
macro_rules! shared_init {
    ($name:ident : $ty:ty, $per_thread:expr, $init:expr) => {
        // SAFETY: `init` writes every element exactly once across the block, and
        // the `sync_threads()` below publishes those writes before `$name` is read.
        let mut $name = unsafe { $crate::GpuShared::<$ty>::uninit() };
        $name.init::<{ $per_thread }>($init);
        $crate::sync_threads();
    };
}

impl<T> Deref for GpuShared<T> {
    type Target = T;

    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> !DerefMut for GpuShared<T> {}

/// Dynamic GPU shared memory allocation.
#[allow(dead_code)]
pub struct DynamicSharedAlloc {
    size: usize,
}

impl DynamicSharedAlloc {
    #[rustc_diagnostic_item = "gpu::base_dynamic_shared"]
    #[inline(never)]
    #[gpu_codegen::memspace_shared(1000)]
    unsafe fn base_ptr() -> *const u8 {
        unimplemented!()
    }

    #[gpu_codegen::device]
    #[inline(always)]
    #[gpu_codegen::memspace_shared(1000)]
    #[gpu_codegen::sync_data(1)] // len is non-divergent
    #[gpu_codegen::ret_sync_data(1000)] // return pointer is divergent
    pub fn alloc<T: Sized>(&mut self, len: usize) -> &'static mut GpuShared<[T]> {
        let size = core::mem::size_of::<T>() * len;
        assert!(size <= self.size);
        self.size -= size;
        unsafe {
            let raw = core::intrinsics::offset(Self::base_ptr(), self.size);
            &mut *(core::ptr::slice_from_raw_parts_mut(raw as *mut T, len) as *const [T]
                as *mut GpuShared<[T]>)
        }
    }
}

/// This trait is implemented for kernel config struct to provide dynamic shared memory allocation.
pub trait DynamicSharedAllocBuilder {
    fn smem_alloc(&self) -> DynamicSharedAlloc;
}

#[cfg(not(feature = "codegen_tests"))]
impl<Config: SafeGpuConfig> DynamicSharedAllocBuilder for Config {
    // This is host-side function.
    fn smem_alloc(&self) -> DynamicSharedAlloc {
        DynamicSharedAlloc { size: self.shared_size() as usize }
    }
}

#[cfg(not(feature = "codegen_tests"))]
unsafe impl cuda_bindings::AsHostKernelParams for DynamicSharedAlloc {
    fn as_kernel_param_data(&self, args: &mut alloc::vec::Vec<*mut ::core::ffi::c_void>) {
        args.push(self as *const _ as _);
    }
}

impl<T> core::ops::Index<usize> for GpuShared<[T]> {
    type Output = GpuShared<T>;

    #[inline(always)]
    #[gpu_codegen::device]
    fn index(&self, idx: usize) -> &GpuShared<T> {
        unsafe { &*((&self.value[idx]) as *const _ as *const GpuShared<T>) }
    }
}

/// N:core::ops::Index dimension, 1, 2, 3
/// Map: Mapping strategy
#[allow(private_bounds)]
pub struct SMemThreadChunk<'a, T: ?Sized + AsSharedSlice, CS: ChunkScope, Map: ScopeUniqueMap<CS>> {
    data: &'a mut GpuShared<T>, // Must be private.
    pub map_params: Map,
    dummy: core::marker::PhantomData<CS>,
}

impl<'a, T: ?Sized + AsSharedSlice, CS: ChunkScope, Map: ScopeUniqueMap<CS>>
    SMemThreadChunk<'a, T, CS, Map>
{
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    #[gpu_codegen::sync_data(1, 2)] // self is guaranteed to be non-divergent and so no check is required
    pub fn chunk_to_scope<CS2: ChunkScope, Map2: ScopeUniqueMap<CS2>>(
        self,
        _scope: CS2,
        map_params: Map2,
    ) -> SMemThreadChunk<'a, T, ChainedScope<CS, CS2>, ChainedMap<CS, CS2, Map, Map2>>
    where
        Map: ScopeUniqueMap<CS>,
        CS: ChunkScope<ToScope = CS2::FromScope>,
        Map2::GlobalIndexType: AsPrimitive<Map::IndexType>,
    {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        SMemThreadChunk {
            data: self.data,
            map_params: ChainedMap::new(self.map_params, map_params),
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
}

trait PrivateTraitGuard {}

#[expect(private_bounds)]
pub trait AsSharedSlice: PrivateTraitGuard {
    type Elem;
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];

    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_slice(&self) -> &[Self::Elem];
}

impl<T> PrivateTraitGuard for [T] {}
impl<T> AsSharedSlice for [T] {
    type Elem = T;
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }

    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<T, const N: usize> PrivateTraitGuard for [T; N] {}
impl<T, const N: usize> AsSharedSlice for [T; N] {
    type Elem = T;
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }

    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<T: ?Sized + AsSharedSlice> GpuShared<T> {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    #[gpu_codegen::sync_data(0, 1)]
    #[gpu_codegen::ret_sync_data(0, 1000)]
    #[rustc_diagnostic_item = "gpu::shared_chunk_mut"]
    pub fn chunk_mut<'a, Map: ScopeUniqueMap<Block2ThreadScope>>(
        &'a mut self,
        map_params: Map,
    ) -> SMemThreadChunk<'a, T, Block2ThreadScope, Map> {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        SMemThreadChunk { data: self, map_params, dummy: PhantomData }
    }

    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(0, 1000)]
    #[gpu_codegen::sync_data(0, 1, 2)]
    #[gpu_codegen::ret_sync_data(0, 1000)]
    pub fn chunk_to_scope<'a, CS, Map: ScopeUniqueMap<CS>>(
        &'a mut self,
        _scope: CS,
        map_params: Map,
    ) -> SMemThreadChunk<'a, T, CS, Map>
    where
        CS: ChunkScope<FromScope = Block>,
    {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        SMemThreadChunk { data: self, map_params, dummy: PhantomData }
    }
}

impl<'a, T: ?Sized + AsSharedSlice, CS: ChunkScope, Map: ScopeUniqueMap<CS>>
    core::ops::Index<Map::IndexType> for SMemThreadChunk<'a, T, CS, Map>
{
    type Output = T::Elem;

    #[inline(always)]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(1000)]
    fn index(&self, idx: Map::IndexType) -> &Self::Output {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let idx = idx.as_();
        let valid = self.map_params.precondition() & idx_precondition;
        assert_ptr(valid, &self.data.value.as_slice()[idx])
    }
}

impl<'a, T: ?Sized + AsSharedSlice, CS: ChunkScope, Map: ScopeUniqueMap<CS>>
    core::ops::IndexMut<Map::IndexType> for SMemThreadChunk<'a, T, CS, Map>
where
    CS: ChunkScope<ToScope = Thread>,
{
    #[inline(always)]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(1000)]
    fn index_mut(&mut self, idx: Map::IndexType) -> &mut Self::Output {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let idx = idx.as_();
        let valid = self.map_params.precondition() & idx_precondition;
        assert_ptr(valid, &mut self.data.value.as_mut_slice()[idx])
    }
}
