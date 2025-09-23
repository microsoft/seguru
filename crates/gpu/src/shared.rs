/// This file contains shared memory related APIs.
///
use core::ops::{Deref, DerefMut};

use crate::assert_ptr;
use crate::chunk::{ScopeUniqueMap, ScopeUniqueMapProvidedMethods};
use crate::chunk_scope::Block2ThreadScope;

/// Static GPU shared memory.
#[rustc_diagnostic_item = "gpu::GpuShared"]
pub struct GpuShared<T: ?Sized> {
    value: T,
}

impl<T> GpuShared<T> {
    #[rustc_diagnostic_item = "gpu::new_shared_mem"]
    #[gpu_codegen::device]
    #[gpu_codegen::sync_data]
    #[inline(never)]
    pub const fn zero() -> Self {
        unimplemented!();
    }
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
    pub fn alloc<T: Sized>(&mut self, len: usize) -> &mut GpuShared<[T]> {
        let size = core::mem::size_of::<T>() * len;
        let (remain_size, len) =
            if size < self.size { (self.size - size, len) } else { (self.size, 0) };
        self.size = remain_size;

        unsafe {
            let raw = core::intrinsics::offset(Self::base_ptr(), remain_size);
            &mut *(core::ptr::slice_from_raw_parts_mut(raw as *mut T, len) as *const [T]
                as *mut GpuShared<[T]>)
        }
    }
}

/// TODO(gpu): Check the use of DynamicSharedAlloc in kernel entry to ensure it
/// is owned by the kernel and thus we will use it as a local variable.
/// TODO(host): Link the host-side dynamic memory size with DynamicSharedAlloc.
impl DynamicSharedAlloc {
    /// Host-side constructor for shared memory allocation.
    pub fn new(size: usize) -> Self {
        Self { size }
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
pub struct SMemThreadChunk<'a, T: ?Sized + AsSharedSlice, Map: ScopeUniqueMap<Block2ThreadScope>> {
    data: &'a mut GpuShared<T>, // Must be private.
    pub map_params: Map,
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
    #[rustc_diagnostic_item = "gpu::shared_chunk_mut"]
    pub fn chunk_mut<'a, Map: ScopeUniqueMap<Block2ThreadScope>>(
        &'a mut self,
        map_params: Map,
    ) -> SMemThreadChunk<'a, T, Map> {
        if !map_params.precondition() {
            core::intrinsics::abort();
        }
        SMemThreadChunk { data: self, map_params }
    }
}

impl<'a, T: ?Sized + AsSharedSlice, Map: ScopeUniqueMap<Block2ThreadScope>>
    core::ops::Index<Map::IndexType> for SMemThreadChunk<'a, T, Map>
{
    type Output = T::Elem;

    #[inline(always)]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(1000)]
    fn index(&self, idx: Map::IndexType) -> &Self::Output {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let valid = self.map_params.precondition() & idx_precondition;
        assert_ptr(valid, &self.data.value.as_slice()[idx])
    }
}

impl<'a, T: ?Sized + AsSharedSlice, Map: ScopeUniqueMap<Block2ThreadScope>>
    core::ops::IndexMut<Map::IndexType> for SMemThreadChunk<'a, T, Map>
{
    #[inline(always)]
    #[gpu_codegen::device]
    #[gpu_codegen::memspace_shared(1000)]
    fn index_mut(&mut self, idx: Map::IndexType) -> &mut Self::Output {
        let (idx_precondition, idx) = self.map_params.local_to_global_index(idx);
        let valid = self.map_params.precondition() & idx_precondition;
        assert_ptr(valid, &mut self.data.value.as_mut_slice()[idx])
    }
}
