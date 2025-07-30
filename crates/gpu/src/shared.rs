/// This file contains shared memory related APIs.
///
use core::ops::{Deref, DerefMut};

/// Static GPU shared memory.
#[rustc_diagnostic_item = "gpu::GpuShared"]
pub struct GpuShared<T: ?Sized> {
    value: T,
}

impl<T: Copy> Copy for GpuShared<T> {}

impl<T: Copy> Clone for GpuShared<T> {
    #[inline]
    fn clone(&self) -> GpuShared<T> {
        *self
    }
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
    #[gpu_codegen::ret_shared]
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for GpuShared<T> {
    // The returned type is not GPUShared and need explicitly tell the compiler
    // that the memory space is 3
    #[gpu_codegen::device]
    #[gpu_codegen::ret_shared]
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// Dynamic GPU shared memory allocation.
#[allow(dead_code)]
pub struct DynamicSharedAlloc {
    size: usize,
}

impl DynamicSharedAlloc {
    #[rustc_diagnostic_item = "gpu::base_dynamic_shared"]
    #[inline(never)]
    unsafe fn base_ptr() -> *const u8 {
        unimplemented!()
    }

    #[gpu_codegen::device]
    #[inline(always)]
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

impl<T> core::ops::IndexMut<usize> for GpuShared<[T]> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn index_mut(&mut self, idx: usize) -> &mut GpuShared<T> {
        unsafe { &mut *((&mut self.value[idx]) as *mut _ as *mut GpuShared<T>) }
    }
}

impl<T, const N: usize> GpuShared<[T; N]> {
    #[gpu_codegen::device]
    #[gpu_codegen::ret_shared]
    #[gpu_codegen::sync_data(0, 1)]
    #[inline(always)]
    pub fn chunk_mut(&mut self, window: usize, idx: super::GpuSharedChunkIdx) -> &mut [T] {
        let offset = idx.as_usize() * window;
        unsafe { crate::subslice_mut(&mut self.value, offset, window) }
    }
}
impl<T> GpuShared<[T]> {
    #[gpu_codegen::device]
    #[gpu_codegen::ret_shared]
    #[gpu_codegen::sync_data(0, 1)]
    #[inline(always)]
    pub fn chunk_mut(&mut self, window: usize, idx: super::GpuSharedChunkIdx) -> &mut [T] {
        let offset = idx.as_usize() * window;
        unsafe { crate::subslice_mut(&mut self.value, offset, window) }
    }
}
