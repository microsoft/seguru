/// This is motivated by thread::scope in Rust std lib.
/// Be careful when modifying this code since it is part of the TCB to support data-race free GPU programming.
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

pub struct ThreadScope<'scope, 'env: 'scope> {
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

#[rustc_diagnostic_item = "gpu::scope"]
#[gpu_codegen::device]
#[inline(never)]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope ThreadScope<'scope, 'env>) -> T + Clone + Send,
{
    f(&ThreadScope { scope: PhantomData, env: PhantomData })
}

/// No Copy or Clone
/// Ensures that the use of GpuChunksMut does not violate both CPU and GPU concurrency rules.
pub struct GpuChunksMut<'a, T> {
    _ptr: &'a mut [T],
    _chunk_size: usize,
    _idx: crate::GpuChunkIdx,
}

// We may allow global memory chunking via GpuChunksMut or subslice_mut?
// However, for shared memory, we want to ensure we inserted the _thread_sync() before changing chunks.
// We also want to allow the use of shared memory for readonly outside of the scope and thus it is better to disallow
// subslice_mut for shared memory.
impl<'env, T> GpuChunksMut<'env, T> {
    /// _thread_scope is a dummy parameter to ensure that
    /// this chunk is only accessible within the scope of the thread.
    #[inline(never)]
    #[rustc_diagnostic_item = "gpu::GpuChunksMut::unique_chunk"]
    #[gpu_codegen::device]
    pub fn unique_chunk<'scope>(
        &self,
        _thread_scope: &'scope ThreadScope<'scope, 'env>,
    ) -> &'scope mut [T] {
        unimplemented!();
    }

    #[rustc_diagnostic_item = "gpu::GpuChunksMut::new"]
    #[gpu_codegen::device]
    #[inline(never)]
    pub fn new(
        ptr: &'env mut [T],
        chunk_size: usize,
        idx: crate::GpuChunkIdx,
    ) -> GpuChunksMut<'env, T> {
        GpuChunksMut { _ptr: ptr, _chunk_size: chunk_size, _idx: idx }
    }
}

#[rustc_diagnostic_item = "gpu::GpuShared"]
pub struct GpuShared<T: Sized> {
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
    #[inline(never)]
    pub const fn zero() -> Self {
        unimplemented!();
    }
}

impl<T> Deref for GpuShared<T> {
    type Target = T;

    #[inline(always)]
    #[gpu_codegen::device]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for GpuShared<T> {
    #[inline(always)]
    #[gpu_codegen::device]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[inline(always)]
#[gpu_codegen::device]
pub fn chunk_mut<T>(original: &mut [T], window: usize, idx: crate::GpuChunkIdx) -> &mut [T] {
    let offset = idx.as_usize() * window;
    crate::subslice_mut(original, offset, window)
}

#[inline(never)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::sync_threads"]
pub fn sync_threads() {
    unimplemented!();
}
