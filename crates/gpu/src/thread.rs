/// This is motivated by thread::scope in Rust std lib.
use core::marker::PhantomData;

pub struct ThreadScope<'scope, 'env: 'scope> {
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

#[gpu_codegen::builtin(gpu.scope)]
#[gpu_codegen::device]
#[inline(never)]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope ThreadScope<'scope, 'env>) -> T + Clone + Send,
{
    f(&ThreadScope { scope: PhantomData, env: PhantomData })
}

#[derive(Clone, Copy)]
pub struct GpuChunksMut<'a, T: 'a> {
    _ptr: &'a [T],
    _chunk_size: usize,
    _idx: crate::GpuChunkIdx,
}

// Assume T::index(i) function is injective
impl<'a, T> GpuChunksMut<'a, T> {
    #[inline(never)]
    #[gpu_codegen::builtin(gpu.next)]
    #[gpu_codegen::device]
    pub fn next(&self) -> &'a mut [T] {
        unimplemented!()
    }
}

#[gpu_codegen::device]
#[inline(never)]
pub fn gpu_chunk_mut<'a, T>(
    _ptr: &'a mut [T],
    _chunk_size: usize,
    _idx: crate::GpuChunkIdx,
) -> GpuChunksMut<'a, T> {
    GpuChunksMut { _ptr, _chunk_size, _idx }
}
