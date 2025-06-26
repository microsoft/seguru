#![feature(negative_impls)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

mod print;

use core::marker::PhantomData;

pub use print::{PushPrintfArg, printf};

/// This is motivated by thread::scope in Rust std lib.
pub struct Scope<'scope, 'env: 'scope> {
    /// Invariance over 'scope, to make sure 'scope cannot shrink,
    /// which is necessary for soundness.
    ///
    /// Without invariance, this would compile fine but be unsound:
    ///
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

#[gpu_codegen::builtin(gpu.scope)]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope ThreadScope<'scope, 'env>) -> T,
{
    let scope = ThreadScope { env: PhantomData, scope: PhantomData };
    f(&scope)
}

pub type Tuple3 = [usize; 3];

#[gpu_codegen::builtin(gpu.grid)]
pub fn grid(_: usize, _: usize, _: usize) -> &'static [Tuple3] {
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.block)]
pub fn block(_: usize, _: usize, _: usize) -> &'static [Tuple3] {
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.launch)]
pub fn launch<'scope, F, T>(_: Tuple3, _: Tuple3, _: F)
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.into_iter)]
pub fn into_iter() {}

pub struct ThreadScope<'scope, 'env: 'scope> {
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl<'scope, 'env: 'scope> ThreadScope<'scope, 'env> {
    pub fn launch<F, T>(&self, _: Tuple3, _: Tuple3, _: F)
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        unimplemented!()
    }
}

#[no_mangle]
#[rustfmt::skip]
pub fn sync() {
    // MLIR: gpu.sync
    unimplemented!()
}

pub enum DimType {
    X,
    Y,
    Z,
}

/// Add a string attribute to the MLIR module.
#[gpu_codegen::builtin(add_mlir_string_attr)]
#[gpu_codegen::device]
#[inline(never)]
pub fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.thread_id)]
pub fn thread_id() -> usize {
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.global_thread_id)]
pub fn global_thread_id() -> usize {
    unimplemented!()
}

pub struct GpuChunksMut<'a, T: 'a> {
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for GpuChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[gpu_codegen::builtin(gpu.chunk_next)]
    fn next(&mut self) -> Option<Self::Item> {
        // This is a placeholder implementation.
        // In a real implementation, this would return the next chunk of mutable data.
        unimplemented!()
    }
}
#[gpu_codegen::builtin(gpu.chunks_mut)]
pub fn gpu_chunk_mut<T>(_v: &[T], _chunk_size: usize) -> GpuChunksMut<'_, T> {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice)]
pub fn subslice<T>(_original: &[T], _offset: usize, _window: usize) -> &[T] {
    unimplemented!()
}

#[inline(never)]
#[gpu_codegen::builtin(gpu.subslice_mut)]
pub fn subslice_mut<T>(_original: &mut [T], _offset: usize, _window: usize) -> &mut [T] {
    unimplemented!()
}

/*  TODO: Define shared memory */
