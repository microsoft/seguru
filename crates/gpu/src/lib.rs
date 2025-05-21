#![feature(negative_impls)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
use core::marker::PhantomData;

pub struct Scope<'scope, 'env: 'scope> {
    /// Invariance over 'scope, to make sure 'scope cannot shrink,
    /// which is necessary for soundness.
    ///
    /// Without invariance, this would compile fine but be unsound:
    ///
    /// ```compile_fail,E0373
    /// gpu::thread::scope(|s| {
    ///     s.spawn(|| {
    ///         let a = String::from("abcd");
    ///         s.spawn(|| println!("{a:?}")); // might run after `a` is dropped
    ///     });
    /// });
    /// ```
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

#[gpu_codegen::builtin(gpu.scope)]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope ThreadScope<'scope, 'env>) -> T,
{
    let scope = ThreadScope {
        env: PhantomData,
        scope: PhantomData,
    };
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

pub struct Thread;
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

impl Thread {
    // Returns the block id, i.e. the index of the current block within the grid along the x, y, or z dimension.
    // MLIR: gpu.block_id
    #[no_mangle]
    pub fn block_id(&self) -> Dim {
        unimplemented!()
    }

    // Returns the number of threads in the thread block (aka the block size) along the x, y, or z dimension.
    // MLIR: gpu.block_dim
    #[no_mangle]
    pub fn block_dim(&self) -> Dim {
        unimplemented!()
    }

    // Returns the number of thread blocks in the grid along the x, y, or z dimension.
    #[no_mangle]
    pub fn grid_dim(&self) -> Dim {
        unimplemented!()
    }

    // Returns the thread id, i.e. the index of the current thread within the block along the x, y, or z dimension.
    // MLIR: gpu.block_dim
    #[no_mangle]
    pub fn local_thread(&self) -> Dim {
        unimplemented!()
    }

    #[no_mangle]
    pub fn global_invocation(&self) -> Dim {
        Dim {
            x: self.block_id().x * self.block_dim().x + self.local_thread().x,
            y: self.block_id().y * self.block_dim().y + self.local_thread().y,
            z: self.block_id().z * self.block_dim().z + self.local_thread().z,
        }
    }

    #[no_mangle]
    pub fn local_thread_id(&self) -> usize {
        self.local_thread().x
            + self.local_thread().y * self.block_dim().x
            + self.local_thread().z * self.block_dim().x * self.block_dim().y
    }

    pub fn global_thread_id(&self) -> usize {
        self.global_invocation().x
            + self.global_invocation().y * self.grid_dim().x
            + self.global_invocation().z * self.grid_dim().x * self.grid_dim().y
    }
}

#[no_mangle]
#[rustfmt::skip]
pub fn sync() {
    // MLIR: gpu.sync
    unimplemented!()
}

#[no_mangle]
pub fn thread2() -> Thread {
    unimplemented!()
}

pub enum DimType {
    X,
    Y,
    Z,
}

#[gpu_codegen::builtin(add_mlir_string_attr)]
pub fn add_mlir_string_attr(_: &'static str) -> usize {
    unimplemented!()
}

#[gpu_codegen::builtin(gpu.printf)]
pub fn printf() -> usize {
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

pub struct SharedWriteGuard<'a, T: ?Sized + 'a> {
    dummy: PhantomData<T>,
    _guard: &'a (),
}

impl<T: ?Sized> !Send for SharedWriteGuard<'_, T> {}

unsafe impl<T: ?Sized + Sync> Sync for SharedWriteGuard<'_, T> {}

impl<T: ?Sized> core::ops::Deref for SharedWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unimplemented!()
    }
}

impl<T: ?Sized> core::ops::DerefMut for SharedWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unimplemented!()
    }
}

pub struct SharedReadGuard<'a, T: ?Sized + 'a> {
    dummy: PhantomData<T>,
    _guard: &'a (),
}

impl<T: ?Sized> !Send for SharedReadGuard<'_, T> {}

unsafe impl<T: ?Sized + Sync> Sync for SharedReadGuard<'_, T> {}

impl<T: ?Sized> core::ops::Deref for SharedReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy)]

pub struct Shared<T> {
    data: PhantomData<T>,
}

impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Shared<T> {
    pub fn new() -> Self {
        unimplemented!()
    }

    pub fn write(&self) -> SharedWriteGuard<'_, T> {
        unimplemented!()
    }

    pub fn read(&self) -> SharedReadGuard<'_, T> {
        unimplemented!()
    }
}
