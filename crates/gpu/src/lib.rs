#![feature(negative_impls)]
use core::marker::PhantomData;
use std::{ops::Deref, thread::ScopedJoinHandle};

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

impl<'scope, 'env: 'scope> ThreadScope<'scope, 'env> {
    pub fn spawn<F, T>(self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        unimplemented!()
    }
}

pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(
        &'scope std::thread::Scope<'scope, 'env>,
        Vec<Vec<ThreadScope<'scope, 'env>>>,
    ) -> T,
{
    unimplemented!()
}

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
    fn thread(&self) -> Thread {
        unimplemented!()
    }
}

impl Thread {
    // Returns the block id, i.e. the index of the current block within the grid along the x, y, or z dimension.
    // MLIR: gpu.block_id
    pub fn block_id(&self) -> Dim {
        unimplemented!()
    }

    // Returns the number of threads in the thread block (aka the block size) along the x, y, or z dimension.
    // MLIR: gpu.block_dim
    pub fn block_dim(&self) -> Dim {
        unimplemented!()
    }

    // Returns the number of thread blocks in the grid along the x, y, or z dimension.
    pub fn grid_dim(&self) -> Dim {
        unimplemented!()
    }

    // Returns the thread id, i.e. the index of the current thread within the block along the x, y, or z dimension.
    // MLIR: gpu.block_dim
    pub fn local_thread(&self) -> Dim {
        unimplemented!()
    }

    pub fn global_invocation(&self) -> Dim {
        Dim {
            x: self.block_id().x * self.block_dim().x + self.local_thread().x,
            y: self.block_id().y * self.block_dim().y + self.local_thread().y,
            z: self.block_id().z * self.block_dim().z + self.local_thread().z,
        }
    }

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

pub fn sync() {
    // MLIR: gpu.sync
    unimplemented!()
}

pub fn thread() -> Thread {
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
