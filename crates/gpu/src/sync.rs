//! Synchronization and atomic operations for GPU programming.

use crate::GpuGlobal;

/// Synchronization within a thread block.
/// Disallow diversed control flow to
/// ensure all threads in a block can reach the sync point to avoid deadlock.
#[inline(never)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::sync_threads"]
#[gpu_codegen::sync_data]
pub fn sync_threads() {
    unimplemented!();
}

/// Define a type to represent the atomic RMW kind from
/// [MLIR ArithBase](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td)
trait AtomicRMWKind {
    const NAME: &str;
}

/// To prevent the user from mixing use of
/// Atomic operation and chunk-based access,
/// we wrap the reference to the data to be modified in an Atomic struct.
/// ensuring that the user cannot access the data without using atomic operations.
/// If user wants to repurpose the data for non-atomic access,
/// they need to drop the Atomic struct first and need a sync across all blocks.
/// For now, sync across all blocks is not supported, we should reject such code in analysis.
/// TODO: Avoid repurposing data for atomic or chunk-based access.
#[rustc_diagnostic_item = "gpu::sync::Atomic"]
pub struct Atomic<'a, T: ?Sized> {
    data: &'a T,
}

impl<'a, T> Atomic<'a, [T]> {
    /// Get a reference to the data inside the Atomic struct.
    /// This is unsafe because the user must ensure that no other thread
    /// is accessing the data at the same time.
    #[inline(always)]
    #[gpu_codegen::device]
    pub fn index(&'a self, i: usize) -> Atomic<'a, T> {
        Atomic { data: &self.data[i] }
    }
}

#[inline(never)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::atomic_rmw"]
unsafe fn _atomic_rmw<T: num_traits::Num>(_mem: &T, _val: T, _kind: &'static str) -> T {
    unimplemented!()
}

/// Atomic read-modify-write operation with kind defined in
/// [MLIR memref::atomic_rmw](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefatomic_rmw-memrefatomicrmwop)
impl<'a, T: ?Sized> Atomic<'a, T> {
    #[inline(never)]
    #[gpu_codegen::device]
    #[rustc_diagnostic_item = "gpu::sync::Atomic::new"]
    pub fn new(data: GpuGlobal<'a, T>) -> Atomic<'a, T> {
        Self { data: data.data }
    }

    #[inline(always)]
    #[gpu_codegen::device]
    #[expect(private_bounds)]
    pub fn atomic_rmw<K: AtomicRMWKind>(&self, val: T) -> T
    where
        T: num_traits::Num,
    {
        unsafe { _atomic_rmw(self.data, val, K::NAME) }
    }
}

macro_rules! def_atomic_rmw_kind {
    ($t:ident, $val:literal, $atomic_fn: ident, $trait:path) => {
        #[doc = concat!("Atomic operation kind for [`", stringify!($atomic_fn), "`]")]
        pub struct $t;
        impl AtomicRMWKind for $t {
            const NAME: &str = concat!(stringify!($val), ": i64");
        }

        impl<'a, T: num_traits::Num> Atomic<'a, T> {
            #[doc = concat!("Equivalent to: atomic_rmw::<[`", stringify!($t), "`]>")]
            #[inline(always)]
            #[gpu_codegen::device]
            pub fn $atomic_fn(&self, val: T) -> T
            where
                T: $trait,
            {
                self.atomic_rmw::<$t>(val)
            }
        }
    };
}

macro_rules! def_atomic_rmw_kinds {
    ($($t:ident, $val:literal, $atomic_fn: ident, $trait:path);* $(;)?) => {
        $(
            def_atomic_rmw_kind!($t, $val, $atomic_fn, $trait);
        )*
    };
}

// Define AtomicRMWKind
// mlir/include/mlir/Dialect/Arith/IR/ArithBase.td
def_atomic_rmw_kinds!(
    AddF, 0, atomic_addf, num_traits::Float;
    AddI, 1, atomic_addi, num_traits::PrimInt;
    Assign, 2, atomic_assign, num_traits::Num;
    MaximumF, 3, atomic_maximumf, num_traits::Float;
    MaxS, 4, atomic_maxs, num_traits::Signed;
    MaxU, 5, atomic_maxu, num_traits::Unsigned;
    MinimumF, 6, atomic_minimumf, num_traits::Float;
    MinS, 7, atomic_mins, num_traits::Signed;
    MinU, 8, atomic_minu, num_traits::Unsigned;
    MulF, 9, atomic_mulf, num_traits::Float;
    MulI, 10, atomic_muli, num_traits::PrimInt;
    OrI, 11, atomic_ori, num_traits::PrimInt;
    AndI, 12, atomic_andi, num_traits::PrimInt;
    MaxNumF, 13, atomic_maxnumf, num_traits::Float;
    MinNumF, 14, atomic_minnumf, num_traits::Float;
);
