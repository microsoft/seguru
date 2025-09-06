//! Synchronization and atomic operations for GPU programming.

/// Synchronization within a thread block.
#[inline(never)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::sync_threads"]
pub fn sync_threads() {
    unimplemented!();
}

/// Define a type to represent the atomic RMW kind from
/// [MLIR ArithBase](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Arith/IR/ArithBase.td)
trait AtomicRMWKind {
    const NAME: &str;
}

#[inline(never)]
#[gpu_codegen::device]
#[rustc_diagnostic_item = "gpu::atomic_rmw"]
fn _atomic_rmw<T: num_traits::Num>(_mem: &mut T, _val: T, _kind: &'static str) -> T {
    unimplemented!()
}

/// Atomic read-modify-write operation with kind defined in
/// [MLIR memref::atomic_rmw](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefatomic_rmw-memrefatomicrmwop)
#[expect(private_bounds)]
pub fn atomic_rmw<K: AtomicRMWKind, T: num_traits::Num>(mem: &mut T, val: T) -> T {
    _atomic_rmw(mem, val, K::NAME)
}

macro_rules! def_atomic_rmw_kind {
    ($t:ident, $val:literal, $atomic_fn: ident, $trait:path) => {
        #[doc = concat!("Atomic operation kind for [`", stringify!($atomic_fn), "`]")]
        pub struct $t;
        impl AtomicRMWKind for $t {
            const NAME: &str = concat!(stringify!($val), ": i64");
        }

        #[doc = concat!("Equivalent to: atomic_rmw::<[`", stringify!($t), "`]>")]
        #[inline(always)]
        #[gpu_codegen::device]
        pub fn $atomic_fn<T: $trait>(mem: &mut T, val: T) -> T {
            atomic_rmw::<$t, T>(mem, val)
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
