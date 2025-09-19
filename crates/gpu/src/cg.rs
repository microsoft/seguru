use crate::dim::{lane_id, warp_id};

/// Similar a thread block tile in a GPU kernel.
/// But the SIZE <= warp size (e.g., 32 for NVIDIA GPUs).
/// If SIZE = 8, stride = 4, then the clusters will be:
/// [0, 4, 8, 12, 16, 20, 24, 28]
/// [1, 5, 9, 13, 17, 21, 25, 29]
/// [2, 6, 10, 14, 18, 22, 26, 30]
/// [3, 7, 11, 15, 19, 23, 27, 31]
/// If SIZE = 8, stride = 1, then the clusters will be:
/// [0, 1, 2, 3, 4, 5, 6, 7]
/// [8, 9, 10, 11, 12, 13, 14, 15]
/// [16, 17, 18, 19, 20, 21, 22, 23]
/// [24, 25, 26, 27, 28, 29, 30, 31]
#[derive(Copy, Clone)]
pub struct ThreadWarpTile<const SIZE: usize = 32, const STRIDE: usize = 1>;

trait ReduxKind {}

trait NvvmReduxSyncKind<T>: ReduxKind {
    const KIND: &'static str;
}

trait SubGroupReduceKind<T>: ReduxKind {
    const SUB_GROUP_KIND: &'static str;
}

macro_rules! impl_redux_kind {
    ($name: ident, $($kind: ident $type: ty),+) => {
        pub struct $name;
        $(impl NvvmReduxSyncKind<$type> for $name {
            const KIND: &'static str = concat!("#nvvm<redux_kind ", stringify!($kind), ">");
        })*
        impl ReduxKind for $name {
        }
    };
}

macro_rules! impl_subgroup_kind {
    ($name: ident, $($kind: ident $type: ty),+) => {
        $(impl SubGroupReduceKind<$type> for $name {
            const SUB_GROUP_KIND: &'static str = concat!("#gpu<all_reduce_op ", stringify!($kind), ">");
        })+
    };
}

impl_redux_kind! {ReduxAdd, add i32, add u32}
impl_subgroup_kind!(ReduxAdd, add i32, add u32);
impl_redux_kind! {ReduxMax, max i32, umax u32}
impl_subgroup_kind!(ReduxMax, maxsi i32, maxui u32);
impl_redux_kind! {ReduxMin, min i32, umin u32}
impl_subgroup_kind!(ReduxMin, minsi i32, minui u32);

impl_redux_kind! {ReduxAnd, and u32}
impl_redux_kind! {ReduxOr, or u32}
impl_redux_kind! {ReduxXor, xor u32}

#[expect(private_bounds)]
pub trait WarpReduceOp<T, Op: ReduxKind> {
    fn redux(&self, op: Op, value: T) -> T;
}

///
///
/// ```rust
/// let warp = gpu::cg::ThreadWarpTile::<16>;
/// let size = warp.size();
/// ```
///
///  ```rust,compile_fail,E0080
/// let warp = gpu::cg::ThreadWarpTile::<3>;
/// let size = warp.size();
/// ```
impl<const SIZE: usize, const STRIDE: usize> ThreadWarpTile<SIZE, STRIDE> {
    const CHECKED_SIZE: u32 = {
        assert!(
            SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8 || SIZE == 16 || SIZE == 32,
            "SIZE must be <= 32 and be a power of 2"
        );
        assert!(STRIDE <= 32 / SIZE && STRIDE >= 1, "STRIDE must be >= 1 and <= 32 / SIZE");
        SIZE as u32
    };
}

/// Implement simple flexible warp with STRIDE = 1.
impl<const SIZE: usize> ThreadWarpTile<SIZE, 1> {
    pub const BASE_THREAD_MASK: u32 = { ((1u64 << Self::CHECKED_SIZE) - 1) as u32 };
    pub const LANE_MASK: u32 = Self::CHECKED_SIZE - 1;

    // warp size == 32 => 0
    // warp size == 1 => 5
    pub const SHIFT_COUNT: u32 = { 5 - Self::CHECKED_SIZE.trailing_zeros() };

    #[gpu_codegen::device]
    #[inline(always)]
    pub const fn size(&self) -> usize {
        Self::CHECKED_SIZE as _
    }

    pub fn meta_group_size(&self) -> usize {
        Self::_meta_group_size()
    }

    pub(crate) fn _meta_group_size() -> usize {
        crate::dim::block_size() / Self::CHECKED_SIZE as usize
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub fn subgroup_id(&self) -> usize {
        Self::_subgroup_id()
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub(crate) fn _subgroup_id() -> usize {
        (warp_id() << Self::SHIFT_COUNT as usize)
            + ((lane_id() & !(Self::LANE_MASK as usize)) >> Self::SHIFT_COUNT as usize)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub fn thread_rank(&self) -> u32 {
        Self::_thread_rank()
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub(crate) fn _thread_rank() -> u32 {
        lane_id() as u32 & Self::LANE_MASK
    }

    /// E.g., when SIZE = 8,
    /// lane_id -> mask
    /// 0 -> 0xff
    /// 1 -> 0xff
    /// 8 -> 0xff00
    /// 9 -> 0xff00
    pub fn thread_mask(&self) -> u32 {
        Self::BASE_THREAD_MASK << (lane_id() as u32 & !Self::LANE_MASK)
    }

    #[gpu_codegen::device]
    #[inline(always)]
    fn reduce_with_shuffle(&self, value: f32, op: impl Fn(f32, f32) -> f32) -> f32 {
        let mut offset = Self::CHECKED_SIZE >> 1;
        let mut value = value;
        while offset > 0 {
            let (peer_val, _) = crate::shuffle!(xor, value, offset, Self::CHECKED_SIZE);
            value = op(value, peer_val);
            offset /= 2;
        }
        value
    }

    // Hardware-specific warp reduce.
    #[gpu_codegen::device]
    #[inline(always)]
    #[expect(private_bounds)]
    pub fn nvcc_redux_sync<Op: NvvmReduxSyncKind<T>, T>(&self, _op: Op, value: T) -> T {
        _redux_sync::<T>(value, self.thread_mask(), <Op as NvvmReduxSyncKind<T>>::KIND)
    }
    /*#[gpu_codegen::device]
    #[inline(always)]
    pub fn run_on_lane_0<T>(self, slice: &mut [T], f: impl FnOnce(&mut T) + Clone + Send) {
        if self.lane_id() == 0 {
            // Build ref from the slice on the wrap
            let threads_per_block = block_id::<X>()
                * block_id::<Y>()
                * block_id::<Z>();
            // TODO: Although not exactly necessary, shall we enforce threads_per_block % SIZE == 0?
            let offset = (threads_per_block / SIZE)
                * (crate::block_id::<X>()
                    + crate::grid_dim::<X>()
                        * (crate::block_id::<Y>()
                            + crate::grid_dim::<Y>()
                                * crate::block_id::<Z>()))
                + self.subgroup_id();

            // SAFETY: The offset is unique per Warp not per GPU thread. Although
            // multiple threads in the same warp may access the same memory
            // location, the `run_on_lane_0` function ensures only lane 0 (a
            // specific thread inside a warp) will execute the closure.
            // Thus it is safe to use it here.
            let local_val = unsafe { crate::subslice_mut(slice, offset, 1) };

            // Call exec_on_thread_0
            f(&mut local_val[0]);
        }
    }*/
}

/*impl<T, Op: ReduxKind> WarpReduceOp<T, Op> for ThreadWarpTile<32, 1>
where
    Op: SubGroupReduceKind<T>,
    Self: FullWarp
{
    fn redux(&self, op: Op, value: T) -> T {
        self.subgroup_reduce(op, value)
    }
}
*/

/// Ideally, we should use `SubGroupReduceKind` here, but MLIR support is limited.
/// Let user use subgroup_reduce directly.
impl<const SIZE: usize, T, Op: ReduxKind> WarpReduceOp<T, Op> for ThreadWarpTile<SIZE, 1>
where
    Op: NvvmReduxSyncKind<T>,
{
    fn redux(&self, op: Op, value: T) -> T {
        self.nvcc_redux_sync(op, value)
    }
}

impl<const SIZE: usize> WarpReduceOp<f32, ReduxMax> for ThreadWarpTile<SIZE> {
    // LLVM 20.1.8 does not support redux.sync.max.f32
    fn redux(&self, _op: ReduxMax, value: f32) -> f32 {
        let mut ret: f32;
        unsafe {
            core::arch::asm!(
                "redux.sync.max.f32 {0:e}, {1:e}, {2:e}",
                out(reg) ret,
                in(reg) value,
                in(reg) self.thread_mask()
            );
        }
        ret
    }
}

/// PTX does not support `redux.sync.add.f32` and so implement it via shuffle.
impl !SubGroupReduceKind<f32> for ReduxAdd {}
impl !NvvmReduxSyncKind<f32> for ReduxAdd {}

impl<const SIZE: usize> WarpReduceOp<f32, ReduxAdd> for ThreadWarpTile<SIZE> {
    fn redux(&self, _op: ReduxAdd, value: f32) -> f32 {
        self.reduce_with_shuffle(value, |a, b| a + b)
    }
}

impl<const SIZE: usize, const STRIDE: usize> ThreadWarpTile<SIZE, STRIDE> {
    /// Reduce by hardware-defined warp.
    /// For now, it only supports `i32` or `u32` types.
    #[rustc_diagnostic_item = "gpu::subgroup_reduce"]
    #[gpu_codegen::device]
    #[inline(never)]
    pub fn _subgroup_reduce<T>(_value: T, _op: &'static str) -> T {
        unimplemented!()
    }

    /// Reduce by software-defined warp.
    #[gpu_codegen::device]
    #[inline(always)]
    #[expect(private_bounds)]
    pub fn subgroup_reduce<Op, T>(self, _op: Op, value: T) -> T
    where
        Op: SubGroupReduceKind<T>,
    {
        Self::_subgroup_reduce::<T>(value, Op::SUB_GROUP_KIND)
    }
}

#[rustc_diagnostic_item = "nvvm::redux_sync"]
#[gpu_codegen::device]
#[inline(never)]
pub fn _redux_sync<T>(_value: T, _mask: u32, _op: &'static str) -> T {
    unimplemented!()
}

#[rustc_diagnostic_item = "gpu::shuffle"]
#[gpu_codegen::device]
#[inline(never)]
pub fn _shuffle<T>(_value: T, _offset: u32, _width: u32, _op: &'static str) -> (T, bool) {
    unimplemented!()
}

/// define a macro to use shuffle with a specific mode xor, up, down.
#[macro_export]
macro_rules! shuffle {
    (xor, $value:expr, $offset:expr, $width:expr) => {{ $crate::cg::_shuffle($value, $offset, $width, "#gpu<shuffle_mode xor>") }};
    (up, $value:expr, $offset:expr, $width:expr) => {{ $crate::cg::_shuffle($value, $offset, $width, "#gpu<shuffle_mode up>") }};
    (down, $value:expr, $offset:expr, $width:expr) => {{ $crate::cg::_shuffle($value, $offset, $width, "#gpu<shuffle_mode down>") }};
    (idx, $value:expr, $offset:expr, $width:expr) => {{ $crate::cg::_shuffle($value, $offset, $width, "#gpu<shuffle_mode idx>") }};
}
