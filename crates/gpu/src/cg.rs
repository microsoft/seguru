use crate::add_mlir_string_attr;

#[rustc_diagnostic_item = concat!("gpu::subgroup_id")]
#[gpu_codegen::device]
#[inline(never)]
#[allow(dead_code)]
fn subgroup_id() -> usize {
    unimplemented!()
}

#[rustc_diagnostic_item = concat!("gpu::lane_id")]
#[gpu_codegen::device]
#[inline(never)]
#[allow(dead_code)]
fn lane_id() -> usize {
    unimplemented!()
}

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
///
pub struct ThreadWarpTile<const SIZE: usize = 32, const STRIDE: usize = 1>();

#[repr(u8)]
pub enum ReduceOp {
    Add,
    Mul,
    MaxF,
    MinF,
}

impl ReduceOp {
    pub const ADD: u8 = ReduceOp::Add as u8;
    pub const MUL: u8 = ReduceOp::Mul as u8;
    pub const MAXF: u8 = ReduceOp::MaxF as u8;
    pub const MINF: u8 = ReduceOp::MinF as u8;
}

#[rustc_diagnostic_item = "gpu::all_reduce"]
#[gpu_codegen::device]
#[inline(never)]
pub fn _all_reduce<T>(_value: T) -> T {
    unimplemented!()
}

/// Reduce by software-defined warp.
#[gpu_codegen::device]
#[inline(always)]
pub fn all_reduce<T, const O: u8>(value: T) -> T {
    match O {
        ReduceOp::ADD => {
            add_mlir_string_attr("#gpu<all_reduce_op add>");
        }
        ReduceOp::MUL => {
            add_mlir_string_attr("#gpu<all_reduce_op mul>");
        }
        ReduceOp::MAXF => {
            add_mlir_string_attr("#gpu<all_reduce_op maxnumf>");
        }
        ReduceOp::MINF => {
            add_mlir_string_attr("#gpu<all_reduce_op minnumf>");
        }
        _ => {}
    }
    _all_reduce(value)
}

impl<const SIZE: usize, const STRIDE: usize> ThreadWarpTile<SIZE, STRIDE> {
    /// Reduce by hardware-defined warp.
    /// FOr now, it only supports `i32` or `u32` types.  
    #[rustc_diagnostic_item = "gpu::subgroup_reduce"]
    #[gpu_codegen::device]
    #[inline(never)]
    pub fn _subgroup_reduce<T>(self, _value: T) -> T {
        unimplemented!()
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub fn subgroup_id(&self) -> usize {
        (crate::thread_id(crate::DimType::X)
            + crate::block_dim(crate::DimType::X)
                * (crate::thread_id(crate::DimType::Y)
                    + crate::block_dim(crate::DimType::Y) * crate::thread_id(crate::DimType::Z)))
            / SIZE
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub fn lane_id(&self) -> usize {
        (crate::thread_id(crate::DimType::X)
            + crate::block_dim(crate::DimType::X)
                * (crate::thread_id(crate::DimType::Y)
                    + crate::block_dim(crate::DimType::Y) * crate::thread_id(crate::DimType::Z)))
            % SIZE
    }

    #[gpu_codegen::device]
    #[inline(always)]
    pub fn run_on_lane_0<T>(self, slice: &mut [T], f: impl FnOnce(&mut T) + Clone + Send) {
        if self.lane_id() == 0 {
            // Build ref from the slice on the wrap
            let threads_per_block = crate::block_dim(crate::DimType::X)
                * crate::block_dim(crate::DimType::Y)
                * crate::block_dim(crate::DimType::Z);
            // TODO: Although not exactly necessary, shall we enforce threads_per_block % SIZE == 0?
            let offset = (threads_per_block / SIZE)
                * (crate::block_id(crate::DimType::X)
                    + crate::grid_dim(crate::DimType::X)
                        * (crate::block_id(crate::DimType::Y)
                            + crate::grid_dim(crate::DimType::Y)
                                * crate::block_id(crate::DimType::Z)))
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
    }

    /// Reduce by software-defined warp.
    #[gpu_codegen::device]
    #[inline(always)]
    pub fn subgroup_reduce<const O: u8, T>(self, value: T) -> T {
        match O {
            ReduceOp::ADD => {
                add_mlir_string_attr("#gpu<all_reduce_op add>");
            }
            ReduceOp::MUL => {
                add_mlir_string_attr("#gpu<all_reduce_op mul>");
            }
            ReduceOp::MAXF => {
                add_mlir_string_attr("#gpu<all_reduce_op maxnumf>");
            }
            ReduceOp::MINF => {
                add_mlir_string_attr("#gpu<all_reduce_op minnumf>");
            }
            _ => {}
        }
        self._subgroup_reduce::<T>(value)
    }
}

#[rustc_diagnostic_item = "nvvm::redux_sync"]
#[gpu_codegen::device]
#[inline(never)]
pub fn _redux_sync<T>(_value: T, _mask: u32) -> T {
    unimplemented!()
}

#[rustc_diagnostic_item = "gpu::shuffle"]
#[gpu_codegen::device]
#[inline(never)]
pub fn _shuffle<T>(_value: T, _offset: u32, _width: u32) -> (T, bool) {
    unimplemented!()
}

/// define a macro to use shuffle with a specific mode xor, up, down.
#[macro_export]
macro_rules! shuffle {
    (xor, $value:expr, $offset:expr, $width:expr) => {{
        add_mlir_string_attr(concat!("#gpu<shuffle_mode xor>"));
        $crate::cg::_shuffle($value, $offset, $width)
    }};
    (up, $value:expr, $offset:expr, $width:expr) => {{
        add_mlir_string_attr(concat!("#gpu<shuffle_mode up>"));
        $crate::cg::_shuffle($value, $offset, $width)
    }};
    (down, $value:expr, $offset:expr, $width:expr) => {{
        add_mlir_string_attr(concat!("#gpu<shuffle_mode down>"));
        $crate::cg::_shuffle($value, $offset, $width)
    }};
    (idx, $value:expr, $offset:expr, $width:expr) => {{
        add_mlir_string_attr(concat!("#gpu<shuffle_mode idx>"));
        $crate::cg::_shuffle($value, $offset, $width)
    }};
}

#[gpu_codegen::device]
#[inline(always)]
pub fn reduce_add_f32(_warp: ThreadWarpTile<32, 1>, value: f32) -> f32 {
    pub const SIZE: u32 = 32;
    let mut offset = SIZE >> 1;
    let mut val = value;
    while offset > 0 {
        let (peer_val, _) = shuffle!(xor, value, offset, SIZE);
        val += peer_val;
        offset /= 2;
    }
    val
}

#[gpu_codegen::device]
#[inline(always)]
pub fn reduce_max_f32(_warp: ThreadWarpTile<32, 1>, value: f32) -> f32 {
    pub const SIZE: u32 = 32;
    let mut offset = SIZE >> 1;
    let mut val = value;
    while offset > 0 {
        let (peer_val, _) = shuffle!(xor, value, offset, SIZE);
        val = if val > peer_val { val } else { peer_val };
        offset /= 2;
    }
    val
}
