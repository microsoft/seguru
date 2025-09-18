pub struct DimX;
pub struct DimY;
pub struct DimZ;

pub trait DimType {
    const DIM_ID: usize;
    const MLIR_DIM: &str;
}

pub(crate) enum DimTypeID {
    X = 0,
    Y = 1,
    Z = 2,
    Max = 3,
}

impl DimType for DimX {
    const DIM_ID: usize = DimTypeID::X as usize;
    const MLIR_DIM: &'static str = "#gpu<dim x>";
}

impl DimType for DimY {
    const DIM_ID: usize = DimTypeID::Y as usize;
    const MLIR_DIM: &'static str = "#gpu<dim y>";
}

impl DimType for DimZ {
    const DIM_ID: usize = DimTypeID::Z as usize;
    const MLIR_DIM: &'static str = "#gpu<dim z>";
}

macro_rules! def_dim_fn {
    ($name: ident, $priv_name: ident, $pub_name: ident, $(#[$meta:meta])*) => {
        $(#[$meta])*
        #[rustc_diagnostic_item = concat!("gpu::", stringify!($name))]
        #[gpu_codegen::device]
        #[inline(never)]
        const fn $priv_name() -> usize {
            unimplemented!()
        }

        $(#[$meta])*
        #[gpu_codegen::device]
        #[inline(always)]
        pub const fn $pub_name<D: DimType>() -> usize {
            crate::add_mlir_string_attr(D::MLIR_DIM);
            $priv_name()
        }
    };
}

def_dim_fn!(global_thread_id, _global_thread_id, global_id,);
def_dim_fn!(thread_id, _thread_id, thread_id,);
def_dim_fn!(block_id, _block_id, block_id,);
def_dim_fn!(block_dim, _block_dim, block_dim, #[gpu_codegen::ret_sync_data(1000)]);
def_dim_fn!(grid_dim, _grid_dim, grid_dim, #[gpu_codegen::ret_sync_data(1000)]);

#[gpu_codegen::device]
#[inline(always)]
#[allow(dead_code)]
pub fn warp_id() -> usize {
    let mut ret: u32;
    unsafe {
        core::arch::asm!("mov.u32 {0:e}, %warpid;", out(reg) ret);
    }
    ret as usize
}

#[gpu_codegen::device]
#[inline(always)]
#[allow(dead_code)]
pub fn lane_id() -> usize {
    let mut laneid: u32;
    unsafe {
        core::arch::asm!("mov.u32 {0:e}, %laneid;", out(reg) laneid);
    }
    laneid as usize
}

#[gpu_codegen::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
pub const fn dim<D: DimType>() -> usize {
    block_dim::<D>() * grid_dim::<D>()
}

#[gpu_codegen::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
pub fn block_size() -> usize {
    block_dim::<DimX>() * block_dim::<DimY>() * block_dim::<DimZ>()
}

#[gpu_codegen::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
#[allow(dead_code)]
pub fn num_blocks() -> usize {
    grid_dim::<DimX>() * grid_dim::<DimY>() * grid_dim::<DimZ>()
}

/// Users should not use it directly. It should only used from gpu_macros.
/// # Safety
/// This function is safe when Config is properly instantiated from host side.
#[doc(hidden)]
#[inline(always)]
#[gpu_codegen::device]
#[cfg(not(feature = "codegen_tests"))]
pub unsafe fn assume_dim_with_config<Config: crate::SafeGpuConfig>() {
    use core::intrinsics::assume;
    unsafe {
        assume(Config::GDIM_X == 0 || Config::GDIM_X as usize == grid_dim::<DimX>());
        assume(Config::GDIM_Y == 0 || Config::GDIM_Y as usize == grid_dim::<DimY>());
        assume(Config::GDIM_Z == 0 || Config::GDIM_Z as usize == grid_dim::<DimZ>());
        assume(Config::BDIM_X == 0 || Config::BDIM_X as usize == block_dim::<DimX>());
        assume(Config::BDIM_Y == 0 || Config::BDIM_Y as usize == block_dim::<DimY>());
        assume(Config::BDIM_Z == 0 || Config::BDIM_Z as usize == block_dim::<DimZ>());
    }
}
