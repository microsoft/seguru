pub struct DimX;
pub struct DimY;
pub struct DimZ;

pub trait DimType {
    const DIM_ID: u8;
    const MLIR_DIM: &str;
}

pub(crate) enum DimTypeID {
    X = 0,
    Y = 1,
    Z = 2,
    Max = 3,
}

impl DimType for DimX {
    const DIM_ID: u8 = DimTypeID::X as u8;
    const MLIR_DIM: &'static str = "#gpu<dim x>";
}

impl DimType for DimY {
    const DIM_ID: u8 = DimTypeID::Y as u8;
    const MLIR_DIM: &'static str = "#gpu<dim y>";
}

impl DimType for DimZ {
    const DIM_ID: u8 = DimTypeID::Z as u8;
    const MLIR_DIM: &'static str = "#gpu<dim z>";
}

macro_rules! def_dim_fn {
    ($name: ident, $priv_name: ident, $pub_name: ident, $(#[$meta:meta])*) => {
        $(#[$meta])*
        #[rustc_diagnostic_item = concat!("gpu::", stringify!($name))]
        #[gpu_macros::device]
        #[inline(never)]
        const fn $priv_name() -> usize {
            unimplemented!()
        }

        $(#[$meta])*
        #[gpu_macros::device]
        #[inline(always)]
        #[expect(clippy::cast_possible_truncation)]
        pub const fn $pub_name<D: DimType>() -> u32 {
            crate::add_mlir_string_attr(D::MLIR_DIM);
            $priv_name() as u32
        }
    };
}

def_dim_fn!(global_thread_id, _global_thread_id, global_id,);
def_dim_fn!(thread_id, _thread_id, thread_id,);
def_dim_fn!(block_id, _block_id, block_id,);
def_dim_fn!(block_dim, _block_dim, block_dim, #[gpu_codegen::ret_sync_data(1000)]);
def_dim_fn!(grid_dim, _grid_dim, grid_dim, #[gpu_codegen::ret_sync_data(1000)]);

/// This is the hardware warp id indicating
/// the warp within a SM.
/// Thus, it is different from the subgroup id /warp id
/// we usually use inside BlockTile.
#[gpu_macros::device]
#[inline(always)]
#[allow(dead_code)]
pub fn sm_warp_id() -> u32 {
    let mut ret: u32;
    unsafe {
        crate::asm!("mov.u32 {0:reg32}, %warpid;", out(reg) ret);
    }
    ret
}

#[gpu_macros::device]
#[inline(always)]
#[allow(dead_code)]
pub fn lane_id() -> u32 {
    let mut laneid: u32;
    unsafe {
        crate::asm!("mov.u32 {0:reg32}, %laneid;", out(reg) laneid);
    }
    laneid
}

#[gpu_macros::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
pub const fn dim<D: DimType>() -> u32 {
    block_dim::<D>() * grid_dim::<D>()
}

#[gpu_macros::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
pub fn block_size() -> u32 {
    block_dim::<DimX>() * block_dim::<DimY>() * block_dim::<DimZ>()
}

#[gpu_macros::device]
#[gpu_codegen::ret_sync_data(1000)]
#[inline(always)]
#[allow(dead_code)]
pub fn num_blocks() -> u32 {
    grid_dim::<DimX>() * grid_dim::<DimY>() * grid_dim::<DimZ>()
}
