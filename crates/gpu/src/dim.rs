pub struct DimX;
pub struct DimY;
pub struct DimZ;

pub trait DimType {
    const DIM_ID: usize;
    const MLIR_DIM: &str;
}

enum DimTypeID {
    X = 0,
    Y = 1,
    Z = 2,
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
def_dim_fn!(block_dim, _block_dim, block_dim, #[gpu_codegen::ret_sync_data(0)]);
def_dim_fn!(grid_dim, _grid_dim, grid_dim, #[gpu_codegen::ret_sync_data(0)]);

#[gpu_codegen::device]
#[gpu_codegen::ret_sync_data(0)]
#[inline(always)]
pub const fn dim<D: DimType>() -> usize {
    block_dim::<D>() * grid_dim::<D>()
}

#[gpu_codegen::device]
#[inline(always)]
pub fn block_thread_ids() -> [usize; 6] {
    [
        thread_id::<DimX>(),
        thread_id::<DimY>(),
        thread_id::<DimZ>(),
        block_id::<DimX>(),
        block_id::<DimY>(),
        block_id::<DimZ>(),
    ]
}

#[derive(Clone, Copy)]
pub struct GpuChunkIdx {
    id: usize,
}

impl Default for GpuChunkIdx {
    #[gpu_codegen::device]
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl GpuChunkIdx {
    #[gpu_codegen::device]
    pub const fn threads_count() -> usize {
        block_dim::<DimX>()
            * grid_dim::<DimX>()
            * block_dim::<DimY>()
            * grid_dim::<DimY>()
            * block_dim::<DimZ>()
            * grid_dim::<DimZ>()
    }

    // TODO: optimize the default 3D to 2D or 1D based on the user-specified grid and block dimensions.
    #[gpu_codegen::device]
    #[inline(always)]
    pub const fn new() -> GpuChunkIdx {
        let block_x = block_dim::<DimX>();
        let grid_x = grid_dim::<DimX>();
        let id_x = block_id::<DimX>() * block_x + thread_id::<DimX>();
        let dim_x = block_x * grid_x;

        let block_y = block_dim::<DimY>();
        let grid_y = grid_dim::<DimY>();
        let id_y = block_id::<DimY>() * block_y + thread_id::<DimY>();
        let dim_y = block_y * grid_y;

        let block_z = block_dim::<DimZ>();
        let id_z = block_id::<DimZ>() * block_z + thread_id::<DimZ>();
        let id_t = id_x + dim_x * (id_y + id_z * dim_y);
        GpuChunkIdx { id: id_t }
    }

    #[gpu_codegen::device]
    #[inline(always)]
    #[allow(dead_code)]
    pub const fn as_usize(&self) -> usize {
        self.id
    }
}

#[derive(Clone, Copy)]
pub struct GpuSharedChunkIdx {
    id: usize,
}

impl Default for GpuSharedChunkIdx {
    #[gpu_codegen::device]
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl GpuSharedChunkIdx {
    #[gpu_codegen::device]
    #[inline(always)]
    pub const fn new() -> GpuSharedChunkIdx {
        let id_x = thread_id::<DimX>();
        let block_x = block_dim::<DimX>();
        let id_y = thread_id::<DimY>();
        let block_y = block_dim::<DimY>();
        let id_z = thread_id::<DimZ>();
        let id_t = id_x + block_x * (id_y + id_z * block_y);
        GpuSharedChunkIdx { id: id_t }
    }

    #[gpu_codegen::device]
    #[inline(always)]
    #[allow(dead_code)]
    pub const fn as_usize(&self) -> usize {
        self.id
    }
}

/// Users should not use it directly. It should only used from gpu_macros.
/// # Safety
/// This function is safe when Config is properly instantiated from host side.
#[doc(hidden)]
#[inline(always)]
#[gpu_codegen::device]
#[cfg(not(feature = "codegen_tests"))]
pub unsafe fn assume_dim_with_config<Config: crate::GPUConfig>() {
    use core::intrinsics::assume;
    unsafe {
        assume(Config::GRID_DIM_X == 0 || Config::GRID_DIM_X as usize == grid_dim::<DimX>());
        assume(Config::GRID_DIM_Y == 0 || Config::GRID_DIM_Y as usize == grid_dim::<DimY>());
        assume(Config::GRID_DIM_Z == 0 || Config::GRID_DIM_Z as usize == grid_dim::<DimZ>());
        assume(Config::BLOCK_DIM_X == 0 || Config::BLOCK_DIM_X as usize == block_dim::<DimX>());
        assume(Config::BLOCK_DIM_Y == 0 || Config::BLOCK_DIM_Y as usize == block_dim::<DimY>());
        assume(Config::BLOCK_DIM_Z == 0 || Config::BLOCK_DIM_Z as usize == block_dim::<DimZ>());
    }
}
