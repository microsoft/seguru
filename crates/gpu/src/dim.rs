pub enum DimType {
    X,
    Y,
    Z,
}

macro_rules! dim_fn {
    ("x", $f: ident) => {{
        $crate::add_mlir_string_attr("#gpu<dim x>");
        $f()
    }};
    ("y", $f: ident) => {{
        $crate::add_mlir_string_attr("#gpu<dim y>");
        $f()
    }};
    ("z", $f: ident) => {{
        $crate::add_mlir_string_attr("#gpu<dim z>");
        $f()
    }};
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
        pub const fn $pub_name(dim: DimType) -> usize {
            match dim {
                DimType::X => dim_fn!("x", $priv_name),
                DimType::Y => dim_fn!("y", $priv_name),
                DimType::Z => dim_fn!("z", $priv_name),
            }
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
pub const fn dim(dim: DimType) -> usize {
    match dim {
        DimType::X => block_dim(DimType::X) * grid_dim(DimType::X),
        DimType::Y => block_dim(DimType::Y) * grid_dim(DimType::Y),
        DimType::Z => block_dim(DimType::Z) * grid_dim(DimType::Z),
    }
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
        block_dim(DimType::X)
            * grid_dim(DimType::X)
            * block_dim(DimType::Y)
            * grid_dim(DimType::Y)
            * block_dim(DimType::Z)
            * grid_dim(DimType::Z)
    }

    // TODO: optimize the default 3D to 2D or 1D based on the user-specified grid and block dimensions.
    #[gpu_codegen::device]
    #[inline(always)]
    pub const fn new() -> GpuChunkIdx {
        let block_x = block_dim(DimType::X);
        let grid_x = grid_dim(DimType::X);
        let id_x = block_id(DimType::X) * block_x + thread_id(DimType::X);
        let dim_x = block_x * grid_x;

        let block_y = block_dim(DimType::Y);
        let grid_y = grid_dim(DimType::Y);
        let id_y = block_id(DimType::Y) * block_y + thread_id(DimType::Y);
        let dim_y = block_y * grid_y;

        let block_z = block_dim(DimType::Z);
        let id_z = block_id(DimType::Z) * block_z + thread_id(DimType::Z);
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
        let id_x = thread_id(DimType::X);
        let block_x = block_dim(DimType::X);
        let id_y = thread_id(DimType::Y);
        let block_y = block_dim(DimType::Y);
        let id_z = thread_id(DimType::Z);
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
        assume(Config::GRID_DIM_X == 0 || Config::GRID_DIM_X as usize == grid_dim(DimType::X));
        assume(Config::GRID_DIM_Y == 0 || Config::GRID_DIM_Y as usize == grid_dim(DimType::Y));
        assume(Config::GRID_DIM_Z == 0 || Config::GRID_DIM_Z as usize == grid_dim(DimType::Z));
        assume(Config::BLOCK_DIM_X == 0 || Config::BLOCK_DIM_X as usize == block_dim(DimType::X));
        assume(Config::BLOCK_DIM_Y == 0 || Config::BLOCK_DIM_Y as usize == block_dim(DimType::Y));
        assume(Config::BLOCK_DIM_Z == 0 || Config::BLOCK_DIM_Z as usize == block_dim(DimType::Z));
    }
}
