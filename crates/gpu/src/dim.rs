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
    ($name: ident, $priv_name: ident, $pub_name: ident) => {
        #[rustc_diagnostic_item = concat!("gpu::", stringify!($name))]
        #[gpu_codegen::device]
        #[inline(never)]
        fn $priv_name() -> usize {
            unimplemented!()
        }

        #[gpu_codegen::device]
        #[inline(always)]
        pub fn $pub_name(dim: DimType) -> usize {
            match dim {
                DimType::X => dim_fn!("x", $priv_name),
                DimType::Y => dim_fn!("y", $priv_name),
                DimType::Z => dim_fn!("z", $priv_name),
            }
        }
    };
}

def_dim_fn!(global_thread_id, _global_thread_id, global_id);
def_dim_fn!(thread_id, _thread_id, thread_id);
def_dim_fn!(block_dim, _block_dim, block_dim);
def_dim_fn!(grid_dim, _grid_dim, grid_dim);

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
    // TODO: optimize the default 3D to 2D or 1D based on the user-specified grid and block dimensions.
    #[gpu_codegen::device]
    #[inline(always)]
    pub fn new() -> GpuChunkIdx {
        let id_x = thread_id(DimType::X);
        let block_x = block_dim(DimType::X);
        let grid_x = grid_dim(DimType::X);
        let dim_x = block_x * grid_x;
        let id_y = thread_id(DimType::Y);
        let block_y = block_dim(DimType::Y);
        let grid_y = grid_dim(DimType::Y);
        let dim_y = block_y * grid_y;
        let id_z = thread_id(DimType::Z);
        let id_t = id_x + dim_x * (id_y + id_z * dim_y);
        GpuChunkIdx { id: id_t }
    }

    #[gpu_codegen::device]
    #[inline(always)]
    #[allow(dead_code)]
    pub fn as_usize(&self) -> usize {
        self.id
    }
}
