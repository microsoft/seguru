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
        #[gpu_codegen::builtin(gpu.$name)]
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
