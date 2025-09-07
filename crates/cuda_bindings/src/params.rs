use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::CudaMemBox;
use crate::ctx::GpuCtxSpace;
use crate::mem::GpuDataMarker;

/// When dim == 0, it indicates that dim is dynamic and should refer to config
/// When BLOCK_DIM_X != 0
///  we should add assume!(block_dim(X) == BLOCK_DIM_X) for optimization
/// When SHARED_SIZE == u32::MAX, we use shared size from config.
/// Such rule is enforced by GPUConfigMethods trait.
pub trait GPUConfig {
    const BLOCK_DIM_X: u32 = 1;
    const BLOCK_DIM_Y: u32 = 1;
    const BLOCK_DIM_Z: u32 = 1;
    const GRID_DIM_X: u32 = 1;
    const GRID_DIM_Y: u32 = 1;
    const GRID_DIM_Z: u32 = 1;
    const SHARED_SIZE: Option<u32> = Some(0);

    fn dynamic_grid_dim_x(&self) -> u32 {
        Self::GRID_DIM_X
    }

    fn dynamic_grid_dim_y(&self) -> u32 {
        Self::GRID_DIM_Y
    }

    fn dynamic_grid_dim_z(&self) -> u32 {
        Self::GRID_DIM_Z
    }

    fn dynamic_block_dim_x(&self) -> u32 {
        Self::BLOCK_DIM_X
    }

    fn dynamic_block_dim_y(&self) -> u32 {
        Self::BLOCK_DIM_Y
    }

    fn dynamic_block_dim_z(&self) -> u32 {
        Self::BLOCK_DIM_Z
    }

    fn dynamic_shared_size(&self) -> u32 {
        Self::SHARED_SIZE.unwrap_or(0)
    }
}

pub trait GPUConfigMethods: GPUConfig {
    #[inline]
    fn is_static(&self) -> bool {
        Self::BLOCK_DIM_X != 0
            && Self::BLOCK_DIM_Y != 0
            && Self::BLOCK_DIM_Z != 0
            && Self::GRID_DIM_X != 0
            && Self::GRID_DIM_Y != 0
            && Self::GRID_DIM_Z != 0
            && Self::SHARED_SIZE.is_some()
    }

    fn grid_dim_x(&self) -> u32 {
        if Self::GRID_DIM_X != 0 { Self::GRID_DIM_X } else { self.dynamic_grid_dim_x() }
    }

    fn grid_dim_y(&self) -> u32 {
        if Self::GRID_DIM_Y != 0 { Self::GRID_DIM_Y } else { self.dynamic_grid_dim_y() }
    }

    fn grid_dim_z(&self) -> u32 {
        if Self::GRID_DIM_Z != 0 { Self::GRID_DIM_Z } else { self.dynamic_grid_dim_z() }
    }

    fn block_dim_x(&self) -> u32 {
        if Self::BLOCK_DIM_X != 0 { Self::BLOCK_DIM_X } else { self.dynamic_block_dim_x() }
    }

    fn block_dim_y(&self) -> u32 {
        if Self::BLOCK_DIM_Y != 0 { Self::BLOCK_DIM_Y } else { self.dynamic_block_dim_y() }
    }

    fn block_dim_z(&self) -> u32 {
        if Self::BLOCK_DIM_Z != 0 { Self::BLOCK_DIM_Z } else { self.dynamic_block_dim_z() }
    }

    fn shared_size(&self) -> u32 {
        if let Some(s) = Self::SHARED_SIZE { s } else { self.dynamic_shared_size() }
    }
}

/// Do not allow override GPUConfigMethods.
impl<T: GPUConfig> GPUConfigMethods for T {}

/// Full static config
pub struct GPUStaticConfig<
    const BX: u32,
    const BY: u32,
    const BZ: u32,
    const GX: u32,
    const GY: u32,
    const GZ: u32,
    const SS: u32,
>;

impl<
    const BX: u32,
    const BY: u32,
    const BZ: u32,
    const GX: u32,
    const GY: u32,
    const GZ: u32,
    const SS: u32,
> GPUConfig for GPUStaticConfig<BX, BY, BZ, GX, GY, GZ, SS>
{
    const BLOCK_DIM_X: u32 = BX;
    const BLOCK_DIM_Y: u32 = BY;
    const BLOCK_DIM_Z: u32 = BZ;
    const GRID_DIM_X: u32 = GX;
    const GRID_DIM_Y: u32 = GY;
    const GRID_DIM_Z: u32 = GZ;
    const SHARED_SIZE: Option<u32> = Some(SS);
}

pub struct GPUDynamicConfig {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_size: u32,
}

// Full dynamic config
impl GPUConfig for GPUDynamicConfig {
    const BLOCK_DIM_X: u32 = 0;
    const BLOCK_DIM_Y: u32 = 0;
    const BLOCK_DIM_Z: u32 = 0;
    const GRID_DIM_X: u32 = 0;
    const GRID_DIM_Y: u32 = 0;
    const GRID_DIM_Z: u32 = 0;
    const SHARED_SIZE: Option<u32> = None;

    fn dynamic_grid_dim_x(&self) -> u32 {
        self.grid_dim_x
    }

    fn dynamic_grid_dim_y(&self) -> u32 {
        self.grid_dim_y
    }

    fn dynamic_grid_dim_z(&self) -> u32 {
        self.grid_dim_z
    }

    fn dynamic_block_dim_x(&self) -> u32 {
        self.block_dim_x
    }

    fn dynamic_block_dim_y(&self) -> u32 {
        self.block_dim_y
    }

    fn dynamic_block_dim_z(&self) -> u32 {
        self.block_dim_z
    }

    fn dynamic_shared_size(&self) -> u32 {
        self.shared_size
    }
}

/// This trait is used to ensure that the types can be used as kernel parameters
/// passed from host to device. User should not implement this trait directly.
/// # Safety:
/// It must ensure that the data passed to the GPU is valid. in addition, the
/// returned pointer vec should be properly constructed to match the layout of
/// the target type. For example, primitive types could be passed by value;
/// paired types(e.g., slices) should be passed as two fields: pointer + length;
#[doc(hidden)]
#[allow(private_bounds)]
pub unsafe trait AsHostKernelParams {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>>;
}

/// We only allow the CudaMemBox to store a type T that is allowed as GpuDataMarker.
unsafe impl<T: Sized + GpuDataMarker, N: GpuCtxSpace> AsHostKernelParams for CudaMemBox<T, N> {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        vec![Box::new(self.as_ptr() as usize)]
    }
}

unsafe impl<T: ?Sized + GpuDataMarker, N: GpuCtxSpace> AsHostKernelParams for &CudaMemBox<T, N>
where
    CudaMemBox<T, N>: AsHostKernelParams,
{
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        CudaMemBox::<T, N>::as_kernel_param_data(*self)
    }
}

unsafe impl<T: ?Sized + Send, N: GpuCtxSpace> AsHostKernelParams for &mut CudaMemBox<T, N>
where
    CudaMemBox<T, N>: AsHostKernelParams,
{
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        CudaMemBox::<T, N>::as_kernel_param_data(*self)
    }
}

unsafe impl<T: Sized + Send, N: GpuCtxSpace> AsHostKernelParams for CudaMemBox<[T], N> {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        vec![Box::new(self.as_ptr() as *const T as usize), Box::new(self.as_ptr().len())]
    }
}

macro_rules! impl_as_kernel_params {
    ($u:ty) => {
        unsafe impl AsHostKernelParams for $u {
            fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
                vec![Box::new(*self)]
            }
        }
    };
    () => {};
}

macro_rules! impl_as_kernel_params_for {
    ($($t:ty),+) => {
        $(
            impl_as_kernel_params!($t);
        )+
    };
}

impl_as_kernel_params_for!(
    bool, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

#[macro_export(local_inner_macros)]
macro_rules! config_dim_field {
    ($name: ident, $dynamic_name: ident, , $var:literal, $idx: literal) => {
        const $name: u32 = $var;
    };
    ($name: ident, $dynamic_name: ident, const, $var:expr, $idx: literal) => {
        const $name: u32 = $var;
    };
    ($name: ident, $dynamic_name: ident, , $var:expr, $idx: literal) => {
        const $name: u32 = 0;
        fn $dynamic_name(&self) -> u32 {
            self.params[$idx]
        }
    };
}

#[macro_export(local_inner_macros)]
macro_rules! config_shared_size_field {
    ($name: ident, $dynamic_name: ident,,$var:literal, $idx: literal) => {
        const $name: Option<u32> = Some($var);
    };
    ($name: ident, $dynamic_name: ident, const, $var:expr, $idx: literal) => {
        const $name: Option<u32> = Some($var);
    };
    ($name: ident, $dynamic_name: ident,,$var:expr, $idx: literal) => {
        const $name: Option<u32> = None;
        fn $dynamic_name(&self) -> u32 {
            self.params[$idx]
        }
    };
}

/// Create a GPU execution config.
/// Automatically setting const dim and shared size if it is const
/// Constant sizes will give optimization hints to both host and device
/// functions.
#[macro_export]
macro_rules! gpu_config {
    ($(@$cgx: tt)? $gx:expr, $(@$cgy: tt)? $gy:expr, $(@$cgz: tt)? $gz:expr, $(@$cbx: tt)? $bx:expr, $(@$cby: tt)? $by:expr, $(@$cbz: tt)? $bz:expr, $(@$css: tt)? $ss:expr) => {{
        struct __GPU_TMP_CONFIG_ {
            params: [u32; 7],
        }

        impl $crate::GPUConfig for __GPU_TMP_CONFIG_ {
            $crate::config_dim_field!(GRID_DIM_X, dynamic_grid_dim_x, $($cgx)?, $gx, 0);
            $crate::config_dim_field!(GRID_DIM_Y, dynamic_grid_dim_y, $($cgy)?, $gy, 1);
            $crate::config_dim_field!(GRID_DIM_Z, dynamic_grid_dim_z, $($cgz)?, $gz, 2);
            $crate::config_dim_field!(BLOCK_DIM_X, dynamic_block_dim_x, $($cbx)?, $bx, 3);
            $crate::config_dim_field!(BLOCK_DIM_Y, dynamic_block_dim_y, $($cby)?, $by, 4);
            $crate::config_dim_field!(BLOCK_DIM_Z, dynamic_block_dim_z, $($cbz)?, $bz, 5);
            $crate::config_shared_size_field!(SHARED_SIZE, dynamic_shared_size, $($css)?, $ss, 6);
        }
        __GPU_TMP_CONFIG_ { params: [$gx, $gy, $gz, $bx, $by, $bz, $ss] }
    }};
}

#[test]
fn test_config() {
    // Simple const dim sizes using literal
    let config = gpu_config!(1, 2, 3, 10, 11, 12, 10);
    assert!(config.is_static());
}

#[test]
fn test_mixed_config() {
    // Const dim sizes using const var
    const GDIM_X: u32 = 1;
    const BDIM_Y: u32 = 11;
    let config = gpu_config!(@const GDIM_X, 2, 3, 10, @const BDIM_Y, 12, 10);
    assert!(config.is_static());

    // Dynamic dim sizes
    let dim = 3;
    let config = gpu_config!(dim, dim, dim, dim, dim, dim, dim);
    assert!(!config.is_static());

    // Mix literal, const var, dynamic var.
    let config = gpu_config!(@const GDIM_X, 2, dim, 10, @const BDIM_Y, 12, 10);
    assert!(!config.is_static());
    assert!(config.grid_dim_x() == 1);
    assert!(config.grid_dim_y() == 2);
    assert!(config.grid_dim_z() == 3);
    assert!(config.block_dim_x() == 10);
    assert!(config.block_dim_y() == 11);
    assert!(config.block_dim_z() == 12);
    assert!(config.shared_size() == 10);
}
