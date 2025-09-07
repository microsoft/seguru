/// GPU execution configuration and safe interfaces for param passing.
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::CudaMemBox;
use crate::ctx::GpuCtxSpace;
use crate::mem::GpuDataMarker;
use crate::unsafe_bindings::CUdevprop;

#[rustc_diagnostic_item = "gpu_params::MAX_THREAD_PER_BLOCK"]
const MAX_THREAD_PER_BLOCK: i32 = 1024;

#[rustc_diagnostic_item = "gpu_params::MAX_BLOCK_DIM_X"]
const MAX_BLOCK_DIM_X: i32 = 1024;

#[rustc_diagnostic_item = "gpu_params::MAX_BLOCK_DIM_Y"]
const MAX_BLOCK_DIM_Y: i32 = 1024;

#[rustc_diagnostic_item = "gpu_params::MAX_BLOCK_DIM_Z"]
const MAX_BLOCK_DIM_Z: i32 = 64;

#[rustc_diagnostic_item = "gpu_params::MAX_GRID_DIM_X"]
const MAX_GRID_DIM_X: i32 = i32::MAX;

#[rustc_diagnostic_item = "gpu_params::MAX_GRID_DIM_Y"]
const MAX_GRID_DIM_Y: i32 = 0xffff;

#[rustc_diagnostic_item = "gpu_params::MAX_GRID_DIM_Z"]
const MAX_GRID_DIM_Z: i32 = 0xffff;

#[rustc_diagnostic_item = "gpu_params::MAX_SHARED_MEM_PER_BLOCK"]
const MAX_SHARED_MEM_PER_BLOCK: i32 = 163 * 1024;

/// Predefined `CUdevprop` for **compile-time checks**.
///
/// This constant encodes architectural limits derived from the
/// [Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability).
/// Note that actual hardware (physical) limits may be lower.
///
/// ### Example
/// For shared memory:
/// - A100 provides only **48 KB** in hardware,
/// - while the architectural limit for SM 8.0 is **163 KB**.
///
/// ### Encoded Properties
/// - `maxThreadsPerBlock` — maximum threads per block.
/// - `maxThreadsDim[3]` — maximum dimensions of a thread block.
/// - `maxGridSize[3]` — maximum dimensions of a grid.
/// - `sharedMemPerBlock` — total static + dynamic shared memory size.
/// - `SIMDWidth` — warp size (usually 32).
/// - Other fields are reserved or unused.
///
/// ### User-defined bounds via environment variables.
/// The compile-time bounds can be overridden by setting the corresponding environment variables.
/// - `MAX_THREAD_PER_BLOCK`
/// - `MAX_BLOCK_DIM_X`
/// - `MAX_BLOCK_DIM_Y`
/// - `MAX_BLOCK_DIM_Z`
/// - `MAX_GRID_DIM_X`
/// - `MAX_GRID_DIM_Y`
/// - `MAX_GRID_DIM_Z`
/// - `MAX_SHARED_MEM_PER_BLOCK`
///
///
/// ### Checking Validity
/// - Compile-time checks are enforced via the `SafeGpuConfig` trait.
/// - Runtime checks perform a HW-level check using the `runtime_check` method in `SafeGpuConfig`.
/// - Share memory size is only loosely checked at compile-time.
/// - TODO: Add bound in kernel function signature to enforce static + dynamic size < sharedMemPerBlock at compile-time.
pub const CU_DEV_PROP: CUdevprop = CUdevprop {
    maxThreadsPerBlock: MAX_THREAD_PER_BLOCK,
    maxThreadsDim: [MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z],
    maxGridSize: [MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z],
    sharedMemPerBlock: MAX_SHARED_MEM_PER_BLOCK,
    totalConstantMemory: 64 * 1024, // unused
    SIMDWidth: 32,                  // unused
    memPitch: i32::MAX,             // unused
    regsPerBlock: 0xffff,           // unused
    clockRate: 1410000,             // unused
    textureAlign: 512,              // unused
};

/// Allows users to define flexible GPU execution configurations.
///
/// ## Behavior and Rules
///
/// - When `dim == 0`, the dimension is considered dynamic and should be provided by the configuration instance.
/// - When `BLOCK_DIM_? != 0`, an `assume!(block_dim::<?>() == BLOCK_DIM_?)` is inserted as an optimization hint for the kernel.
/// - `SHARED_SIZE` determines dynamic shared memory:
///   - `None` → dynamic shared memory is determined at runtime.
///   - `Some(value)` → dynamic shared memory is constant and known at compile-time.
/// - The `SafeGpuConfig` trait enforces these rules.
///
/// ## Trust Model
///
/// Implementing `GPUConfig` is safe. Kernel execution does not rely on trusting the user-provided configuration.
/// Instead, all execution paths go through `SafeGpuConfig`, which performs compile-time and runtime validation.
///
/// ## Recommended to implement via `gpu_config!` macro
///
/// * Static configuration (all dimensions are static):
/// ```rust
/// const BX: u32 = 16;
/// cuda_bindings::gpu_config!(@const BX,1,1,1,1,1,0);
/// ```
///
/// * Dynamic configuration (BDIM_X is dynamic):
/// ```rust
/// let dyn_bdim_x: u32 = 12;
/// cuda_bindings::gpu_config!(dyn_bdim_x,1,1,1,1,1,0);
/// ```
///
/// Refer to `gpu_config!` macro documentation for details.
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

macro_rules! assert_compiletime_valid_config {
    ($expr:expr, $bound: expr) => {
        assert!(
            $expr as u128 <= $bound as u128,
            concat!("Expecting gpu-config: ", stringify!($expr), " <= ", stringify!($bound)),
        );
    };
}

macro_rules! assert_runtime_valid_config {
    ($expr:expr, $bound: expr) => {
        assert!(
            $expr as u128 <= $bound as u128,
            concat!(
                "Expecting gpu-config: ",
                stringify!($expr),
                " ({}) <= ",
                stringify!($bound),
                "({})."
            ),
            $expr,
            $bound
        );
    };
}

macro_rules! assert_valid_block_size {
    () => {
        assert_compiletime_valid_config!(Self::BLOCK_DIM_X, CU_DEV_PROP.maxThreadsDim[0]);
        assert_compiletime_valid_config!(Self::BLOCK_DIM_Y, CU_DEV_PROP.maxThreadsDim[1]);
        assert_compiletime_valid_config!(Self::BLOCK_DIM_Z, CU_DEV_PROP.maxThreadsDim[2]);
        assert_compiletime_valid_config!(
            Self::BLOCK_DIM_X * Self::BLOCK_DIM_Y,
            CU_DEV_PROP.maxThreadsPerBlock
        );
        assert_compiletime_valid_config!(
            Self::BLOCK_DIM_X * Self::BLOCK_DIM_Z,
            CU_DEV_PROP.maxThreadsPerBlock
        );
        assert_compiletime_valid_config!(
            Self::BLOCK_DIM_Y * Self::BLOCK_DIM_Z,
            CU_DEV_PROP.maxThreadsPerBlock
        );
        assert_compiletime_valid_config!(
            Self::BLOCK_DIM_X * Self::BLOCK_DIM_Y * Self::BLOCK_DIM_Z,
            CU_DEV_PROP.maxThreadsPerBlock
        );
    };
}

/// A safe wrapper for `GPUConfig` providing compile-time and runtime validation.
///
/// ## Compile-Time Checks
///
/// Ensures that all static values comply with the hardware limits defined in [`CU_DEV_PROP`].
/// If kernel execution uses a types implementing this GPUConfig manually, or created via
/// `gpu_config!`, will automatically perform these checks.
///
/// ### Example
///
/// * Define unused invalid static config with manual impl.
///
/// ```rust
/// struct InvalidConfig;
///
/// impl cuda_bindings::GPUConfig for InvalidConfig {
///    const BLOCK_DIM_Z: u32 = 1025;
/// }
/// let config = InvalidConfig; // No errors since it is not used.
/// ```
///
/// * Use invalid static config with manual impl.
/// ```rust,compile_fail,E0080
/// use cuda_bindings::SafeGpuConfig;
/// struct InvalidConfig;
///
/// impl cuda_bindings::GPUConfig for InvalidConfig {
///    const BLOCK_DIM_Z: u32 = 1025;
/// }
/// let config = InvalidConfig;
/// config.block_dim_z();
/// ```
///
/// See more in `gpu_config!` macro documentation.
///
/// ## Runtime Checks
///
/// Ensures that dynamic values are within the hardware limits at execution time.
/// This guarantees that kernels launched with dynamic configurations remain safe.
/// The `runtime_check` method verifies:
/// - `cu_dev_prop`: Actual device properties queried from the GPU.
/// - `max_dynamic_size`: Maximum dynamic shared memory size supported by the kernel.
pub trait SafeGpuConfig: GPUConfig {
    // Ensure that the block dim is valid.
    // This is a compile-time check.
    const CONFIG_CHECK: () = {
        assert_valid_block_size!();
        let _ = Self::SMEM_SIZE;
    };
    const BDIM_X: u32 = {
        assert_valid_block_size!();
        Self::BLOCK_DIM_X
    };
    const BDIM_Y: u32 = {
        assert_valid_block_size!();
        Self::BLOCK_DIM_Y
    };
    const BDIM_Z: u32 = {
        assert_valid_block_size!();
        Self::BLOCK_DIM_Z
    };

    const GDIM_X: u32 = {
        assert_compiletime_valid_config!(Self::GRID_DIM_X, CU_DEV_PROP.maxGridSize[0]);
        Self::GRID_DIM_X
    };
    const GDIM_Y: u32 = {
        assert_compiletime_valid_config!(Self::GRID_DIM_Y, CU_DEV_PROP.maxGridSize[1]);
        Self::GRID_DIM_Y
    };
    const GDIM_Z: u32 = {
        assert_compiletime_valid_config!(Self::GRID_DIM_Z, CU_DEV_PROP.maxGridSize[2]);
        Self::GRID_DIM_Z
    };
    const SMEM_SIZE: u32 = {
        if let Some(s) = Self::SHARED_SIZE {
            assert_compiletime_valid_config!(s, MAX_SHARED_MEM_PER_BLOCK);
            s
        } else {
            0
        }
    };

    /// Dynamic block size must be valid.
    /// If the static block size is invalid, it will cause a compilation error.
    /// If the block size is dynamic, we defer the check to runtime.
    fn runtime_check(&self, cu_dev_prop: CUdevprop, max_dynamic_size: i32) {
        assert_runtime_valid_config!(self.block_dim_x(), cu_dev_prop.maxThreadsDim[0]);
        assert_runtime_valid_config!(self.block_dim_y(), cu_dev_prop.maxThreadsDim[1]);
        assert_runtime_valid_config!(self.block_dim_z(), cu_dev_prop.maxThreadsDim[2]);
        assert_runtime_valid_config!(
            self.block_dim_x() * self.block_dim_y() * self.block_dim_z(),
            cu_dev_prop.maxThreadsPerBlock
        );
        assert_runtime_valid_config!(self.grid_dim_x(), cu_dev_prop.maxGridSize[0]);
        assert_runtime_valid_config!(self.grid_dim_y(), cu_dev_prop.maxGridSize[1]);
        assert_runtime_valid_config!(self.grid_dim_z(), cu_dev_prop.maxGridSize[2]);
        if let Some(s) = Self::SHARED_SIZE {
            assert_runtime_valid_config!(s, max_dynamic_size);
        } else {
            assert_runtime_valid_config!(self.shared_size(), max_dynamic_size);
        }
    }

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
        if Self::GDIM_X != 0 { Self::GRID_DIM_X } else { self.dynamic_grid_dim_x() }
    }

    fn grid_dim_y(&self) -> u32 {
        if Self::GDIM_Y != 0 { Self::GRID_DIM_Y } else { self.dynamic_grid_dim_y() }
    }

    fn grid_dim_z(&self) -> u32 {
        if Self::GDIM_Z != 0 { Self::GRID_DIM_Z } else { self.dynamic_grid_dim_z() }
    }

    fn block_dim_x(&self) -> u32 {
        if Self::BDIM_X != 0 { Self::BDIM_X } else { self.dynamic_block_dim_x() }
    }

    fn block_dim_y(&self) -> u32 {
        if Self::BDIM_Y != 0 { Self::BDIM_Y } else { self.dynamic_block_dim_y() }
    }

    fn block_dim_z(&self) -> u32 {
        if Self::BDIM_Z != 0 { Self::BDIM_Z } else { self.dynamic_block_dim_z() }
    }

    fn shared_size(&self) -> u32 {
        if let Some(s) = Self::SHARED_SIZE { s } else { self.dynamic_shared_size() }
    }
}

/// Do not allow override SafeGpuConfig.
impl<T: ?Sized + GPUConfig> SafeGpuConfig for T {}

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
///
/// Examples:
///
/// * Valid static config:
/// ```rust
/// let config = cuda_bindings::gpu_config!(1, 2, 3, 1, 1024, 1, 10);
/// ```
///
/// * Valid dynamic + static mixed config:
/// ```rust
/// fn create_config(bdim_x: u32) {
///     cuda_bindings::gpu_config!(1, 2, 3, bdim_x, 32, 1, 10);
/// }
/// create_config(32);
/// ```
///
/// * Invalid static config using gpu_config:
/// ```rust,compile_fail,E0080
/// let config = cuda_bindings::gpu_config!(1, 2, 3, 1025, 1, 1, 10);
/// ```
///
/// * Invalid dynamic mem size using gpu_config:
/// ```rust,compile_fail,E0080
/// let config = cuda_bindings::gpu_config!(1, 2, 3, 1, 1, 1, 0x100000);
/// ```
///
/// * Invalid dynamic + static config using gpu_config:
/// ```rust,compile_fail,E0080
/// fn create_config(bdim_y: u32) {
///    let config = cuda_bindings::gpu_config!(1, 2, 3, 32, bdim_y, 33, 10);
/// }
/// ```
///
/// * Invalid config detected at runtime in runtime_check:
///
/// ```rust,should_panic
/// fn create_config(bdim_x: u32) {
///    use cuda_bindings::SafeGpuConfig;
///     cuda_bindings::gpu_config!(1, 2, 3, bdim_x, 32, 1, 10)
///         .runtime_check(
///             cuda_bindings::params::CU_DEV_PROP,
///             cuda_bindings::params::CU_DEV_PROP.sharedMemPerBlock
///         );
/// }
/// create_config(33); // This will panic at runtime.
/// ```
///
#[macro_export]
macro_rules! gpu_config {
    ($(@$cgx: tt)? $gx:expr, $(@$cgy: tt)? $gy:expr, $(@$cgz: tt)? $gz:expr, $(@$cbx: tt)? $bx:expr, $(@$cby: tt)? $by:expr, $(@$cbz: tt)? $bz:expr, $(@$css: tt)? $ss:expr) => {{
        #[allow(non_camel_case_types, dead_code)]
        struct __GPU_TMP_CONFIG_ {
            params: [u32; 7],
        }

        // Non-critical compile-time checks.
        // This is to make the failure happens earlier before using the config in `launch_fn`.
        // It can be skipped if user choose to implement GPUConfig directly without using this macro.
        // But we will still do the check when using the config in `launch_fn`.
        const _: () = {
            use $crate::SafeGpuConfig;
            __GPU_TMP_CONFIG_::CONFIG_CHECK
        };

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
fn test_mixed_config() {
    // Const dim sizes using const var
    const GDIM_X: u32 = 1;
    const BDIM_Y: u32 = 8;
    const SMEM_SIZE: u32 = 10;
    let config = gpu_config!(@const GDIM_X, 2, 3, 9, @const BDIM_Y, 10, @const SMEM_SIZE);
    assert!(config.is_static());

    // Dynamic dim sizes
    let dim = 3;
    let config = gpu_config!(dim, dim, dim, dim, dim, dim, dim);
    assert!(!config.is_static());

    // Mix literal, const var, dynamic var.
    let config = gpu_config!(@const GDIM_X, 2, dim, 9, @const BDIM_Y, 12, 10);
    assert!(!config.is_static());
    assert!(config.grid_dim_x() == 1);
    assert!(config.grid_dim_y() == 2);
    assert!(config.grid_dim_z() == 3);
    assert!(config.block_dim_x() == 9);
    assert!(config.block_dim_y() == 8);
    assert!(config.block_dim_z() == 12);
    assert!(config.shared_size() == 10);
}

#[test]
#[should_panic(
    expected = "Expecting gpu-config: self.block_dim_x() * self.block_dim_y() * self.block_dim_z() (1056) <= cu_dev_prop.maxThreadsPerBlock(1024)"
)]
fn test_runtime_failed_assert() {
    let dim = 33;
    let config = gpu_config!(1, 2, 3, dim, 32, 1, 10);
    config.runtime_check(CU_DEV_PROP, MAX_SHARED_MEM_PER_BLOCK);
}
