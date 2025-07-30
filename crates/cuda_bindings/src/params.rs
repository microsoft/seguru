use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::CudaMemBox;
use crate::ctx::GpuCtxSpace;

pub struct GPUConfig {
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
}

pub(crate) trait AsKernelParamsGuard {}

/// This trait is used to ensure that the types can be used as kernel parameters passed from host to device.
#[allow(private_bounds)]
pub trait AsHostKernelParams: AsKernelParamsGuard {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>>;
}

impl<T: Sized, N: GpuCtxSpace> AsKernelParamsGuard for CudaMemBox<T, N> {}

impl<T: Sized, N: GpuCtxSpace> AsHostKernelParams for CudaMemBox<T, N> {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        vec![Box::new(self.as_ptr() as usize)]
    }
}

impl<T: ?Sized, N: GpuCtxSpace> AsKernelParamsGuard for &CudaMemBox<T, N> {}

impl<T: ?Sized, N: GpuCtxSpace> AsHostKernelParams for &CudaMemBox<T, N>
where
    CudaMemBox<T, N>: AsHostKernelParams,
{
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        CudaMemBox::<T, N>::as_kernel_param_data(*self)
    }
}

impl<T: ?Sized, N: GpuCtxSpace> AsKernelParamsGuard for &mut CudaMemBox<T, N> {}

impl<T: ?Sized, N: GpuCtxSpace> AsHostKernelParams for &mut CudaMemBox<T, N>
where
    CudaMemBox<T, N>: AsHostKernelParams,
{
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        CudaMemBox::<T, N>::as_kernel_param_data(*self)
    }
}

impl<T, N: GpuCtxSpace> AsKernelParamsGuard for CudaMemBox<[T], N> {}
impl<T: Sized, N: GpuCtxSpace> AsHostKernelParams for CudaMemBox<[T], N> {
    fn as_kernel_param_data(&self) -> Vec<Box<dyn core::any::Any>> {
        vec![Box::new(self.as_ptr() as *const T as usize), Box::new(self.as_ptr().len())]
    }
}

macro_rules! impl_as_kernel_params {
    ($u:ty) => {
        impl AsKernelParamsGuard for $u {}
        impl AsHostKernelParams for $u {
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
