/// Allow the safe conversion from host types Self to device types T.
trait SafeHostToDev<T> {}

/// Expose the convert function to users.
/// This trait is sealed to prevent arbitrary implementations.
/// Only types that implement SafeHostToDev can implement this trait.
/// This ensures that only safe conversions are allowed.
#[allow(private_bounds)]
pub trait HostToDev<T>: Sized + SafeHostToDev<T> {
    fn convert(self) -> T {
        unimplemented!()
    }
}

impl<T> SafeHostToDev<T> for T {}

impl<T> HostToDev<T> for T {}

#[cfg(not(feature = "codegen_tests"))]
impl<'a, T: ?Sized> SafeHostToDev<&'a T> for &'a cuda_bindings::CudaMemBox<T> {}

#[cfg(not(feature = "codegen_tests"))]
impl<'a, T: ?Sized> HostToDev<&'a T> for &'a cuda_bindings::CudaMemBox<T> {}

#[cfg(not(feature = "codegen_tests"))]
impl<'a, T: ?Sized> SafeHostToDev<crate::global::GpuGlobal<'a, T>>
    for &'a mut cuda_bindings::CudaMemBox<T>
{
}

#[cfg(not(feature = "codegen_tests"))]
impl<'a, T: ?Sized> HostToDev<crate::global::GpuGlobal<'a, T>>
    for &'a mut cuda_bindings::CudaMemBox<T>
{
}
