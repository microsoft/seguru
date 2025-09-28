/// Disallow arbitrary implementations of HostToDev.
/// Allow the safe conversion from host types Self to device types T.
trait HostToDevPrivateSeal<T> {}

/// Expose the convert function to users.
/// This trait is sealed to prevent arbitrary implementations.
/// Only types that implement HostToDevPrivateSeal can implement this trait.
/// This ensures that only safe conversions are allowed,
/// ensuring safe host-to-device interface.
#[allow(private_bounds)]
pub trait HostToDev<T>: Sized + HostToDevPrivateSeal<T> {
    fn convert(self) -> T {
        unimplemented!()
    }
}

impl<T> HostToDevPrivateSeal<T> for T {}

impl<T> HostToDev<T> for T {}

#[cfg(not(feature = "codegen_tests"))]
impl<'a, 'b: 'a, T: ?Sized> HostToDevPrivateSeal<&'a T> for &'a cuda_bindings::TensorView<'b, T> {}

/// Allow host-side &`CudaMemBox<T>`  to device-side &T
#[cfg(not(feature = "codegen_tests"))]
impl<'a, 'b: 'a, T: ?Sized> HostToDev<&'a T> for &'a cuda_bindings::TensorView<'b, T> {}

/// Allow host-side &mut CudaMemBox<T>  to device-side GpuGlobal<T>
#[cfg(not(feature = "codegen_tests"))]
impl<'a, 'b: 'a, T: ?Sized> HostToDevPrivateSeal<crate::global::GpuGlobal<'a, T>>
    for &'a mut cuda_bindings::TensorViewMut<'b, T>
{
}

/// Allow host-side T to device-side T
#[cfg(not(feature = "codegen_tests"))]
impl<'a, 'b: 'a, T: ?Sized> HostToDev<crate::global::GpuGlobal<'a, T>>
    for &'a mut cuda_bindings::TensorViewMut<'b, T>
{
}
