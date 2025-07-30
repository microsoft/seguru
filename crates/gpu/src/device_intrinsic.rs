/// This module provides missing GPU intrinsics provided by "libdevice.10.bc".
/// Some intrinsics are defined in Rust vstd but not all of them.
/// Refer to /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
macro_rules! impl_dev_intrinsics {
    ($name:ident, $t: ty) => {
        #[inline(never)]
        #[rustc_diagnostic_item = concat!("gpu::device_intrinsics::", stringify!($name))]
        #[gpu_codegen::device]
        fn $name(self) -> $t {
            unimplemented!()
        }
    };
}

macro_rules! impl_dev_intrinsics3 {
    ($name:ident, $t: ty) => {
        #[inline(never)]
        #[rustc_diagnostic_item = concat!("gpu::device_intrinsics::", stringify!($name))]
        #[gpu_codegen::device]
        fn $name(self, _y: $t, _z: $t) -> $t {
            unimplemented!()
        }
    };
}

/// codegen_test cannot find rust/deps/compiler_builtins-0.1.152.
macro_rules! impl_dev_intrinsics_for_core {
    () => {
        impl_dev_intrinsics!(max, Self);
        impl_dev_intrinsics!(min, Self);
        impl_dev_intrinsics!(sqrt, Self);
        impl_dev_intrinsics!(ceil, Self);
        impl_dev_intrinsics!(exp, Self);
        impl_dev_intrinsics!(exp2, Self);
        impl_dev_intrinsics!(sin, Self);
        impl_dev_intrinsics!(cos, Self);
        impl_dev_intrinsics!(log, Self);
    };
}

/// This trait provides the device intrinsics for floating-point types
/// that are not defined by Rust core::intrinsics.
pub trait GPUDeviceFloatIntrinsics: Sized {
    impl_dev_intrinsics3!(fma, Self);
    impl_dev_intrinsics!(rsqrt, Self);
    impl_dev_intrinsics!(expm1, Self);
    impl_dev_intrinsics!(sinh, Self);
    impl_dev_intrinsics!(cosh, Self);
    impl_dev_intrinsics!(tan, Self);
    impl_dev_intrinsics!(tanh, Self);
    impl_dev_intrinsics!(pow, Self);
    impl_dev_intrinsics!(log10, Self);
    impl_dev_intrinsics!(log1p, Self);
    impl_dev_intrinsics!(log2, Self);
    impl_dev_intrinsics_for_core!();
}

impl GPUDeviceFloatIntrinsics for f32 {}
impl GPUDeviceFloatIntrinsics for f64 {}
