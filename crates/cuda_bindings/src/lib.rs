// Suppress the flurry of warnings caused by using "C" naming conventions
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![no_std]

extern crate alloc;
use alloc::string::{String, ToString};
mod chunk;
mod ctx;
mod mem;
mod params;

// This matches bindgen::Builder output
#[allow(dead_code)]
#[allow(unused_imports)]
#[allow(improper_ctypes)]
#[doc(hidden)]
mod unsafe_bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use chunk::{GpuChunkable, GpuChunkable2D, GpuChunkableMut, GpuChunkableMut2D};
pub use ctx::{
    CtxSpaceZero, GpuActiveToken, GpuCtxCreateAndUseToken, GpuCtxGuard, GpuCtxHandle, GpuCtxSpace,
    GpuCtxToken, GpuCtxZeroGuard, GpuFunction, GpuModule, GpuToken,
};
pub use mem::CudaMemBox;
pub use params::{AsHostKernelParams, GPUConfig};
pub use unsafe_bindings::CUctx_flags;

#[allow(unused_imports)]
use crate::unsafe_bindings::*; // Private;

pub const CUDA_SUCCESS: CUresult = CUresult::CUDA_SUCCESS;

pub enum CudaError {
    Err(CUresult),
    Unknown(String),
    MemCopyOutOfBound,
}

fn get_cuda_error_name(error: CUresult) -> String {
    let mut c_str_ptr: *const core::ffi::c_char = core::ptr::null();
    let result = unsafe { cuGetErrorName(error, &mut c_str_ptr as *mut _) };

    if result == CUDA_SUCCESS && !c_str_ptr.is_null() {
        let c_str = unsafe { core::ffi::CStr::from_ptr(c_str_ptr) };
        c_str.to_string_lossy().into_owned()
    } else {
        "<Unknown CUDA Error>".to_string()
    }
}

impl core::fmt::Debug for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaError::Err(code) => {
                let msg = get_cuda_error_name(*code);
                write!(f, "CUDA Error: {}", msg)
            }
            CudaError::MemCopyOutOfBound => write!(f, "CUDA Error: Memory copy out of bounds"),
            CudaError::Unknown(msg) => write!(f, "CUDA Error: {}", msg),
        }
    }
}

impl core::fmt::Display for CudaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[macro_export]
macro_rules! load_module_from_extern {
    ($ctx: expr, $name: ident) => {{
        extern "C" {
            static $name: u8;
        }
        $ctx.new_module(&$name)
    }};
}
