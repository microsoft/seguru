// Suppress the flurry of warnings caused by using "C" naming conventions
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(register_tool)]
#![register_tool(gpu_codegen)]

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
    GpuCtxGuard, GpuCtxHandle, GpuCtxSpace, GpuCtxZeroGuard, GpuFunction, GpuModule, cuda_ctx,
    cuda_scope,
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
        let c_str = unsafe { std::ffi::CStr::from_ptr(c_str_ptr) };
        c_str.to_string_lossy().into_owned()
    } else {
        "<Unknown CUDA Error>".to_string()
    }
}

impl std::fmt::Debug for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
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

#[test]
fn test_cuda_mem_single_ctx() {
    let value1: [f32; 10] = [123f32; 10];
    let mut value2: [f32; 10] = [456f32; 10];
    assert!(value1 != value2);
    cuda_ctx(0, |ctx| {
        let x = ctx.new_gmem_with_len::<f32>(10).unwrap();
        x.copy_from_host(&value1, 10, ctx).expect("Failed to copy memory to host");
        x.copy_to_host(&mut value2, 10, ctx).unwrap();
    });
    assert!(value1 == value2);
}
#[test]
fn test_cuda_mem_multiple_ctx() {
    cuda_scope(|ct, active| {
        let value1: u32 = 123u32;
        let mut value2: u32 = 456u32;
        let (ctx_h, ct) = GpuCtxHandle::new(ct, 0, CUctx_flags::CU_CTX_SCHED_AUTO);
        let x = {
            let ctx = ctx_h.activate(active);
            let x = ctx.new_gmem::<u32>().unwrap();
            x.copy_from_host(&value1, &ctx).expect("Failed to copy memory to host");
            let _ = ctx.new_gmem_with_len::<u32>(10);
            x
        };
        let (ctx_h2, _) = GpuCtxHandle::new(ct, 0, CUctx_flags::CU_CTX_SCHED_AUTO);
        {
            let ctx2 = ctx_h2.activate(active);
            let z = ctx2.new_gmem_with_len::<u32>(10).unwrap();
            z.copy_from_host(&[0; 10], 10, &ctx2).expect("Failed to copy memory to host");
        }
        let ctx1: GpuCtxGuard<'_, '_, _> = ctx_h.activate(active);
        x.copy_to_host(&mut value2, &ctx1).unwrap();
        assert!(value1 == value2);
    });
}

#[test]
#[should_panic(expected = "already borrowed: BorrowMutError")]
fn test_cuda_no_nested_scope() {
    cuda_scope(|_, _| {
        cuda_scope(|_, _| {});
    })
}
