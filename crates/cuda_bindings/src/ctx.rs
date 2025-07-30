use alloc::boxed::Box;
use alloc::vec;
/// The use of Gpu context is guarded by GpuToken, GpuCtxIdToken, GpuCtxActiveToken.
use alloc::vec::Vec;
use core::marker::PhantomData;

use typed_arena::Arena;

use super::unsafe_bindings::*;
use super::{CUDA_SUCCESS, CudaError};
use crate::{AsHostKernelParams, GPUConfig};

macro_rules! eprintln {
    () => {
        panic!()
    };
    ($($arg:tt)*) => {{
        panic!($($arg)*);
    }};
}

/// CtxSpace is a trait to ensure that we can create a GPU context per context space N.
/// This is used to ensure that only one context is created per context space N.
/// After a context with space N is created, we can only create a context with space Succ<N>.
pub trait GpuCtxSpace {}
pub struct CtxSpaceZero;
pub struct Succ<N: GpuCtxSpace>(core::marker::PhantomData<N>);

impl GpuCtxSpace for CtxSpaceZero {}
impl<N: GpuCtxSpace> GpuCtxSpace for Succ<N> {}

// GPU instance token guarantees that the GPU is initialized and can be used.
#[derive(Debug)]
pub struct GpuToken {
    dummy: PhantomData<()>,
}

impl GpuToken {
    /// # Safety
    /// This function is safe if no more than one GPU instance is created across the whole process.
    pub unsafe fn new() -> Self {
        let ret = unsafe { cuInit(0) };
        if ret != CUDA_SUCCESS {
            eprintln!("cuInit fails err = {}", CudaError::Err(ret));
        }
        GpuToken { dummy: PhantomData }
    }
}

/// This token is used to ensure that a single GPU context is active.
pub struct GpuActiveToken {
    _marker: PhantomData<()>,
}

/// This token is used to ensure that a single GPU context is created per context space N.
pub struct GpuCtxIdToken<N: GpuCtxSpace> {
    _marker: PhantomData<N>,
}

/// This is a combo toke to create a GPU context.
#[allow(dead_code)]
pub struct GpuCtxToken<'a, N: GpuCtxSpace> {
    pub token: GpuCtxIdToken<N>,
    pub gpu: &'a GpuToken,
}

/// This is a per-CPU thread token to create and use a GPU context.
pub struct GpuCtxCreateAndUseToken {
    active: GpuActiveToken,
    token: GpuCtxIdToken<CtxSpaceZero>,
}

impl GpuCtxCreateAndUseToken {
    /// # Safety
    /// This function is safe if no more than one GpuCtxCreateAndUseToken is created for the current CPU thread
    pub const unsafe fn new() -> Self {
        GpuCtxCreateAndUseToken {
            active: GpuActiveToken { _marker: PhantomData },
            token: GpuCtxIdToken { _marker: PhantomData },
        }
    }

    pub fn expose(self) -> (GpuActiveToken, GpuCtxIdToken<CtxSpaceZero>) {
        (self.active, self.token)
    }
}

/// We ensure each CPU thread has an unique handle per N.
/// The N is a context namespace ID to guard to access to GPU memory and module.
/// This is used to ensure that only when this context is active, the
/// GPU memory and module created by this context can be accessed.
/// No Copy or Clone.
/// That is, we allow multiple CtxHandle to be created at a single CPU thread,
/// but only one CtxGuard can be used at a time.
#[allow(dead_code)]
pub struct GpuCtxHandle<N: GpuCtxSpace> {
    /// arena must be declared first to ensure it is dropped first.
    arena: Arena<Box<dyn GpuCtxArenaTrait>>,
    inner: GpuCtxHandleInner,
    _marker: PhantomData<N>,
}

impl<N: GpuCtxSpace> core::ops::Deref for GpuCtxHandle<N> {
    type Target = GpuCtxHandleInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GpuCtxHandleInner {
    ctx: CUcontext,
    _flags: CUctx_flags, // TODO: This is not used yet, but we may delete it later.
    _dev: CUdevice,      // TODO: This is not used yet, but we may delete it later.
}

/// GpuCtxGuard<'ctx, 'a, N> can only be created by GpuCtxHandle<'ctx, N> + GpuActiveToken<'a>.
/// At each timepoint, only a single GpuCtxGuard<'ctx, 'a, ?> can be created to represent the active context.
/// This is ensured by the single GpuActiveToken per CPU thread.
/// When creating the CtxGuard, we ensure that the context is pushed to the current thread.
/// When the CtxGuard is dropped, we pop the context from the current thread.
/// No Copy or Clone
#[allow(dead_code)]
pub struct GpuCtxGuard<'ctx, 'a, N: GpuCtxSpace> {
    pub(crate) ctx: &'ctx GpuCtxHandle<N>,
    _marker: PhantomData<&'a &'ctx N>,
}

pub type GpuCtxZeroGuard<'ctx, 'a> = GpuCtxGuard<'ctx, 'a, CtxSpaceZero>;
impl<'ctx, 'a, N: GpuCtxSpace> core::ops::Deref for GpuCtxGuard<'ctx, 'a, N> {
    type Target = GpuCtxHandle<N>;

    fn deref(&self) -> &'ctx Self::Target {
        self.ctx
    }
}

pub(crate) trait GpuCtxArenaTrait: 'static {
    fn as_any(&mut self) -> &mut dyn core::any::Any;
}

impl<'ctx, N: GpuCtxSpace> GpuCtxHandle<N> {
    /// Private constructor to ensure it is only created through `init_gpu`
    /// # Safety
    /// This function is safe if no more than one context is created for the current CPU thread
    /// It requires `GpuToken` to be initialized first.
    /// It checks the error code from CUDA API calls.
    /// GpuCtxToken guarantees that only one context is created per context id N.
    pub fn new<'gpu>(
        c: GpuCtxToken<'gpu, N>,
        dev_id: u32,
        flags: CUctx_flags,
    ) -> (Self, GpuCtxToken<'gpu, Succ<N>>) {
        // # Safety:
        // This function is safe if no more than one context is created for current cpu thread.
        // It requires GpuToken to be initialized first.
        // and it checks the error code from CUDA API calls.
        unsafe {
            let mut ctx: CUcontext = core::mem::zeroed();
            let mut dev: CUdevice = core::mem::zeroed();
            let err = cuDeviceGet(&mut dev as *mut _ as _, dev_id as _);
            if err != CUDA_SUCCESS {
                panic!("Failed to get device ({}): {}", dev_id, CudaError::Err(err));
            }
            let err = cuCtxCreate_v2(&mut ctx as *mut _ as _, flags as _, dev);
            if err != CUDA_SUCCESS {
                panic!(
                    "Failed to create context ({}, {:?}): {}",
                    dev_id,
                    flags,
                    CudaError::Err(err)
                );
            }
            (
                GpuCtxHandle::<N> {
                    arena: Arena::new(),
                    inner: GpuCtxHandleInner { ctx, _flags: flags, _dev: dev },
                    _marker: PhantomData,
                },
                GpuCtxToken { gpu: c.gpu, token: GpuCtxIdToken { _marker: PhantomData } },
            )
        }
    }

    #[allow(private_bounds)]
    pub fn alloc_typed<T: GpuCtxArenaTrait + 'ctx>(&'ctx self, v: T) -> &'ctx mut T {
        let any: &mut dyn GpuCtxArenaTrait = self.arena.alloc(Box::new(v)).as_mut();
        any.as_any().downcast_mut::<T>().expect("Failed to downcast to T")
    }

    // Since we only have a single GpuActiveToken per CPU thread,
    // we guarantee that only one GpuCtx is alive at a time.
    pub fn activate<'a>(&'ctx self, _active: &'a mut GpuActiveToken) -> GpuCtxGuard<'ctx, 'a, N> {
        // Safety: This is safe since we ensure that only one context is created per CPU thread.
        unsafe {
            cuCtxPushCurrent_v2(self.ctx);
        }
        GpuCtxGuard { ctx: self, _marker: PhantomData }
    }
}

impl Drop for GpuCtxHandleInner {
    fn drop(&mut self) {
        let ret;
        unsafe {
            ret = cuCtxDestroy_v2(self.ctx);
        }
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to destroy GPU_CTX: {}", CudaError::Err(ret));
        }
    }
}

impl<'ctx, 'a, N: GpuCtxSpace> Drop for GpuCtxGuard<'ctx, 'a, N> {
    fn drop(&mut self) {
        let ret;
        unsafe {
            let mut ctx: CUcontext = core::mem::zeroed();
            ret = cuCtxPopCurrent_v2(&mut ctx);
        }
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to pop GPU_CTX: {}", CudaError::Err(ret));
        }
    }
}

#[derive(Debug)]
pub struct GpuModule<N: GpuCtxSpace> {
    module: CUmodule,
    _marker: PhantomData<N>,
}

impl<N: GpuCtxSpace + 'static> GpuCtxArenaTrait for GpuModule<N> {
    fn as_any(&mut self) -> &mut (dyn core::any::Any) {
        self
    }
}

impl<N: GpuCtxSpace> Drop for GpuModule<N> {
    fn drop(&mut self) {
        let ret = unsafe { cuModuleUnload(self.module) };
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to unload GPU_CTX module: {}", CudaError::Err(ret));
        }
    }
}

/// A Cuda function can only be used in the context creating the module.
pub struct GpuFunction<'ctx, N: GpuCtxSpace> {
    func: CUfunction,
    _marker: PhantomData<&'ctx N>,
}

pub struct CudaStream {
    raw: CUstream,
}

impl<'ctx, 'a, N: GpuCtxSpace + 'static> GpuCtxGuard<'ctx, 'a, N> {
    /// Block for the current context's tasks to complete.
    pub fn sync(&self) -> Result<(), CudaError> {
        let ret;
        unsafe {
            ret = cuCtxSynchronize();
        }
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Err(ret));
        }
        Ok(())
    }

    pub fn get_func(
        &self,
        m: &GpuModule<N>,
        func_name: &str,
    ) -> Result<GpuFunction<'ctx, N>, CudaError> {
        let func_name_cstr = alloc::ffi::CString::new(func_name).expect("Failed to create CString");
        let mut f: CUfunction;
        let ret;
        unsafe {
            f = core::mem::zeroed();
            ret = cuModuleGetFunction(&mut f as *mut _ as _, m.module, func_name_cstr.as_ptr());
        }
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Err(ret));
        }
        Ok(GpuFunction { func: f, _marker: PhantomData })
    }

    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu_macros::host to generate dummy api checking code.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_coop_kernel(
        &self,
        m: &GpuModule<N>,
        func_name: &str,
        config: GPUConfig,
        shared_mem_size: usize,
        host_args: &Vec<&dyn AsHostKernelParams>,
        stream: Option<&CudaStream>,
        is_async: bool,
    ) -> Result<(), CudaError> {
        let f = self.get_func(m, func_name)?;
        self.launch_coop_fn(&f, config, shared_mem_size, host_args, stream, is_async)
    }

    /// The host must use arguments that implement `AsHostKernelParams`.
    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu_macros::host to generate dummy api checking code.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_kernel(
        &self,
        m: &GpuModule<N>,
        func_name: &str,
        config: GPUConfig,
        shared_mem_size: usize,
        host_args: &Vec<&dyn AsHostKernelParams>,
        stream: Option<&CudaStream>,
        is_async: bool,
    ) -> Result<(), CudaError> {
        let f = self.get_func(m, func_name)?;
        self.launch_fn(&f, config, shared_mem_size, host_args, stream, is_async)
    }

    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu_macros::host to generate dummy api checking code.
    pub unsafe fn launch_coop_fn(
        &self,
        f: &GpuFunction<'ctx, N>,
        config: GPUConfig,
        shared_mem_size: usize,
        host_args: &Vec<&dyn AsHostKernelParams>,
        stream: Option<&CudaStream>,
        is_async: bool,
    ) -> Result<(), CudaError> {
        let mut kernel_data = vec![];

        host_args.iter().for_each(|arg| {
            kernel_data.extend(arg.as_kernel_param_data());
        });

        let mut kernel_args = kernel_data
            .into_iter()
            .map(|data| Box::into_raw(data) as *mut core::ffi::c_void)
            .collect::<Vec<_>>();

        let res = unsafe {
            cuLaunchCooperativeKernel(
                f.func,
                config.grid_dim_x,
                config.grid_dim_y,
                config.grid_dim_z,
                config.block_dim_x,
                config.block_dim_y,
                config.block_dim_z,
                shared_mem_size as u32,
                stream.map_or(core::ptr::null_mut(), |s| s.raw),
                kernel_args.as_mut_ptr() as _,
            )
        };

        if res != CUDA_SUCCESS {
            return Err(CudaError::Err(res as _));
        }
        if !is_async {
            self.sync()?;
        }
        Ok(())
    }

    /// The host must use arguments that implement `AsHostKernelParams`.
    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu_macros::host to generate dummy api checking code.
    pub unsafe fn launch_fn(
        &self,
        f: &GpuFunction<'ctx, N>,
        config: GPUConfig,
        shared_mem_size: usize,
        host_args: &Vec<&dyn AsHostKernelParams>,
        stream: Option<&CudaStream>,
        is_async: bool,
    ) -> Result<(), CudaError> {
        let mut kernel_data = vec![];

        host_args.iter().for_each(|arg| {
            kernel_data.extend(arg.as_kernel_param_data());
        });

        let mut kernel_args = kernel_data
            .into_iter()
            .map(|data| Box::into_raw(data) as *mut core::ffi::c_void)
            .collect::<Vec<_>>();

        let res = unsafe {
            cuLaunchKernel(
                f.func,
                config.grid_dim_x,
                config.grid_dim_y,
                config.grid_dim_z,
                config.block_dim_x,
                config.block_dim_y,
                config.block_dim_z,
                shared_mem_size as u32,
                stream.map_or(core::ptr::null_mut(), |s| s.raw),
                kernel_args.as_mut_ptr() as _,
                core::ptr::null_mut(),
            )
        };

        if res != CUDA_SUCCESS {
            return Err(CudaError::Err(res as _));
        }
        if !is_async {
            self.sync()?;
        }
        Ok(())
    }

    pub fn new_module(&self, bin: *const u8) -> Result<&mut GpuModule<N>, CudaError> {
        unsafe {
            let mut m: CUmodule = core::mem::zeroed();
            let ret = cuModuleLoadData(&mut m as *mut _ as _, bin as _);
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
            let m = GpuModule { module: m, _marker: PhantomData };
            Ok(self.ctx.alloc_typed(m))
        }
    }
}
