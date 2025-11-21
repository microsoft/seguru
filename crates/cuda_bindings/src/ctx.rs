/// The use of Gpu context is guarded by GpuToken, GpuCtxIdToken, GpuCtxActiveToken.
/// This module exposes 4 unsafe functions:
/// - `GpuToken::new()`
/// - `GpuCtxCreateAndUseToken::new()`
/// - `GpuCtxGuard::launch_kernel()`
/// - `GpuCtxGuard::launch_coop_kernel()`
///
/// Those functions should not be used directly.
/// Instead, use the safe wrappers in `gpu_host` crate.
/// and use `gpu::host` to generate host code.
use alloc::boxed::Box;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use typed_arena::Arena;

use super::unsafe_bindings::*;
use super::{CUDA_SUCCESS, CudaError};
use crate::{AsHostKernelParams, SafeGpuConfig};

#[cfg(feature = "gpu")]
macro_rules! eprintln {
    ($($arg:tt)*) => {{}};
}

/// CtxSpace is a trait to ensure that we can create a GPU context per context space N.
/// This is used to ensure that only one context is created per context space N.
/// After a context with space N is created, we can only create a context with space `Succ<N>`.
pub trait GpuCtxSpace: 'static {
    fn is_primary() -> bool {
        false
    }
}
pub struct CtxSpaceZero;
pub struct Succ<N: GpuCtxSpace>(core::marker::PhantomData<N>);

impl GpuCtxSpace for CtxSpaceZero {
    fn is_primary() -> bool {
        true
    }
}
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
    dev_prop: CUdevprop, // TODO: Put it to dev-specific struct.
    major: i32,
    minor: i32,
    _flags: CUctx_flags, // TODO: This is not used yet, but we may delete it later.
    _dev: CUdevice,      // TODO: This is not used yet, but we may delete it later.
}

impl GpuCtxHandleInner {
    pub fn get_dev_prop(&self) -> &CUdevprop {
        &self.dev_prop
    }

    pub fn get_compute_capability(&self) -> (i32, i32) {
        (self.major, self.minor)
    }

    pub fn get_ctx(&self) -> CUcontext {
        self.ctx
    }

    pub fn get_dev(&self) -> CUdevice {
        self._dev
    }
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
        ctx_token: GpuCtxToken<'gpu, N>,
        dev_id: u32,
        flags: CUctx_flags,
    ) -> (Self, GpuCtxToken<'gpu, Succ<N>>) {
        let mut err;

        // Safety: It is safe to run since we provided correct CUdevice type
        // and GpuToken is valid indicating that CUDA is initialized.
        // We check the error code to ensure that the dev_ptr is initialized.
        let mut dev_uninit = MaybeUninit::<CUdevice>::uninit();
        let dev = unsafe {
            err = cuDeviceGet(dev_uninit.as_mut_ptr(), dev_id as _);
            assert!(
                err == CUDA_SUCCESS,
                "Failed to get device({}): error {}",
                dev_id,
                CudaError::Err(err)
            );
            dev_uninit.assume_init()
        };
        let mut current_ctx_uninit = MaybeUninit::<CUcontext>::uninit();
        // Safety: It is safe since we provide valid CUcontext pointer and dev is valid.
        let current_ctx = unsafe {
            err = cuCtxGetCurrent(current_ctx_uninit.as_mut_ptr());
            assert!(
                err == CUDA_SUCCESS,
                "Failed to create context ({}, {:?}): error {}",
                dev_id,
                flags,
                CudaError::Err(err)
            );
            current_ctx_uninit.assume_init()
        };
        let ctx = if !current_ctx.is_null() && N::is_primary() {
            current_ctx
        } else {
            let mut ctx_uninit = MaybeUninit::<CUcontext>::uninit();
            #[cfg(cuda_has_ctx_create_v4)]
            let mut create_params = CUctxCreateParams_st {
                numExecAffinityParams: 0,
                execAffinityParams: core::ptr::null_mut(),
                cigParams: core::ptr::null_mut(),
            };
            // Safety: It is safe since we provide valid CUctxCreateParams and CUcontext pointers and dev is valid.
            unsafe {
                #[cfg(cuda_has_ctx_create_v4)]
                {
                    err = cuCtxCreate_v4(
                        ctx_uninit.as_mut_ptr(),
                        &mut create_params as *mut _,
                        flags as _,
                        dev,
                    );
                }
                #[cfg(not(cuda_has_ctx_create_v4))]
                {
                    err = cuCtxCreate_v2(ctx_uninit.as_mut_ptr(), flags as _, dev);
                }

                if err != CUDA_SUCCESS {
                    panic!(
                        "Failed to create context ({}, {:?}): error {}",
                        dev_id,
                        flags,
                        CudaError::Err(err)
                    );
                }
                ctx_uninit.assume_init()
            }
        };

        let mut dev_prop_uninit = MaybeUninit::<CUdevprop>::uninit();
        // Safety: It is safe since dev_prop is valid and dev is valid.
        let dev_prop = unsafe {
            cuDeviceGetProperties(dev_prop_uninit.as_mut_ptr(), dev_id as _);
            assert!(err == CUDA_SUCCESS, "Failed to get device({}) properties", dev_id);
            dev_prop_uninit.assume_init()
        };

        let (major, minor) = unsafe {
            let mut major_uninit = MaybeUninit::<i32>::uninit();
            let mut minor_uninit = MaybeUninit::<i32>::uninit();
            err = cuDeviceComputeCapability(
                major_uninit.as_mut_ptr(),
                minor_uninit.as_mut_ptr(),
                dev,
            );
            assert!(err == CUDA_SUCCESS, "Failed to get device({}) compute capability", dev_id);
            (major_uninit.assume_init(), minor_uninit.assume_init())
        };
        (
            GpuCtxHandle::<N> {
                arena: Arena::new(),
                inner: GpuCtxHandleInner { ctx, dev_prop, major, minor, _flags: flags, _dev: dev },
                _marker: PhantomData,
            },
            GpuCtxToken { gpu: ctx_token.gpu, token: GpuCtxIdToken { _marker: PhantomData } },
        )
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
        let mut ctx_uninit = MaybeUninit::<CUcontext>::uninit();
        // Safety: This is safe since we provide valid CUcontext pointer.
        unsafe {
            ret = cuCtxPopCurrent_v2(ctx_uninit.as_mut_ptr());
        }
        // We do not want to panic in drop since it may cause double panic.
        // For example, when the context is poisoned by kernel launch failure,
        // the drop of GpuCtxHandleInner will panic, and if we panic here again
        if ret != CUDA_SUCCESS {
            // do not use panic since it will cause a double panic.
            eprintln!("Failed to pop GPU_CTX: {}", CudaError::Err(ret));
        }
    }
}

#[derive(Debug)]
pub struct GpuModule<N: GpuCtxSpace> {
    module: CUmodule,
    cached_functions: BTreeMap<String, (CUfunction, i32)>,
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
    max_dyn_shared_size: i32,
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
        if let Some(&(func, max_dyn_shared_size)) = m.cached_functions.get(func_name) {
            return Ok(GpuFunction { func, max_dyn_shared_size, _marker: PhantomData });
        }

        let func_name_cstr = alloc::ffi::CString::new(func_name).expect("Failed to create CString");
        let ret;
        let mut func_uninit = MaybeUninit::<CUfunction>::uninit();

        // Safety: it is safe since m.module is valid and error is checked before using result.
        let func = unsafe {
            ret = cuModuleGetFunction(func_uninit.as_mut_ptr(), m.module, func_name_cstr.as_ptr());
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
            func_uninit.assume_init()
        };

        let mut max_dyn_shared_size_uninit = MaybeUninit::uninit();

        // Safety: it is safe since func is valid and error is checked before using result.
        let max_dyn_shared_size = unsafe {
            let ret = cuFuncGetAttribute(
                max_dyn_shared_size_uninit.as_mut_ptr(),
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                func,
            );
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
            max_dyn_shared_size_uninit.assume_init()
        };
        Ok(GpuFunction { func, max_dyn_shared_size, _marker: PhantomData })
    }

    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu::host to generate dummy api checking code.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_coop_kernel<C: SafeGpuConfig>(
        &self,
        m: &GpuModule<N>,
        func_name: &str,
        config: C,
        host_args: &[&dyn AsHostKernelParams],
        stream: Option<&CudaStream>,
    ) -> Result<(), CudaError> {
        let f = self.get_func(m, func_name)?;
        self.launch_coop_fn(&f, config, host_args, stream)
    }

    /// The host must use arguments that implement `AsHostKernelParams`.
    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu::host to generate dummy api checking code.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_kernel<C: SafeGpuConfig>(
        &self,
        m: &GpuModule<N>,
        func_name: &str,
        config: C,
        host_args: &[&dyn AsHostKernelParams],
        stream: Option<&CudaStream>,
    ) -> Result<(), CudaError> {
        let f = self.get_func(m, func_name)?;
        self.launch_fn(&f, config, host_args, stream)
    }

    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu::host to generate dummy api checking code.
    unsafe fn launch_coop_fn<C: SafeGpuConfig>(
        &self,
        f: &GpuFunction<'ctx, N>,
        config: C,
        host_args: &[&dyn AsHostKernelParams],
        stream: Option<&CudaStream>,
    ) -> Result<(), CudaError> {
        config.runtime_check(self.dev_prop, f.max_dyn_shared_size);
        let mut kernel_args: Vec<*mut core::ffi::c_void> = Vec::with_capacity(host_args.len());
        host_args.iter().for_each(|arg| arg.as_kernel_param_data(&mut kernel_args));
        if config.shared_size() != 0 {
            kernel_args
                .push(Box::into_raw(Box::new(config.shared_size())) as *mut core::ffi::c_void);
        }

        let res = unsafe {
            cuLaunchCooperativeKernel(
                f.func,
                config.grid_dim_x(),
                config.grid_dim_y(),
                config.grid_dim_z(),
                config.block_dim_x(),
                config.block_dim_y(),
                config.block_dim_z(),
                config.shared_size(),
                stream.map_or(core::ptr::null_mut(), |s| s.raw),
                kernel_args.as_mut_ptr() as _,
            )
        };

        if res != CUDA_SUCCESS {
            return Err(CudaError::Err(res as _));
        }
        Ok(())
    }

    /// The host must use arguments that implement `AsHostKernelParams`.
    /// # Safety
    /// This is safe if the argument types match the kernel's expected types.
    /// To make it safe, use the gpu::host to generate dummy api checking code.
    unsafe fn launch_fn<C: SafeGpuConfig>(
        &self,
        f: &GpuFunction<'ctx, N>,
        config: C,
        host_args: &[&dyn AsHostKernelParams],
        stream: Option<&CudaStream>,
    ) -> Result<(), CudaError> {
        config.runtime_check(self.dev_prop, f.max_dyn_shared_size);
        let mut kernel_args: Vec<*mut core::ffi::c_void> = Vec::with_capacity(host_args.len());
        host_args.iter().for_each(|arg| arg.as_kernel_param_data(&mut kernel_args));
        let res = unsafe {
            cuLaunchKernel(
                f.func,
                config.grid_dim_x(),
                config.grid_dim_y(),
                config.grid_dim_z(),
                config.block_dim_x(),
                config.block_dim_y(),
                config.block_dim_z(),
                config.shared_size(),
                stream.map_or(core::ptr::null_mut(), |s| s.raw),
                kernel_args.as_mut_ptr() as _,
                core::ptr::null_mut(),
            )
        };

        if res != CUDA_SUCCESS {
            return Err(CudaError::Err(res as _));
        }
        Ok(())
    }

    pub fn new_module(&self, bin: *const u8) -> Result<&mut GpuModule<N>, CudaError> {
        let mut module_uninit = MaybeUninit::<CUmodule>::uninit();
        let module = unsafe {
            let ret = cuModuleLoadData(module_uninit.as_mut_ptr(), bin as _);
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
            module_uninit.assume_init()
        };
        // This is safe since module is valid.
        let num_functions = unsafe {
            let mut count = 0;
            let ret = cuModuleGetFunctionCount(&mut count, module);
            assert!(ret == CUDA_SUCCESS, "Failed to get function count");
            count
        };

        let mut functions = vec![core::ptr::null_mut() as CUfunction; num_functions as usize];
        // This is safe since it will fill in kernel function pointers once no errors.
        unsafe {
            let ret = cuModuleEnumerateFunctions(functions.as_mut_ptr(), num_functions, module);
            if ret != CUDA_SUCCESS {
                return Err(CudaError::Err(ret));
            }
        }

        let mut cached_functions = BTreeMap::new();
        for f in functions {
            let mut name_ptr = MaybeUninit::<*const ::core::ffi::c_char>::uninit();
            // This is safe since f is valid.
            let name = unsafe {
                let ret = cuFuncGetName(name_ptr.as_mut_ptr(), f);
                assert!(ret == CUDA_SUCCESS, "Failed to get function name");
                let c_str = core::ffi::CStr::from_ptr(name_ptr.assume_init());
                c_str.to_str().unwrap().to_string()
            };
            unsafe {
                let ret = cuFuncLoad(f);
                assert!(ret == CUDA_SUCCESS, "Failed to load function");
            }
            // Safety: it is safe since func is valid and error is checked before using result.
            let mut max_dyn_shared_size_uninit = MaybeUninit::uninit();
            let max_dyn_shared_size = unsafe {
                let ret = cuFuncGetAttribute(
                    max_dyn_shared_size_uninit.as_mut_ptr(),
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    f,
                );
                assert!(ret == CUDA_SUCCESS, "Failed to get function attribute");
                max_dyn_shared_size_uninit.assume_init()
            };
            cached_functions.insert(name, (f, max_dyn_shared_size));
        }
        let m = GpuModule { module, cached_functions, _marker: PhantomData };
        Ok(self.ctx.alloc_typed(m))
    }
}
