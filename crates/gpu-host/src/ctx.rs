use cuda_bindings::{
    CUctx_flags, CtxSpaceZero, GpuActiveToken, GpuCtxCreateAndUseToken, GpuCtxHandle, GpuCtxToken,
    GpuCtxZeroGuard, GpuModule, GpuToken,
};

static GPU: std::sync::OnceLock<GpuToken> = std::sync::OnceLock::new();

thread_local! {
    /// Create a thread-local GPU_CTX context
    /// Cuda allow us to create a single context per CPU thread, so we use a thread-local variable to store it.
    /// SAFETY: This is safe because we ensure only one context is created per thread.
    static CTX_TOKEN: core::cell::RefCell<Option<GpuCtxCreateAndUseToken>> = const { core::cell::RefCell::new(Some(unsafe {GpuCtxCreateAndUseToken::new()})) };
}

/// All CUDA operations must be done in cuda_scope created by this function.
/// It allows developers to use mutiple GPU contexts in a single CPU thread.
/// The use of different GPU contexts is guarded by GpuCtxToken and GpuActiveToken.
/// It provides a safe and flexible way to manage GPU contexts than Cuda Runtime API.
/// It statically ensures that
///     1. Multiple GPU contexts for the same device to be created in a single CPU thread.
///     2. Only one GPU context can be active at a time in a single CPU thread.
///     3. no cross-context memory or module access.
///     4. no cross-context kernel launch.
///     5. Safe memory creation and destroy.
/// Thus the context is guaranteed to be used safely when code compiles.
pub fn cuda_scope<T>(
    f: impl for<'a> FnOnce(GpuCtxToken<'a, CtxSpaceZero>, &mut GpuActiveToken) -> T,
) -> T {
    // SAFETY: This function is safe since no more than one GPU context is created
    let instance = GPU.get_or_init(|| unsafe { GpuToken::new() });
    CTX_TOKEN.with(|token| {
        let ret = if let Some(token) = token.borrow_mut().take() {
            // If we already have an active context, we can use it.
            let (mut active, id_token) = token.expose();
            let ct = GpuCtxToken {
                token: id_token,
                gpu: instance,
            };
            f(ct, &mut active)
        } else {
            panic!("No active GPU context found. Ensure cuda_scope is not called inside another cuda_scope.");
        };
        token.borrow_mut().replace(unsafe{GpuCtxCreateAndUseToken::new()});
        ret
    })
}

/// This function is used to create a single GPU context with the given context space CtxSpaceZero.
/// It is useful in most cases where a process only needs a single GPU context.
pub fn cuda_ctx<T>(
    dev_id: u32,
    f: impl for<'ctx, 'a> FnOnce(&GpuCtxZeroGuard<'ctx, 'a>, &'ctx GpuModule<CtxSpaceZero>) -> T,
) -> T {
    // SAFETY: This function is safe since no more than one GPU context is created
    let instance = GPU.get_or_init(|| unsafe { GpuToken::new() });
    CTX_TOKEN.with(|token| {
        let ret = if let Some(token) = token.borrow_mut().take() {
            // If we already have an active context, we can use it.
            let (mut active, id_token) = token.expose();
            let ct = GpuCtxToken {
                token: id_token,
                gpu: instance,
            };
            let (ctx_h, _) = GpuCtxHandle::<CtxSpaceZero>::new(ct, dev_id, CUctx_flags::CU_CTX_SCHED_AUTO);
            let ctx = ctx_h.activate(&mut active);
            // This is safe since our rustc-gpu will generate gpu_bin_cst
            let m = unsafe { crate::load_module_from_extern!(ctx, gpu_bin_cst).expect("Failed to load default gpu_bin_cst module. Please use gpu_macros + rustc_gpu to compile.") };
            f(&ctx, m)
        } else {
            panic!("No active GPU context found. Ensure cuda_scope is not called inside another cuda_scope.");
        };
        // This is safe since the above session ends and so function is executed at current ctx.
        token.borrow_mut().replace(unsafe{GpuCtxCreateAndUseToken::new()});
        ret
    })
}
