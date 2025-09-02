mod ctx;

pub use ctx::{cuda_ctx, cuda_ctx_no_mod, cuda_scope};
pub use cuda_bindings::*;

pub fn get_fn_name<T>(_: T) -> String {
    let name = std::any::type_name::<T>();
    gpu_name::convert_def_path_to_gpu_sym_name(name)
}

#[test]
fn test_cuda_mem_single_ctx() {
    let value1: [f32; 10] = [123f32; 10];
    let mut value2: [f32; 10] = [456f32; 10];
    assert!(value1 != value2);
    cuda_ctx_no_mod(0, |ctx| {
        let x = ctx.new_gmem_with_len::<f32>(10, &[0f32; 10]).unwrap();
        x.copy_from_host(&value1, 10, ctx).expect("Failed to copy memory to host");
        x.copy_to_host(&mut value2, 10, ctx).unwrap();
    });
    assert!(value1 == value2);
}

#[test]
#[should_panic(expected = "Failed to load default gpu_bin_cst module.")]
fn test_cuda_load_wrong_mod() {
    #[unsafe(no_mangle)]
    static gpu_bin_cst: [u8; 0] = [];
    cuda_ctx(0, |_, _| {});
}

#[test]
fn test_cuda_mem_multiple_ctx() {
    cuda_scope(|ct, active| {
        let value1: u32 = 123u32;
        let mut value2: u32 = 456u32;
        let (ctx_h, ct) = GpuCtxHandle::new(ct, 0, CUctx_flags::CU_CTX_SCHED_AUTO);
        let x = {
            let ctx = ctx_h.activate(active);
            let x = ctx.new_gmem::<u32>(0).unwrap();
            x.copy_from_host(&value1, &ctx).expect("Failed to copy memory to host");
            let _ = ctx.new_gmem_with_len::<u32>(10, &[0; 10]);
            x
        };
        let (ctx_h2, _) = GpuCtxHandle::new(ct, 0, CUctx_flags::CU_CTX_SCHED_AUTO);
        {
            let ctx2 = ctx_h2.activate(active);
            let z = ctx2.new_gmem_with_len::<u32>(10, &[0; 10]).unwrap();
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
