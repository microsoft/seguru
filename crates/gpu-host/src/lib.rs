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
        let x = ctx.new_tensor_view(&value1).unwrap();
        x.copy_to_host(&mut value2).unwrap();
    });
    assert!(value1 == value2, "value1: {:?}, value2: {:?}", value1, value2);
}

#[test]
fn test_tensor_index() {
    let value1: [f32; 10] = (0..10).map(|x| x as f32).collect::<Vec<_>>().try_into().unwrap();
    cuda_ctx_no_mod(0, |ctx| {
        let mut x = ctx.new_tensor_view(&value1).unwrap();
        for i in 0..9 {
            let mut e = x.index_mut(i..i + 2);
            let x = &mut e;
            assert!(x.len() == 2);
            let mut host_value = [0.0f32; 2];
            assert!(x.copy_to_host(&mut host_value).is_ok());
            assert!(host_value[0] == i as f32);
            assert!(host_value[1] == (i + 1) as f32);
        }
    });
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
            let x = ctx.new_tensor::<u32>(&value1).unwrap();
            let _ = ctx.new_tensor_view(&[0; 10]);
            x
        };
        let (ctx_h2, _) = GpuCtxHandle::new(ct, 0, CUctx_flags::CU_CTX_SCHED_AUTO);
        {
            let ctx2 = ctx_h2.activate(active);
            let _ = ctx2.new_tensor(&[0; 10]).unwrap();
        }
        let ctx1: GpuCtxGuard<'_, '_, _> = ctx_h.activate(active);
        x.as_tensor_view_mut(&ctx1).copy_to_host(&mut value2).unwrap();
        assert!(value1 == value2);
    });
}

#[test]
#[should_panic(expected = "RefCell already borrowed")]
fn test_cuda_no_nested_scope() {
    cuda_scope(|_, _| {
        cuda_scope(|_, _| {});
    })
}
