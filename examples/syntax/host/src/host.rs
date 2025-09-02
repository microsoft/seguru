#![allow(clippy::too_many_arguments)]

/*
mod internal {
    /// Manually inserted function to test the host side API matches the GPU side API.
    /// This should be generated automatically by the macro.
    /// This is needed in order to force the compiler to link the GPU code.
    #[used]
    static FORCE_LINK: fn(
        a: gpu::GpuChunkable2D<u32>,
        b: gpu::GpuChunkableMut2D<u32>,
        c: &gpu_host::CudaMemBox<[u32]>,
        f: gpu::GpuChunkableMut<f32>,
        g: &gpu_host::CudaMemBox<[f32]>,
    ) = dummy_api_checker_kernel_launch_wrapper;
    fn dummy_api_checker_kernel_launch_wrapper(
        a: gpu::GpuChunkable2D<u32>,
        b: gpu::GpuChunkableMut2D<u32>,
        c: &gpu_host::CudaMemBox<[u32]>,
        f: gpu::GpuChunkableMut<f32>,
        g: &gpu_host::CudaMemBox<[f32]>,
    ) {
        manual_test_gpu_arith::kernel_arith::<30>(a, b, c, f, g);
    }
}
*/
use syntax_gpu::kernel_arith as kernel_arith_gpu;
#[gpu_macros::host(kernel_arith_gpu::<30>)]
pub fn kernel_arith(
    a: gpu::GpuChunkable2D<u32>,
    b: gpu::GpuChunkableMut2D<u32>,
    c: &gpu_host::CudaMemBox<[u32]>,
    f: &mut gpu_host::CudaMemBox<[f32]>,
    f_width: usize,
    g: &gpu_host::CudaMemBox<[f32]>,
) {
}

/*
#[allow(non_upper_case_globals)]
const const_share_size_kernel_arith: usize = manual_test_gpu_arith::shared_size_kernel_arith;
pub fn kernel_arith(
    a: gpu_host::GpuChunkable<u32>,
    b: gpu_host::GpuChunkableMut<u32>,
    c: &gpu_host::CudaMemBox<[u32]>,
    f: gpu_host::GpuChunkableMut<f32>,
    g: &gpu_host::CudaMemBox<[f32]>,
) -> Result<(), gpu_host::CudaError> {
    let config = gpu_host::GPUConfig {
        grid_dim_x: 1,
        grid_dim_y: 1,
        grid_dim_z: 1,
        block_dim_x: 4,
        block_dim_y: 1,
        block_dim_z: 1,
    };
    let mut args_for_launching: Vec<&dyn AsKernelParams> = vec![];
    args_for_launching.push(&a);
    args_for_launching.push(&b);
    args_for_launching.push(&c);
    args_for_launching.push(&f);
    args_for_launching.push(&g);
    let func_name_cstr = std::ffi::CString::new("kernel_arith").unwrap();
    let res = gpu_host::launch_kernel(
        "kernel_arith",
        config,
        const_share_size_kernel_arith,
        &args_for_launching,
        true,
    )?;
    Ok(())
}
// This wrapper should be generated automatically. Note that it should have known the
// kernel function's name!
//fn kernel_launch_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) -> i32 {
//    // Build the args array
//    let a_ptr = a.as_ptr() as *const ::std::os::raw::c_void;
//    let a_ptr_rptr: *const *const ::std::os::raw::c_void = &a_ptr;
//    let a_ptr_ptr = a_ptr_rptr as *const ::std::os::raw::c_void;
//    let a_size = a.len();
//    let a_size_rptr: *const usize = &a_size;
//    let a_size_ptr: *const ::std::os::raw::c_void = a_size_rptr as *const ::std::os::raw::c_void;
//    let a_window_rptr: *const usize = &a_window;
//    let a_window_ptr: *const ::std::os::raw::c_void =
//        a_window_rptr as *const ::std::os::raw::c_void;
//    let b_ptr = b.as_ptr() as *const ::std::os::raw::c_void;
//    let b_ptr_rptr: *const *const ::std::os::raw::c_void = &b_ptr;
//    let b_ptr_ptr = b_ptr_rptr as *const ::std::os::raw::c_void;
//    let b_size = b.len();
//    let b_size_rptr: *const usize = &b_size;
//    let b_size_ptr: *const ::std::os::raw::c_void = b_size_rptr as *const ::std::os::raw::c_void;
//    let b_window_rptr: *const usize = &b_window;
//    let b_window_ptr: *const ::std::os::raw::c_void =
//        b_window_rptr as *const ::std::os::raw::c_void;
//    let args: &mut [*const ::std::os::raw::c_void] = &mut [
//        a_ptr_ptr,
//        a_ptr_ptr,
//        a_size_ptr,
//        a_size_ptr,
//        a_size_ptr,
//        a_size_ptr,
//        a_window_ptr,
//        b_ptr_ptr,
//        b_ptr_ptr,
//        b_size_ptr,
//        b_size_ptr,
//        b_size_ptr,
//        b_size_ptr,
//        b_window_ptr,
//    ];
//
//    let func_name_cstr = std::ffi::CString::new("kernel_arith_wrapper").unwrap();
//    let func_name = func_name_cstr.as_ptr() as *const ::std::os::raw::c_char;
//    let args_ptr = args.as_mut_ptr() as *mut ::std::os::raw::c_void;
//
//    let mut res;
//    unsafe {
//        res = gpu_host::gpu_launch_kernel(
//            func_name,
//            1,
//            1,
//            1,
//            4,
//            1,
//            1,
//            0,
//            args_ptr,
//            core::ptr::null_mut(),
//        );
//    }
//
//    if res != 0 {
//        return res;
//    }
//
//    unsafe {
//        res = gpu_host::gpu_device_sync();
//    }
//
//    res
//}
*/
