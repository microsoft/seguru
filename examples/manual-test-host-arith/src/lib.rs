#![allow(clippy::too_many_arguments)]

use std::env;
use std::process::ExitCode;
use std::str::FromStr;

mod internal {
    /// Manually inserted function to test the host side API matches the GPU side API.
    /// This should be generated automatically by the macro.
    /// This is needed in order to force the compiler to link the GPU code.
    #[allow(dead_code)]
    fn dummy_api_checker_kernel_launch_wrapper(
        a: &[u32],
        a_window: usize,
        b: &mut [u32],
        b_window: usize,
        c: &[u32],
        f: &mut [f32],
        f_window: usize,
        g: &[f32],
    ) {
        manual_test_gpu_arith::kernel_arith(a, a_window, b, b_window, c, f, f_window, g);
    }
}

#[gpu_macros::host(manual_test_gpu_arith::kernel_arith)]
fn kernel_arith(
    a: &gpu::GpuChunkable2D<u32>,
    b: &gpu::GpuChunkableMut2D<u32>,
    c: &[u32],
    f: &gpu::GpuChunkableMut<f32>,
    g: &[f32],
) {
    let config = cuda_bindings::GPUConfig {
        grid_dim_x: 1,
        grid_dim_y: 1,
        grid_dim_z: 1,
        block_dim_x: 1,
        block_dim_y: 4,
        block_dim_z: 1,
    };
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
//        res = cuda_bindings::gpu_launch_kernel(
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
//        res = cuda_bindings::gpu_device_sync();
//    }
//
//    res
//}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut len: usize = 4;
    let window: usize = 1;

    if args.len() >= 2 {
        // Take length here
        len = i32::from_str(&args[1]).unwrap() as usize;
        println!("{}: length set to {}", args[0], len);
    }

    if cuda_bindings::init() != 0 {
        eprintln!("{}: failed to initialise gpu", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::load_module() != 0 {
        eprintln!("{}: failed to load module", args[0]);
        return ExitCode::from(1);
    }

    // Allocate the two host-side arrays
    // TODO: Make h_a non-mutable
    let h_a: &[u32] = &[1, 2, 3, 4];
    let h_b: &mut [u32] = &mut [5, 6, 7, 8];
    let h_c: &[u32] = &[10, 11, 12, 13];
    let h_f: &mut [f32] = &mut [0.0, 0.0, 0.0, 0.0];
    let h_g: &[f32] = &[1.1, 2.2, 3.3, 4.4];

    // Allocate the two device-side arrays and build them into slices
    let d_a;
    let d_b;
    let d_c;
    let d_f;
    let d_g;

    if let Some(dev_slice) = cuda_bindings::memalloc::<u32>(len * std::mem::size_of::<u32>()) {
        d_a = dev_slice;
    } else {
        eprintln!("{}: failed to allocate d_a", args[0]);
        return ExitCode::from(1);
    }

    if let Some(dev_slice) = cuda_bindings::memalloc::<u32>(len * std::mem::size_of::<u32>()) {
        d_b = dev_slice;
    } else {
        eprintln!("{}: failed to allocate d_b", args[0]);
        return ExitCode::from(1);
    }

    if let Some(dev_slice) = cuda_bindings::memalloc::<u32>(len * std::mem::size_of::<u32>()) {
        d_c = dev_slice;
    } else {
        eprintln!("{}: failed to allocate d_c", args[0]);
        return ExitCode::from(1);
    }

    if let Some(dev_slice) = cuda_bindings::memalloc::<f32>(len * std::mem::size_of::<f32>()) {
        d_f = dev_slice;
    } else {
        eprintln!("{}: failed to allocate d_f", args[0]);
        return ExitCode::from(1);
    }

    if let Some(dev_slice) = cuda_bindings::memalloc::<f32>(len * std::mem::size_of::<f32>()) {
        d_g = dev_slice;
    } else {
        eprintln!("{}: failed to allocate d_g", args[0]);
        return ExitCode::from(1);
    }

    // Copy host to device
    if cuda_bindings::memcpy(
        d_a,
        h_a,
        len * std::mem::size_of::<u32>(),
        cuda_bindings::GPU_MEMCPY_H2D,
    ) != 0
    {
        eprintln!("{}: failed to copy a", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::memcpy(
        d_b,
        h_b,
        len * std::mem::size_of::<u32>(),
        cuda_bindings::GPU_MEMCPY_H2D,
    ) != 0
    {
        eprintln!("{}: failed to copy b", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::memcpy(
        d_c,
        h_c,
        len * std::mem::size_of::<u32>(),
        cuda_bindings::GPU_MEMCPY_H2D,
    ) != 0
    {
        eprintln!("{}: failed to copy c", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::memcpy(
        d_f,
        h_f,
        len * std::mem::size_of::<f32>(),
        cuda_bindings::GPU_MEMCPY_H2D,
    ) != 0
    {
        eprintln!("{}: failed to copy f", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::memcpy(
        d_g,
        h_g,
        len * std::mem::size_of::<f32>(),
        cuda_bindings::GPU_MEMCPY_H2D,
    ) != 0
    {
        eprintln!("{}: failed to copy f", args[0]);
        return ExitCode::from(1);
    }

    let d_a_c = gpu::GpuChunkable2D::<u32> {
        slice: d_a,
        size_x: 1,
    };

    let d_b_c = gpu::GpuChunkableMut2D::<u32> {
        slice: d_b,
        size_x: 1,
    };

    let d_f_c = gpu::GpuChunkableMut::<f32> { slice: d_f, window };

    // Now do the kernel
    if launch_kernel_arith(&d_a_c, &d_b_c, d_c, &d_f_c, d_g) != 0 {
        eprintln!("{}: failed to execute kernel", args[0]);
        return ExitCode::from(1);
    }

    // Copy back from device
    if cuda_bindings::memcpy(
        h_b,
        d_b,
        len * std::mem::size_of::<u32>(),
        cuda_bindings::GPU_MEMCPY_D2H,
    ) != 0
    {
        eprintln!("{}: failed to copy b back", args[0]);
        return ExitCode::from(1);
    }

    if cuda_bindings::memcpy(
        h_f,
        d_f,
        len * std::mem::size_of::<f32>(),
        cuda_bindings::GPU_MEMCPY_D2H,
    ) != 0
    {
        eprintln!("{}: failed to copy f back", args[0]);
        return ExitCode::from(1);
    }

    for (i, bi) in h_b.iter().enumerate().take(len) {
        println!("b[{}] = {}", i, bi);
    }

    for (i, fi) in h_f.iter().enumerate().take(len) {
        println!("f[{}] = {}", i, fi);
    }

    ExitCode::SUCCESS
}
