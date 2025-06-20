use std::env;
use std::process::ExitCode;
use std::str::FromStr;

// This wrapper should be generated automatically. Note that it should have known the
// kernel function's name!
fn kernel_launch_wrapper(a: &[u8], a_window: usize, b: &mut [u8], b_window: usize) -> i32 {
    // Build the args array
    let a_ptr = a.as_ptr() as *const ::std::os::raw::c_void;
    let a_ptr_rptr: *const *const ::std::os::raw::c_void = &a_ptr;
    let a_ptr_ptr = a_ptr_rptr as *const ::std::os::raw::c_void;
    let a_size = a.len();
    let a_size_rptr: *const usize = &a_size;
    let a_size_ptr: *const ::std::os::raw::c_void = a_size_rptr as *const ::std::os::raw::c_void;
    let a_window_rptr: *const usize = &a_window;
    let a_window_ptr: *const ::std::os::raw::c_void =
        a_window_rptr as *const ::std::os::raw::c_void;
    let b_ptr = b.as_ptr() as *const ::std::os::raw::c_void;
    let b_ptr_rptr: *const *const ::std::os::raw::c_void = &b_ptr;
    let b_ptr_ptr = b_ptr_rptr as *const ::std::os::raw::c_void;
    let b_size = b.len();
    let b_size_rptr: *const usize = &b_size;
    let b_size_ptr: *const ::std::os::raw::c_void = b_size_rptr as *const ::std::os::raw::c_void;
    let b_window_rptr: *const usize = &b_window;
    let b_window_ptr: *const ::std::os::raw::c_void =
        b_window_rptr as *const ::std::os::raw::c_void;
    let args: &mut [*const ::std::os::raw::c_void] = &mut [
        a_ptr_ptr,
        a_ptr_ptr,
        a_size_ptr,
        a_size_ptr,
        a_size_ptr,
        a_size_ptr,
        a_window_ptr,
        b_ptr_ptr,
        b_ptr_ptr,
        b_size_ptr,
        b_size_ptr,
        b_size_ptr,
        b_size_ptr,
        b_window_ptr,
    ];

    let func_name_cstr = std::ffi::CString::new("kernel_arith_wrapper").unwrap();
    let func_name = func_name_cstr.as_ptr() as *const ::std::os::raw::c_char;
    let args_ptr = args.as_mut_ptr() as *mut ::std::os::raw::c_void;

    let mut res;
    unsafe {
        res = cuda_bindings::gpu_launch_kernel(
            func_name,
            1,
            1,
            1,
            4,
            1,
            1,
            0,
            args_ptr,
            core::ptr::null_mut(),
        );
    }

    if res != 0 {
        return res;
    }

    unsafe {
        res = cuda_bindings::gpu_device_sync();
    }

    res
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let len: usize = 4;
    let window: usize = 1;
    let mut fake_len = len;

    if args.len() >= 2 {
        // Take fake length here
        fake_len = i32::from_str(&args[1]).unwrap() as usize;
        println!("{}: faking length as {}", args[0], fake_len);
    }

    // Temporary workaround. cuda_bindings should offer safe wrapper for these
    unsafe {
        if cuda_bindings::gpu_init() != 0 {
            eprintln!("{}: failed to initialise gpu", args[0]);
            return ExitCode::from(1);
        }

        if cuda_bindings::gpu_load_module() != 0 {
            eprintln!("{}: failed to load module", args[0]);
            return ExitCode::from(1);
        }
    }

    // Allocate the two host-side arrays
    // TODO: Make h_a non-mutable
    let h_a: &mut [u8] = &mut [1, 2, 3, 4];
    let h_b: &mut [u8] = &mut [5, 6, 7, 8];

    // Allocate the two device-side arrays and build them into slices
    let d_a;
    let d_b;
    let d_a_ptr;
    let d_b_ptr;
    unsafe {
        d_a_ptr = cuda_bindings::gpu_memalloc(len * std::mem::size_of::<u8>()) as *mut u8;
        d_b_ptr = cuda_bindings::gpu_memalloc(len * std::mem::size_of::<u8>()) as *mut u8;
        d_a = std::slice::from_raw_parts_mut(d_a_ptr, fake_len);
        d_b = std::slice::from_raw_parts_mut(d_b_ptr, fake_len);
    }

    // Copy host to device
    unsafe {
        if cuda_bindings::gpu_memcpy(
            d_a_ptr as *mut ::std::os::raw::c_void,
            h_a.as_mut_ptr() as *mut ::std::os::raw::c_void,
            len * std::mem::size_of::<u8>(),
            cuda_bindings::GPU_MEMCPY_H2D,
        ) != 0
        {
            eprintln!("{}: failed to copy a", args[0]);
            return ExitCode::from(1);
        }

        if cuda_bindings::gpu_memcpy(
            d_b_ptr as *mut ::std::os::raw::c_void,
            h_b.as_mut_ptr() as *mut ::std::os::raw::c_void,
            len * std::mem::size_of::<u8>(),
            cuda_bindings::GPU_MEMCPY_H2D,
        ) != 0
        {
            eprintln!("{}: failed to copy b", args[0]);
            return ExitCode::from(1);
        }
    }

    // Now do the kernel
    if kernel_launch_wrapper(d_a, window, d_b, window) != 0 {
        eprintln!("{}: failed to execute kernel", args[0]);
        return ExitCode::from(1);
    }

    // Copy back from device
    unsafe {
        if cuda_bindings::gpu_memcpy(
            h_b.as_mut_ptr() as *mut ::std::os::raw::c_void,
            d_b_ptr as *mut ::std::os::raw::c_void,
            len * std::mem::size_of::<u8>(),
            cuda_bindings::GPU_MEMCPY_D2H,
        ) != 0
        {
            eprintln!("{}: failed to copy b back", args[0]);
            return ExitCode::from(1);
        }
    }

    for (i, bi) in h_b.iter().enumerate().take(len) {
        println!("b[{}] = {}", i, bi);
    }

    ExitCode::SUCCESS
}
