mod host;

use host::launch_kernel_arith;

use std::env;
use std::process::ExitCode;
use std::str::FromStr;

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
