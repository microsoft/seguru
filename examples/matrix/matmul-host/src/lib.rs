mod host;

use host::inner_product_kernel;

fn cpu_inner_product(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn print_square_matrix_side_by_side(m: &[f32], o: &[f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            print!("{:6.1} ", m[i * n + j]);
        }
        for j in 0..n {
            print!("{:6.1} ", o[i * n + j]);
        }
        println!();
    }
}

fn print_square_matrix(m: &[f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            print!("{:6.1} ", m[i * n + j]);
        }
        println!();
    }
}

pub fn run_host_matmul<'ctx>(
    ctx: &gpu_host::GpuCtxZeroGuard<'ctx, '_>,
    m: &'ctx gpu_host::GpuModule<gpu_host::CtxSpaceZero>,
    n: usize,
    dim: u32,
) -> Result<(), gpu_host::CudaError> {
    let mut initial_array = vec![];
    for i in 0..(n * n) {
        initial_array.push((i % 32) as f32);
    }

    let mut initial_c = vec![0.0; n * n];

    let h_a: &[f32] = &initial_array;
    let h_b: &[f32] = &initial_array;
    let h_c: &mut [f32] = &mut initial_c.clone();
    let h_c_cpu: &mut [f32] = &mut initial_c;

    let d_a = ctx.new_gmem_with_len::<f32>(n * n)?;
    let d_b = ctx.new_gmem_with_len::<f32>(n * n)?;
    let d_c = ctx.new_gmem_with_len::<f32>(n * n)?;
    use std::cmp::min;
    d_a.copy_from_host(h_a, min(h_a.len(), n * n), ctx)?;
    d_b.copy_from_host(h_b, min(h_b.len(), n * n), ctx)?;
    d_c.copy_from_host(h_c, min(h_c.len(), n * n), ctx)?;

    let d_c_c = gpu::GpuChunkableMut2D::<f32>::new(d_c, n);

    // Now do the kernel
    // block_dim_x * block_dim_y * block_dim_z must be less than or equal to 1024
    assert!(dim < 32 || dim % 32 == 0, "dim must be a multiple of 32 or less than 32");
    let grid_dim = if dim > 32 { dim / 32 } else { 1 };
    let block_dim = if dim > 32 { 32 } else { dim };
    let config = gpu_host::gpu_config!(grid_dim, grid_dim, 1, block_dim, block_dim, 1, 0);
    let start = std::time::Instant::now();
    inner_product_kernel(config, ctx, m, d_a, d_b, d_c_c, n).expect("Kernel execution failed");
    let elapsed = start.elapsed();
    println!("GPU execution time: {:?}", elapsed);

    d_c.copy_to_host(h_c, min(h_c.len(), n * n), ctx)?;

    // Perform CPU side validation
    println!("running on cpu...");
    let start = std::time::Instant::now();
    cpu_inner_product(h_a, h_b, h_c_cpu, n);
    let elapsed = start.elapsed();
    println!("CPU execution time: {:?}", elapsed);

    println!("validating...");
    for i in 0..h_c.len() {
        if h_c[i] != h_c_cpu[i] {
            print!("error: kernel result is wrong at {}. {} != {}", i, h_c_cpu[i], h_c[i]);
            if n <= 10 {
                println!("printing matrix...");
                println!("a and b:");
                print_square_matrix_side_by_side(h_a, h_b, n);
                println!("gpu:");
                print_square_matrix(h_c, n);
                println!("cpu:");
                print_square_matrix(h_c_cpu, n);
            } else {
                println!("not printing matrix because it's too big")
            }
            return Ok(());
        }
    }

    println!("matrix correct");

    Ok(())
}
